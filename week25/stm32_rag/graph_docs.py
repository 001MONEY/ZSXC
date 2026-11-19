"""文档知识图谱（GraphRAG 的数据层）：Chroma 里每个块的层级 / 符号 / 邻接关系。

与 `graph.py` 的分工：

| | `graph.py` 器件事实图 | 本模块 文档关系图 |
| --- | --- | --- |
| 数据来源 | `mcu.py` 静态表 | Chroma 的 chunk metadata |
| 节点 | Device / Pin / Timer / Usart / DmaChannel / Bus / Irq | Doc / Chapter / Section / Chunk / Symbol |
| 回答 | 「PA8 只能是 TIM1_CH1」这类确定事实 | 「这个符号在哪讲、谁定义了它、哪块和它相关」 |

**不需要 LLM 抽实体**：建库时 metadata 里已经有 `symbols`（分号分隔）/`symbol`/`chapter`/
`section_id`/`chunk_index`/`page`，直接读一遍就是图。chunk id 与 Chroma id 完全一致
（PDF 是 `{section_id}#{i}`，HAL 是 `{doc}:{path}:{line}#{i}`），所以图上查到的 id 能直接回查文本。

建图与查询：

    python -m stm32_rag graph-docs --build
    python -m stm32_rag graph-docs --stats
    python -m stm32_rag graph-docs --symbol HAL_GPIO_Init
    python -m stm32_rag graph-docs --chunk "rm:..."
    python -m stm32_rag graph-docs --expand <chunk-id>
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .config import DB_DIR, NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, SOURCE_COLLECTIONS
from .graph import _WRITE_WORDS  # 复用只读查询守卫，两个图的写保护口径保持一致

DOC_LABELS = ("Doc", "Chapter", "Section", "Chunk", "Symbol", "Peripheral")
DOC_RELS = (
    "IN_DOC",
    "IN_CHAPTER",
    "IN_SECTION",
    "MENTIONS",
    "DEFINES",
    "CALLS",
    "USES_PERIPHERAL",
    "SAME_AS",
    "NEXT",
)

# 建立索引时用来给“跨语料扩展”排序：UM 是 HAL 的说明书，优先于源码与例程
SOURCE_PRIORITY = {"um": 0, "rm": 1, "examples": 2, "hal": 3, "ds": 4}

# metadata 里 doc 字段缺省时按来源补一个
SOURCE_DOC = {"rm": "RM0008", "ds": "DS", "um": "UM1850", "hal": "HAL", "examples": "HAL Examples"}

DEFINE_KINDS = {"function", "typedef", "macro", "define"}

_SYMBOL_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{2,}$")

# 函数体里的调用：HAL_xxx( / LL_xxx( / __HAL_xxx(
_CALL_RE = re.compile(r"\b((?:__)?(?:HAL|LL)_[A-Za-z0-9_]+)\s*\(")
# 这些是 C 语言关键字/常用宏，不算调用关系
_CALL_SKIP = {"if", "for", "while", "switch", "sizeof", "return"}

# 外设名：用来把文档图接到器件图上（TIM1 / USART1 / GPIOA / DMA1_Channel4 ...）
_PERIPHERAL_RE = re.compile(
    r"\b(TIM\d{1,2}|USART\d|UART\d|SPI\d|I2C\d|ADC\d|GPIO[A-G]|DMA[12]_Channel\d)\b"
)


@dataclass
class DocGraphResult:
    """建图结果摘要。"""

    nodes: dict[str, int]
    relationships: dict[str, int]
    collections: dict[str, int]
    skipped: list[str]

    def summary(self) -> str:
        node_text = "、".join(f"{label} {count}" for label, count in self.nodes.items() if count)
        rel_text = "、".join(f"{name} {count}" for name, count in self.relationships.items() if count)
        coll_text = "、".join(f"{name} {count}" for name, count in self.collections.items())
        lines = [f"节点：{node_text or '无'}", f"关系：{rel_text or '无'}", f"语料：{coll_text}"]
        if self.skipped:
            lines.append("跳过（未建库的集合）：" + "、".join(self.skipped))
        return "\n".join(lines)


class DocGraphStore:
    """Neo4j 连接 + 文档图建图 + 多跳查询。"""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        timeout: float | None = None,
    ) -> None:
        try:
            from neo4j import GraphDatabase  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - 依赖缺失
            raise RuntimeError(
                "缺少 neo4j 驱动：uv pip install --python <venv>/Scripts/python.exe neo4j"
            ) from exc
        self.uri = uri or NEO4J_URI
        self.database = database or NEO4J_DATABASE
        self._driver = GraphDatabase.driver(
            self.uri,
            auth=(user or NEO4J_USER, password or NEO4J_PASSWORD),
            **(({"connection_timeout": timeout}) if timeout else {}),
        )

    # ------------------------------------------------------------ 连接
    def ping(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> DocGraphStore:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def run(self, cypher: str, params: dict | None = None, limit: int = 200) -> list[dict]:
        if _WRITE_WORDS.search(cypher):
            raise ValueError("只允许只读查询：CREATE/MERGE/DELETE/SET 等写操作已被拒绝")
        with self._driver.session(database=self.database) as session:
            rows = []
            for i, record in enumerate(session.run(cypher, params or {})):
                if i >= limit:
                    break
                rows.append(dict(record))
            return rows

    # ------------------------------------------------------------ 建图
    def build(self, reset: bool = False) -> DocGraphResult:
        """从 Chroma 的 metadata 生成文档图；`MERGE` 幂等，边每次全量重建。"""
        from .index import shared_client

        client = shared_client(DB_DIR)
        existing = {c.name for c in client.list_collections()}

        chunk_rows: list[dict] = []
        chapter_rows: list[dict] = []
        section_rows: list[dict] = []
        doc_rows: list[dict] = []
        mention_rows: list[dict] = []
        define_rows: list[dict] = []
        next_rows: list[dict] = []
        call_counter: Counter[tuple[str, str]] = Counter()
        symbol_df: Counter[str] = Counter()
        peripheral_rows: list[dict] = []
        symbol_peripheral: Counter[tuple[str, str]] = Counter()
        peripherals_seen: set[str] = set()
        collections: dict[str, int] = {}
        skipped: list[str] = []

        for source, collection in SOURCE_COLLECTIONS.items():
            if collection not in existing:
                skipped.append(f"{source}({collection})")
                continue
            data = client.get_collection(collection).get(include=["metadatas", "documents"])
            ids = data["ids"] or []
            metas = data["metadatas"] or []
            texts = data["documents"] or []
            collections[source] = len(ids)

            default_doc = SOURCE_DOC.get(source, source)
            chapters: set[str] = set()
            sections: set[str] = set()
            per_section: dict[str, list[tuple[int, str]]] = defaultdict(list)

            for chunk_id, meta, text in zip(ids, metas, texts):
                doc = str(meta.get("doc") or default_doc)
                chapter = str(meta.get("chapter") or "")
                section_id = str(meta.get("section_id") or "")
                chapter_key = f"{doc}|{chapter}" if chapter else ""
                section_key = f"{doc}|{section_id}" if section_id else ""
                chunk_index = int(meta.get("chunk_index") or 0)
                kind = str(meta.get("kind") or "")
                # RM0008 与 DS 会出现相同的小节号（如 5.1.1#0），图里必须带上来源才是唯一主键；
                # HAL/UM 的 id 本身已带来源前缀，就不再重复叠加
                node_id = chunk_id if chunk_id.startswith(f"{source}:") else f"{source}:{chunk_id}"

                chunk_rows.append(
                    {
                        "id": node_id,
                        "chroma_id": chunk_id,
                        "source": source,
                        "doc": doc,
                        "kind": kind,
                        "title": str(meta.get("title") or ""),
                        "page": int(meta.get("page") or 0),
                        "tokens": int(meta.get("tokens") or 0),
                        "chunk_index": chunk_index,
                        "chapter_key": chapter_key,
                        "section_key": section_key,
                        "preview": (str(meta.get("symbols") or meta.get("symbol") or ""))[:120],
                    }
                )
                if chapter_key and chapter_key not in chapters:
                    chapters.add(chapter_key)
                    chapter_rows.append({"key": chapter_key, "doc": doc, "name": chapter})
                if section_key and section_key not in sections:
                    sections.add(section_key)
                    section_rows.append(
                        {"key": section_key, "doc": doc, "name": section_id, "chapter": chapter}
                    )
                if section_key:
                    per_section[section_key].append((chunk_index, node_id))

                # 块里提到了哪些外设：正文 + 标题（标题里常带 TIM1_CH1 这类信息）
                found = set(_PERIPHERAL_RE.findall(f"{meta.get('title') or ''} {text or ''}"))
                for peripheral in list(found)[:12]:
                    peripherals_seen.add(peripheral)
                    peripheral_rows.append({"chunk": node_id, "peripheral": peripheral})
                    if single:
                        symbol_peripheral[(str(single), peripheral)] += 1

                named = [item for item in str(meta.get("symbols") or "").split(";") if item]
                single = meta.get("symbol")
                if single and single not in named:
                    named.append(str(single))
                named = [item for item in dict.fromkeys(named) if _SYMBOL_OK.match(item)]
                for symbol in named:
                    mention_rows.append({"chunk": node_id, "symbol": symbol})
                    symbol_df[symbol] += 1
                if kind in DEFINE_KINDS and single and _SYMBOL_OK.match(str(single)):
                    define_rows.append({"chunk": node_id, "symbol": str(single)})
                    # 函数体里调用了谁（只扫 HAL / 例程的函数块，PDF 段落不扫）
                    if kind == "function" and source in ("hal", "examples") and text:
                        for callee in _CALL_RE.findall(text):
                            if callee != single and callee not in _CALL_SKIP:
                                call_counter[(str(single), callee)] += 1

            for items in per_section.values():
                items.sort()
                for (_, left), (_, right) in zip(items, items[1:]):
                    next_rows.append({"a": left, "b": right})

        for doc in {row["doc"] for row in chunk_rows}:
            doc_rows.append({"name": doc})

        with self._driver.session(database=self.database) as session:
            if reset:
                for label in DOC_LABELS:
                    session.run(f"MATCH (n:{label}) DETACH DELETE n")
            for label in DOC_LABELS:
                session.run(
                    f"CREATE CONSTRAINT {label.lower()}_doc_name IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{'id' if label == 'Chunk' else 'key' if label in ('Chapter', 'Section') else 'name'} IS UNIQUE"
                )
            # 边全量重建：语料重建后 chunk id 会变，残留旧边会让扩展指到不存在的块
            session.run(
                f"MATCH ()-[r:{'|'.join(DOC_RELS)}]->() DELETE r"
            )

            self._batched(session, "UNWIND $rows AS row MERGE (d:Doc {name: row.name})", doc_rows)
            self._batched(
                session,
                "UNWIND $rows AS row MERGE (c:Chapter {key: row.key}) SET c.doc = row.doc, c.name = row.name",
                chapter_rows,
            )
            self._batched(
                session,
                "UNWIND $rows AS row MERGE (s:Section {key: row.key}) "
                "SET s.doc = row.doc, s.name = row.name, s.chapter = row.chapter",
                section_rows,
            )
            self._batched(
                session,
                "UNWIND $rows AS row MERGE (c:Chunk {id: row.id}) "
                "SET c.chroma_id = row.chroma_id, c.source = row.source, c.doc = row.doc, c.kind = row.kind, "
                "c.title = row.title, c.page = row.page, c.tokens = row.tokens, c.chunk_index = row.chunk_index, "
                "c.preview = row.preview",
                chunk_rows,
            )
            self._batched(
                session,
                "UNWIND $rows AS row MERGE (s:Symbol {name: row.name}) SET s.df = row.df",
                [{"name": name, "df": count} for name, count in symbol_df.items()],
            )

            self._batched(
                session,
                "UNWIND $rows AS row MATCH (c:Chunk {id: row.id}) MATCH (d:Doc {name: row.doc}) "
                "MERGE (c)-[:IN_DOC]->(d)",
                chunk_rows,
            )
            self._batched(
                session,
                "UNWIND $rows AS row MATCH (c:Chunk {id: row.id}) MATCH (ch:Chapter {key: row.chapter_key}) "
                "MERGE (c)-[:IN_CHAPTER]->(ch)",
                [row for row in chunk_rows if row["chapter_key"]],
            )
            self._batched(
                session,
                "UNWIND $rows AS row MATCH (c:Chunk {id: row.id}) MATCH (s:Section {key: row.section_key}) "
                "MERGE (c)-[:IN_SECTION]->(s)",
                [row for row in chunk_rows if row["section_key"]],
            )
            self._batched(
                session,
                "UNWIND $rows AS row MATCH (c:Chunk {id: row.chunk}) MATCH (s:Symbol {name: row.symbol}) "
                "MERGE (c)-[:MENTIONS]->(s)",
                mention_rows,
            )
            self._batched(
                session,
                "UNWIND $rows AS row MATCH (c:Chunk {id: row.chunk}) MATCH (s:Symbol {name: row.symbol}) "
                "MERGE (c)-[:DEFINES]->(s)",
                define_rows,
            )
            # 只保留两端都在语料里的调用关系，避免悬挂节点
            call_rows = [
                {"caller": caller, "callee": callee, "count": count}
                for (caller, callee), count in call_counter.items()
                if caller in symbol_df and callee in symbol_df
            ]
            self._batched(
                session,
                "UNWIND $rows AS row MATCH (a:Symbol {name: row.caller}) MATCH (b:Symbol {name: row.callee}) "
                "MERGE (a)-[r:CALLS]->(b) SET r.count = row.count",
                call_rows,
            )
            self._batched(
                session,
                "UNWIND $rows AS row MATCH (a:Chunk {id: row.a}) MATCH (b:Chunk {id: row.b}) "
                "MERGE (a)-[:NEXT]->(b)",
                next_rows,
            )

            # ---- 外设节点：把文档图与器件事实图接到一起 ----
            self._batched(
                session,
                "UNWIND $rows AS row MERGE (p:Peripheral {name: row.name})",
                [{"name": name} for name in sorted(peripherals_seen)],
            )
            self._batched(
                session,
                "UNWIND $rows AS row MATCH (c:Chunk {id: row.chunk}) MATCH (p:Peripheral {name: row.peripheral}) "
                "MERGE (c)-[:USES_PERIPHERAL]->(p)",
                peripheral_rows,
            )
            self._batched(
                session,
                "UNWIND $rows AS row MATCH (s:Symbol {name: row.symbol}) MATCH (p:Peripheral {name: row.peripheral}) "
                "MERGE (s)-[:USES_PERIPHERAL]->(p)",
                [
                    {"symbol": symbol, "peripheral": peripheral}
                    for (symbol, peripheral) in symbol_peripheral
                ],
            )
            # 同名外设（TIM1/USART1/DMA1_Channel4）与器件事实图的节点对齐，供跨图查询
            for label in ("Timer", "Usart", "DmaChannel"):
                session.run(
                    f"MATCH (p:Peripheral) MATCH (n:{label}) WHERE n.name = p.name MERGE (p)-[:SAME_AS]->(n)"
                )
            counts = self.stats()
        return DocGraphResult(
            nodes=counts["nodes"],
            relationships=counts["relationships"],
            collections=collections,
            skipped=skipped,
        )

    @staticmethod
    def _batched(session, cypher: str, rows: list[dict], size: int = 2000) -> None:
        for start in range(0, len(rows), size):
            session.run(cypher, rows=rows[start : start + size])

    def reset(self) -> None:
        """只删文档图的节点（不会动器件事实图）。"""
        with self._driver.session(database=self.database) as session:
            for label in DOC_LABELS:
                session.run(f"MATCH (n:{label}) DETACH DELETE n")

    def stats(self) -> dict:
        with self._driver.session(database=self.database) as session:
            nodes = {
                label: session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
                for label in DOC_LABELS
            }
            rels = {
                row["name"]: row["c"]
                for row in session.run(
                    f"MATCH ()-[r:{'|'.join(DOC_RELS)}]->() RETURN type(r) AS name, count(r) AS c"
                )
            }
        return {"nodes": nodes, "relationships": rels}

    # ------------------------------------------------------------ 查询
    def symbol_chunks(self, name: str, limit: int = 12) -> list[dict]:
        """一个符号在哪些块里出现/被定义，按来源优先级排序。"""
        rows = self.run(
            """
            MATCH (s:Symbol {name: $name})<-[r:MENTIONS|DEFINES]-(c:Chunk)
            RETURN c.id AS id, c.chroma_id AS chroma_id, c.source AS source, c.doc AS doc, c.kind AS kind,
                   c.title AS title, c.page AS page, type(r) AS rel
            """,
            {"name": name},
            limit=limit * 4,
        )
        rows.sort(key=lambda row: (SOURCE_PRIORITY.get(row["source"], 9), row["rel"] != "DEFINES", row["page"]))
        return rows[:limit]

    def expand_from(
        self,
        seed_ids: list[str],
        per_symbol: int = 2,
        limit: int = 6,
        max_df: int = 60,
    ) -> list[dict]:
        """种子块（Chroma chunk id）→ 共同符号 → 其它块（跨语料）。

        `max_df` 过滤掉「GPIO」这类到处出现的泛符号，否则一扩散就把上下文占满。
        """
        if not seed_ids:
            return []
        rows = self.run(
            """
            MATCH (seed:Chunk) WHERE seed.chroma_id IN $seeds
            MATCH (seed)-[:MENTIONS|DEFINES]->(s:Symbol)
            WHERE s.df <= $max_df
            MATCH (s)<-[:MENTIONS|DEFINES]-(other:Chunk)
            WHERE NOT other.chroma_id IN $seeds
            RETURN s.name AS symbol, s.df AS df, other.chroma_id AS chroma_id, other.source AS source,
                   other.title AS title, other.page AS page
            """,
            {"seeds": seed_ids, "max_df": max_df},
            limit=400,
        )

        seen: set[tuple[str, str, str]] = set()
        per: Counter[str] = Counter()
        picked: list[dict] = []
        rows.sort(key=lambda row: (SOURCE_PRIORITY.get(row["source"], 9), row["df"], row["chroma_id"]))
        for row in rows:
            key = (row["source"], row["chroma_id"], row["symbol"])
            if key in seen or per[row["symbol"]] >= per_symbol:
                continue
            seen.add(key)
            per[row["symbol"]] += 1
            picked.append(row)
            if len(picked) >= limit:
                break
        return picked

    def chunk_info(self, chunk_id: str) -> dict | None:
        rows = self.run(
            """
            MATCH (c:Chunk) WHERE c.id = $id OR c.chroma_id = $id
            OPTIONAL MATCH (c)-[:MENTIONS]->(s:Symbol)
            OPTIONAL MATCH (c)-[:IN_SECTION]->(sec:Section)
            OPTIONAL MATCH (c)-[:IN_CHAPTER]->(ch:Chapter)
            RETURN c.id AS id, c.chroma_id AS chroma_id, c.source AS source, c.doc AS doc, c.kind AS kind,
                   c.title AS title, c.page AS page, c.tokens AS tokens,
                   [x IN collect(DISTINCT s.name) WHERE x IS NOT NULL] AS symbols,
                   [x IN collect(DISTINCT sec.name) WHERE x IS NOT NULL][0] AS section,
                   [x IN collect(DISTINCT ch.name) WHERE x IS NOT NULL][0] AS chapter
            """,
            {"id": chunk_id},
        )
        return rows[0] if rows else None

    def neighbors(self, chunk_id: str, limit: int = 6) -> list[dict]:
        rows = self.run(
            """
            MATCH (c:Chunk)-[:NEXT]-(n:Chunk) WHERE c.id = $id OR c.chroma_id = $id
            RETURN DISTINCT n.id AS id, n.chroma_id AS chroma_id, n.source AS source,
                   n.title AS title, n.page AS page
            LIMIT $limit
            """,
            {"id": chunk_id, "limit": limit},
        )
        return rows

    def peripheral_chunks(self, name: str, limit: int = 10) -> list[dict]:
        """哪些块提到了这个外设（按来源优先级 + 页码排序）。"""
        rows = self.run(
            """
            MATCH (c:Chunk)-[:USES_PERIPHERAL]->(p:Peripheral {name: $name})
            RETURN c.chroma_id AS chroma_id, c.source AS source, c.doc AS doc, c.kind AS kind,
                   c.title AS title, c.page AS page
            """,
            {"name": name.upper()},
            limit=limit * 8,
        )
        rows.sort(key=lambda row: (SOURCE_PRIORITY.get(row["source"], 9), row["page"] or 0, row["chroma_id"]))
        return rows[:limit]

    def peripheral_link(self, name: str) -> str:
        """跨图查询：外设在器件事实图里的信息（总线 / 通道引脚 / DMA / 中断）。"""
        rows = self.run(
            """
            MATCH (p:Peripheral {name: $name})-[:SAME_AS]->(n)
            OPTIONAL MATCH (pin:Pin)-[tc:TIM_CH]->(n)
            OPTIONAL MATCH (n)-[dm:DMA]->(dc:DmaChannel)
            OPTIONAL MATCH (n)-[:CLOCKED_BY]->(b:Bus)
            OPTIONAL MATCH (n)-[:IRQ]->(i:Irq)
            RETURN labels(n)[0] AS label, n.name AS name,
                   [x IN collect(DISTINCT b.name) WHERE x IS NOT NULL][0] AS bus,
                   [x IN collect(DISTINCT pin.name + '_' + tc.channel) WHERE x IS NOT NULL] AS channels,
                   [x IN collect(DISTINCT dc.name) WHERE x IS NOT NULL] AS dma,
                   [x IN collect(DISTINCT i.name) WHERE x IS NOT NULL] AS irq
            """,
            {"name": name.upper()},
        )
        if not rows:
            return ""
        row = rows[0]
        parts = [f"器件图：{row['label']} {row['name']}" + (f"，挂在 {row['bus']}" if row["bus"] else "")]
        if row["channels"]:
            parts.append("通道 " + "、".join(row["channels"]))
        if row["dma"]:
            parts.append("DMA " + "、".join(row["dma"]))
        if row["irq"]:
            parts.append("中断 " + "、".join(row["irq"]))
        return "，".join(parts)

    # ------------------------------------------------------------ 文本
    def call_graph(self, name: str, depth: int = 2, limit: int = 20, incoming: bool = False) -> list[dict]:
        """调用链：默认「我调用了谁」，`incoming=True` 则「谁调用了我」。"""
        depth = max(1, min(int(depth), 3))  # 长度不能参数化，必须拼进语句：先限幅
        pattern = f"(g:Symbol)-[:CALLS*1..{depth}]->(f:Symbol {{name: $name}})" if incoming else f"(f:Symbol {{name: $name}})-[:CALLS*1..{depth}]->(g:Symbol)"
        rows = self.run(
            f"""
            MATCH path = {pattern}
            WITH g, min(length(path)) AS hops
            OPTIONAL MATCH (g)<-[:DEFINES]-(c:Chunk)
            RETURN g.name AS name, hops,
                   [x IN collect(DISTINCT c.chroma_id) WHERE x IS NOT NULL][0] AS chunk,
                   [x IN collect(DISTINCT c.page) WHERE x IS NOT NULL][0] AS page
            ORDER BY hops, name
            LIMIT $limit
            """,
            {"name": name, "limit": limit},
        )
        return rows

    def describe(self, kind: str, target: str = "") -> str:
        kind = (kind or "").strip().lower()
        if kind == "stats":
            counts = self.stats()
            lines = ["节点：" + "、".join(f"{k} {v}" for k, v in counts["nodes"].items() if v)]
            lines.append("关系：" + "、".join(f"{k} {v}" for k, v in counts["relationships"].items() if v))
            return "\n".join(lines)

        if kind == "symbol":
            rows = self.symbol_chunks(target, limit=12)
            if not rows:
                return f"图谱里没有符号 {target}（未建图或符号名不对）"
            lines = [f"{target} 出现在 {len(rows)} 个块："]
            lines.extend(
                f"- [{row['source']}] {row['title'] or row['kind']}"
                + (f" p.{row['page']}" if row["page"] else "")
                + ("（定义）" if row["rel"] == "DEFINES" else "")
                for row in rows
            )
            return "\n".join(lines)

        if kind == "chunk":
            facts = self.chunk_info(target)
            if facts is None:
                return f"图谱里没有块 {target}"
            lines = [
                f"{facts['id']}",
                f"- 来源：{facts['source']} / {facts['doc']} / {facts['kind']}"
                + (f" / p.{facts['page']}" if facts["page"] else ""),
                f"- 标题：{facts['title']}",
                f"- 小节：{facts['chapter']} → {facts['section']}",
                f"- 符号：{'、'.join(facts['symbols']) or '无'}",
            ]
            neighbors = self.neighbors(target, limit=4)
            if neighbors:
                lines.append("- 相邻块：" + "、".join(f"[{row['source']}] {row['title']}" for row in neighbors))
            return "\n".join(lines)

        if kind == "expand":
            rows = self.expand_from([target], per_symbol=3, limit=10)
            if not rows:
                return "没有找到可扩展的邻域（符号太泛或种子块没有符号）"
            return "\n".join(f"- 经由 {row['symbol']}（df={row['df']}）→ [{row['source']}] {row['title']}" for row in rows)

        if kind in ("calls", "callers"):
            rows = self.call_graph(target, depth=2, limit=20, incoming=kind == "callers")
            if not rows:
                return f"图谱里没有 {target} 的{'上游调用者' if kind == 'callers' else '下游调用'}（或该符号不在 HAL/例程里）"
            title = f"谁调用了 {target}（反向）" if kind == "callers" else f"{target} 调用链（2 跳）"
            lines = [title + "："]
            lines.extend(
                f"- {'←' if kind == 'callers' else '→'} {row['name']}"
                + (f"（{row['hops']} 跳" + (f"，定义于 p.{row['page']}" if row["page"] else "") + "）" if row["hops"] else "")
                for row in rows
            )
            return "\n".join(lines)

        if kind == "peripheral":
            name = target.upper()
            rows = self.peripheral_chunks(name, limit=10)
            link = self.peripheral_link(name)
            if not rows and not link:
                return f"图谱里没有外设 {name}（或该名字没在语料里出现）"
            lines = [f"外设 {name}"]
            if link:
                lines.append("- " + link)
            if rows:
                lines.append(f"- 提到它的块（前 {len(rows)} 个）：")
                lines.extend(
                    f"  · [{row['source']}] {row['title'] or row['kind']}"
                    + (f" p.{row['page']}" if row["page"] else "")
                    for row in rows
                )
            return "\n".join(lines)

        if kind == "cypher":
            rows = self.run(target, limit=50)
            return "\n".join(str(row) for row in rows) or "查询没有返回结果"

        return f"未知的查询类型：{kind}"


# ---------------------------------------------------------------- 单例

_STORE: DocGraphStore | None = None
_PINGED: bool | None = None


def doc_graph(ping: bool = True) -> DocGraphStore | None:
    """给检索层用的单例；Neo4j 不可用时返回 None（图谱是增强不是依赖）。"""
    global _STORE, _PINGED
    if _STORE is None:
        try:
            _STORE = DocGraphStore(timeout=2.0)
        except RuntimeError:
            return None
    if ping and _PINGED is None:
        _PINGED = _STORE.ping()
    if ping and not _PINGED:
        return None
    return _STORE


def doc_graph_available() -> bool:
    return doc_graph(ping=True) is not None
