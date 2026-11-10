"""混合检索：向量召回 + 关键词(BM25) 召回 -> RRF 融合 -> LLM 重排 -> 同节邻居补全。

为什么必须加关键词通道：`GPIOx_CRL`、`MODE[1:0]`、`0x4444 4444` 这类寄存器符号，
向量模型几乎分不出区别（它们字面高度相似、语义都是「寄存器」），
只有字面匹配才能把它们排到第一。RRF 只用排名不用分数，两路分数量纲不同也没关系。
"""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .config import RetrieveConfig, get_llm
from .index import Stm32Index

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")

# 手册第 7 章覆盖 F101/102/103/104（Low-, medium-, high-, XL-density），
# 第 8 章覆盖带以太网/USB OTG 的 F105/107（Connectivity line）。
# 同一寄存器两章各有一份描述，按型号召回时优先用对应版本。
_DEVICE_HINTS = (
    (re.compile(r"f10[1-4]", re.IGNORECASE), "low-medium-high-xl-density"),
    (re.compile(r"f10[5-7]|connectivity", re.IGNORECASE), "connectivity-line"),
)


_BIT_QUERY_RE = re.compile(r"位域|位定义|哪些位|bit\s|作用|含义|怎么设置|如何配置|取值", re.IGNORECASE)
_MAP_QUERY_RE = re.compile(r"地址|偏移|offset|复位值|reset value|映射表|register map", re.IGNORECASE)
_FIGURE_QUERY_RE = re.compile(r"figure\s*(\d+)", re.IGNORECASE)


def named_figures(query: str) -> set[int]:
    """问题里点名了哪几张图，如“看 Figure 8 的时钟树” -> {8}。"""
    return {int(n) for n in _FIGURE_QUERY_RE.findall(query)}


def density_hint(query: str) -> str:
    """从问题里的型号推出应该看哪套章节，推不出就返回空。"""
    for pattern, density in _DEVICE_HINTS:
        if pattern.search(query):
            return density
    return ""


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        word = match.group()
        tokens.append(word)
        if "_" in word:  # GPIOx_CRL -> gpiox_crl + gpiox + crl
            tokens.extend(p for p in word.split("_") if p)
    return tokens


class BM25:
    """标准 Okapi BM25，带倒排表，1300 个块打分是毫秒级。"""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs_tokens)
        self.doc_len = [len(t) for t in docs_tokens]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, tokens in enumerate(docs_tokens):
            for term, freq in Counter(tokens).items():
                self.postings[term].append((idx, freq))
        self.idf = {
            term: math.log(1 + (self.n_docs - len(post) + 0.5) / (len(post) + 0.5))
            for term, post in self.postings.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        result = [0.0] * self.n_docs
        if not self.n_docs or not self.avgdl:
            return result
        for term in set(query_tokens):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for idx, freq in postings:
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                result[idx] += idf * freq * (self.k1 + 1) / denom
        return result

    def top_k(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        scored = self.scores(query_tokens)
        ranked = sorted(range(len(scored)), key=lambda i: scored[i], reverse=True)
        return [(i, scored[i]) for i in ranked[:k] if scored[i] > 0]


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict
    score: float = 0.0
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None
    expanded: bool = False

    @property
    def label(self) -> str:
        title = self.metadata.get("title", "")
        pages = self.metadata.get("pages", self.metadata.get("page", ""))
        return f"{title}｜p.{pages}"


@dataclass
class _Corpus:
    ids: list[str]
    texts: list[str]
    metas: list[dict]
    tokens: list[list[str]] = field(default_factory=list)
    by_id: dict[str, int] = field(default_factory=dict)
    by_section: dict[str, dict[int, int]] = field(default_factory=dict)


class CorpusStore:
    """把全量语料 + 分词结果缓存到磁盘，避免每次启动重新分词。"""

    def __init__(self, index: Stm32Index, cache_dir: Path | None = None) -> None:
        self.index = index
        self.cache_dir = Path(cache_dir or index.db_dir)
        self.cache_file = self.cache_dir / f"{index.name}.bm25.pkl"
        self._corpus: _Corpus | None = None

    def load(self) -> _Corpus:
        if self._corpus is not None:
            return self._corpus

        ids, texts, metas = self.index.corpus()
        if not ids:
            raise RuntimeError("向量库是空的，先运行 build。")
        tokens: list[list[str]] | None = None
        if self.cache_file.is_file():
            try:
                with self.cache_file.open("rb") as handle:
                    cached = pickle.load(handle)
                if cached.get("ids") == ids:
                    tokens = cached["tokens"]
            except Exception:
                tokens = None
        if tokens is None:
            tokens = [tokenize(t) for t in texts]
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                with self.cache_file.open("wb") as handle:
                    pickle.dump({"ids": ids, "tokens": tokens}, handle)
            except Exception:
                pass

        by_id = {doc_id: i for i, doc_id in enumerate(ids)}
        by_section: dict[str, dict[int, int]] = defaultdict(dict)
        for i, meta in enumerate(metas):
            by_section[str(meta.get("section_id", ""))][int(meta.get("chunk_index", 0))] = i

        self._corpus = _Corpus(
            ids=ids,
            texts=texts,
            metas=metas,
            tokens=tokens,
            by_id=by_id,
            by_section=dict(by_section),
        )
        return self._corpus


class HybridRetriever:
    def __init__(
        self,
        index: Stm32Index | None = None,
        cfg: RetrieveConfig | None = None,
        llm=None,
        store: CorpusStore | None = None,
    ) -> None:
        self.index = index or Stm32Index()
        self.cfg = cfg or RetrieveConfig()
        self._llm = llm
        self.store = store or CorpusStore(self.index)
        self._corpus: _Corpus | None = None
        self._bm25: BM25 | None = None

    # ------------------------------------------------------------ 准备
    def ensure_ready(self) -> _Corpus:
        if self._corpus is None:
            self._corpus = self.store.load()
            self._bm25 = BM25(self._corpus.tokens)
        return self._corpus

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    # ------------------------------------------------------------ 检索
    def _vector_hits(self, query: str, k: int) -> list[Hit]:
        corpus = self.ensure_ready()
        n = min(k, max(1, len(corpus.ids)))
        data = self.index.collection.query(
            query_embeddings=[self.index.embed_query(query)],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        for i, doc_id in enumerate(data["ids"][0]):
            distance = float(data["distances"][0][i])
            hits.append(
                Hit(
                    id=doc_id,
                    text=data["documents"][0][i] or "",
                    metadata=data["metadatas"][0][i] or {},
                    score=1.0 - distance,  # 余弦距离转相似度
                    vector_rank=i + 1,
                )
            )
        return hits

    def _bm25_hits(self, query: str, k: int) -> list[Hit]:
        corpus = self.ensure_ready()
        assert self._bm25 is not None
        hits: list[Hit] = []
        for rank, (idx, score) in enumerate(self._bm25.top_k(tokenize(query), k), start=1):
            hits.append(
                Hit(
                    id=corpus.ids[idx],
                    text=corpus.texts[idx],
                    metadata=corpus.metas[idx],
                    score=score,
                    bm25_rank=rank,
                )
            )
        return hits

    def _rrf(self, vector_hits: list[Hit], bm25_hits: list[Hit]) -> list[Hit]:
        fused: dict[str, Hit] = {}
        bonus: dict[str, float] = defaultdict(float)
        for rank, hit in enumerate(vector_hits, start=1):
            fused.setdefault(hit.id, hit)
            fused[hit.id].vector_rank = rank
            bonus[hit.id] += 1.0 / (self.cfg.rrf_k + rank)
        for rank, hit in enumerate(bm25_hits, start=1):
            existing = fused.setdefault(hit.id, hit)
            existing.bm25_rank = rank
            bonus[hit.id] += 1.0 / (self.cfg.rrf_k + rank)
        ranked = sorted(fused.values(), key=lambda h: bonus[h.id], reverse=True)
        for hit in ranked:
            hit.score = bonus[hit.id]
        return ranked

    def _rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        listing = []
        for i, hit in enumerate(hits, start=1):
            snippet = hit.text[:700].replace("\n", " ")
            listing.append(f"[{i}] {hit.label}\n{snippet}")
        prompt = (
            "你是检索结果重排器。判断每个候选片段对回答问题的有用程度，0-10 分。\n"
            "只输出 JSON，格式：{\"scores\": {\"1\": 8, \"2\": 3}}\n\n"
            f"问题：{query}\n\n候选片段：\n" + "\n\n".join(listing)
        )
        try:
            raw = self.llm.invoke(prompt).content
        except Exception:
            return hits
        scores: dict[int, float] = {}
        for key, value in re.findall(r'"(\d+)"\s*:\s*(\d+(?:\.\d+)?)', raw):
            scores[int(key)] = float(value)
        if not scores:
            return hits
        for i, hit in enumerate(hits, start=1):
            hit.rerank_score = scores.get(i, 0.0)
        return sorted(hits, key=lambda h: (h.rerank_score if h.rerank_score is not None else 0.0), reverse=True)

    def _expand(self, hits: list[Hit], budget_tokens: int) -> list[Hit]:
        """同一小节的相邻块补进来，避免答案正好被切在块边界上。"""
        if not self.cfg.expand_neighbors:
            return hits
        corpus = self.ensure_ready()
        present = {h.id for h in hits}
        result: list[Hit] = []
        used = 0
        for hit in hits:
            size = int(hit.metadata.get("tokens", 0) or 0)
            # 预算对主片段同样生效，否则两个 650 token 的块照样能绕过 1000 token 的上限
            if result and used + size > budget_tokens:
                break
            result.append(hit)
            used += size
            section_id = str(hit.metadata.get("section_id", ""))
            index_in_section = int(hit.metadata.get("chunk_index", 0))
            # 先补后一块：块边界通常把表格/列表截在后面，+1 比 -1 有用得多
            for offset in (1, -1):
                other = corpus.by_section.get(section_id, {}).get(index_in_section + offset)
                if other is None:
                    continue
                doc_id = corpus.ids[other]
                if doc_id in present:
                    continue
                size = int(corpus.metas[other].get("tokens", 0) or 0)
                if used + size > budget_tokens:
                    continue
                present.add(doc_id)
                used += size
                result.append(
                    Hit(
                        id=doc_id,
                        text=corpus.texts[other],
                        metadata=corpus.metas[other],
                        score=hit.score,
                        expanded=True,
                    )
                )
        return result

    def search(self, query: str, cfg: RetrieveConfig | None = None) -> list[Hit]:
        cfg = cfg or self.cfg
        self.ensure_ready()

        target = density_hint(query) if cfg.device_boost else ""

        vector_hits = self._vector_hits(query, cfg.vector_k)
        bm25_hits = self._bm25_hits(query, cfg.bm25_k)
        fused = self._rrf(vector_hits, bm25_hits)
        if target:
            fused = self._prefer_density(fused, target)
        fused = self._demote_register_maps(fused, query)
        figures = named_figures(query)
        if figures:
            fused = self._prefer_named_figures(fused, figures)

        pool = fused[: cfg.rerank_pool]
        if cfg.use_rerank:
            pool = self._rerank(query, pool)
            if target:  # 重排后仍把目标版本的片段排在前面
                pool.sort(key=lambda h: 0 if h.metadata.get("density") == target else 1)
            if figures:  # 点名的图也保持在前面
                pool = self._prefer_named_figures(pool, figures)
        pool = self._limit_figures(pool, cfg.max_figures_in_context, figures)

        # 同一小节先只留最相关的一块：否则一个长寄存器小节会占满 top-k，
        # 而它剩下的块本来就会被邻居补全拉进来
        top = self._collapse_sections(pool)[: cfg.final_k]
        expanded = self._expand(top, cfg.max_context_tokens)
        return self._limit_per_section(expanded, cfg.max_per_section)

    @staticmethod
    def _collapse_sections(hits: list[Hit]) -> list[Hit]:
        seen: set[str] = set()
        kept: list[Hit] = []
        for hit in hits:
            section_id = str(hit.metadata.get("section_id", ""))
            if section_id in seen:
                continue
            seen.add(section_id)
            kept.append(hit)
        return kept

    @staticmethod
    def _prefer_density(hits: list[Hit], target: str) -> list[Hit]:
        """稳定地目标密度章节的块提到前面，其余保持原有顺序兜底。"""
        matched = [h for h in hits if h.metadata.get("density") == target]
        if not matched:
            return hits
        return matched + [h for h in hits if h.metadata.get("density") != target]

    @staticmethod
    def _limit_figures(hits: list[Hit], limit: int, keep: set[int] | None = None) -> list[Hit]:
        """插图只占固定名额：否则 364 个图块会把寄存器题的正文证据挤出去。"""
        keep = keep or set()
        kept: list[Hit] = []
        figures_kept = 0
        for hit in hits:
            if hit.metadata.get("kind") == "figure":
                number = int(hit.metadata.get("figure_no", -1))
                if number not in keep:
                    if figures_kept >= limit:
                        continue
                    figures_kept += 1
            kept.append(hit)
        return kept

    @staticmethod
    def _prefer_named_figures(hits: list[Hit], numbers: set[int]) -> list[Hit]:
        """问题点名“Figure 8”时，把那张图排到最前。"""
        matched = [
            h
            for h in hits
            if h.metadata.get("kind") == "figure"
            and int(h.metadata.get("figure_no", -1)) in numbers
        ]
        if not matched:
            return hits
        matched_ids = {h.id for h in matched}
        return matched + [h for h in hits if h.id not in matched_ids]

    @staticmethod
    def _demote_register_maps(hits: list[Hit], query: str) -> list[Hit]:
        """问位域含义时，把“只列地址/偏移的寄存器映射表”排到后面。

        这些表能客地址类问题，但对“某位是干什么的”几乎无用，却常常因为
        标题里带着寄存器名而被向量召回排到前面。
        """
        if not _BIT_QUERY_RE.search(query) or _MAP_QUERY_RE.search(query):
            return hits
        return sorted(hits, key=lambda h: 1 if "register map" in str(h.metadata.get("title", "")).lower() else 0)

    def _limit_per_section(self, hits: list[Hit], limit: int) -> list[Hit]:
        """同一个小节最多留 limit 块。长小节常被切 3~5 块，全带进 Prompt 会把名额挤占掉。"""
        if limit <= 0:
            return hits
        used: Counter[str] = Counter()
        kept: list[Hit] = []
        for hit in hits:
            section_id = str(hit.metadata.get("section_id", ""))
            if used[section_id] >= limit:
                continue
            used[section_id] += 1
            kept.append(hit)
        return kept
