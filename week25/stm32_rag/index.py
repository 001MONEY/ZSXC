"""Chroma 建库与读取。

用原生 `chromadb.PersistentClient` + 显式传入向量（和串讲里 6.1 节一致），
好处是：批量写入可控、失败可重试、已写入的 id 自动跳过，断点续跑不会重复计费。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chromadb
from langchain_core.documents import Document

from .chunker import build_documents
from .config import (
    COLLECTION_NAME,
    DB_DIR,
    EMBED_DIM,
    PDF_PATH,
    BuildConfig,
    get_embeddings,
)
from .figures import extract_figures, figure_documents

EMBED_GROUP = 200  # 每个线程一次提交给 embed_documents 的文本条数

# 同一路径只保留一个 PersistentClient：多实例并发访问同一个 chroma 目录时，
# Rust 后端会报 “Error creating hnsw segment reader: Nothing found on disk”。
_CLIENTS: dict[str, "chromadb.PersistentClient"] = {}


def shared_client(db_dir: Path):
    key = str(Path(db_dir).resolve()).lower()
    if key not in _CLIENTS:
        _CLIENTS[key] = chromadb.PersistentClient(path=key)
    return _CLIENTS[key]


def _sanitize(metadata: dict) -> dict:
    """Chroma 只接受 str/int/float/bool。"""
    clean: dict = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif value is None:
            continue
        else:
            clean[key] = str(value)
    return clean


class Stm32Index:
    def __init__(
        self,
        collection: str = COLLECTION_NAME,
        db_dir: Path = DB_DIR,
        embeddings=None,
    ) -> None:
        self.name = collection
        self.db_dir = Path(db_dir)
        self.embeddings = embeddings or get_embeddings()
        self._client = shared_client(self.db_dir)

    # ------------------------------------------------------------ 集合
    @property
    def collection(self):
        return self._client.get_or_create_collection(
            name=self.name, metadata={"hnsw:space": "cosine"}
        )

    def exists(self) -> bool:
        return self.name in {c.name for c in self._client.list_collections()}

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.name)
        except Exception:
            pass

    def _swap(self, work_name: str) -> None:
        """用临时集合顶替正式集合：删旧的，再把临时集合改名过去。"""
        self.reset()
        self._client.get_collection(work_name).modify(name=self.name)

    # ------------------------------------------------------------ 嵌入
    def _embed_group(self, texts: list[str], retries: int = 3) -> list[list[float]]:
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return self.embeddings.embed_documents(texts)
            except Exception as exc:  # 网络抖动 / 限流
                last = exc
                time.sleep(2.0 * attempt)
        raise RuntimeError(f"嵌入失败（重试 {retries} 次）：{last}")

    def embed_texts(self, texts: list[str], workers: int = 4, quiet: bool = False) -> list[list[float]]:
        groups = [texts[i : i + EMBED_GROUP] for i in range(0, len(texts), EMBED_GROUP)]
        vectors: list[list[float]] = [[] for _ in texts]
        offsets = [i for i in range(0, len(texts), EMBED_GROUP)]
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(self._embed_group, group): idx for idx, group in enumerate(groups)}
            for future in as_completed(futures):
                idx = futures[future]
                vectors[offsets[idx] : offsets[idx] + len(groups[idx])] = future.result()
                done += len(groups[idx])
                if not quiet:
                    print(f"\r  向量化 {done}/{len(texts)}", end="", flush=True)
        if not quiet:
            print()
        bad = next((v for v in vectors if len(v) != EMBED_DIM), None)
        if bad is not None:
            raise RuntimeError(
                f"嵌入维度异常：期望 {EMBED_DIM}，实际 {len(bad)}。"
                "请检查 STM32_EMBED_MODEL / STM32_EMBED_DIM 是否与已有集合一致。"
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)

    # ------------------------------------------------------------ 建库
    def build(
        self,
        pdf_path: Path = PDF_PATH,
        cfg: BuildConfig | None = None,
        rebuild: bool = False,
        workers: int = 4,
        batch: int = 128,
        quiet: bool = False,
        source_name: str | None = None,
        doc: str = "RM0008",
    ) -> dict:
        cfg = cfg or BuildConfig()
        started = time.time()

        # 先解析：PDF 缺失、解析失败都发生在动旧集合之前
        sections, documents = build_documents(pdf_path, cfg, source_name=source_name, doc=doc)
        if not documents:
            raise RuntimeError("解析结果为空，未改动现有集合。")
        figures = extract_figures(pdf_path, cfg) if cfg.include_figures else []
        if figures:
            documents = documents + figure_documents(figures, pdf_path.name)
        if not quiet:
            print(
                f"解析完成：{len(sections)} 个小节 + {len(figures)} 张插图"
                f" -> {len(documents)} 个块（{time.time() - started:.1f}s）"
            )

        result = self.add_documents(
            documents, rebuild=rebuild, workers=workers, batch=batch, quiet=quiet
        )
        result["sections"] = len(sections)
        result["figures"] = len(figures)
        result["seconds"] = round(time.time() - started, 1)
        return result

    def add_documents(
        self,
        documents: list[Document],
        *,
        rebuild: bool = False,
        workers: int = 4,
        batch: int = 128,
        quiet: bool = False,
    ) -> dict:
        """把已切好的 Document 列表写进集合（已存在的 id 自动跳过，可断点续跑）。"""
        if not documents:
            raise RuntimeError("待写入的块为空，未改动现有集合。")
        started = time.time()

        # rebuild 先写进临时集合，写完再切换：中途失败不会把现有问答弄挂
        work_name = f"{self.name}__rebuild" if rebuild else self.name
        if rebuild:
            try:
                self._client.delete_collection(work_name)
            except Exception:
                pass
        collection = self._client.get_or_create_collection(
            name=work_name, metadata={"hnsw:space": "cosine"}
        )

        existing = set(collection.get(include=[])['ids'])
        todo = [d for d in documents if d.id not in existing]
        if not todo:
            if not quiet:
                print(f"所有块已存在（{len(existing)} 条），跳过写入。")
            if rebuild:
                self._swap(work_name)
            return {"chunks": len(documents), "added": 0, "skipped": len(documents)}

        if not quiet:
            print(f"待向量化 {len(todo)} 块（已存在 {len(existing)} 块），嵌入模型 {self.embeddings.model}")
        vectors = self.embed_texts([d.page_content for d in todo], workers=workers, quiet=quiet)

        for start in range(0, len(todo), batch):
            slice_docs: list[Document] = todo[start : start + batch]
            collection.add(
                ids=[d.id for d in slice_docs],
                embeddings=vectors[start : start + batch],
                documents=[d.page_content for d in slice_docs],
                metadatas=[_sanitize(d.metadata) for d in slice_docs],
            )
            if not quiet:
                print(f"\r  写入 {min(start + batch, len(todo))}/{len(todo)}", end="", flush=True)
        if not quiet:
            print()

        if rebuild:
            self._swap(work_name)

        return {
            "chunks": len(documents),
            "added": len(todo),
            "skipped": len(documents) - len(todo),
            "seconds": round(time.time() - started, 1),
        }

    # ------------------------------------------------------------ 读取
    def corpus(self) -> tuple[list[str], list[str], list[dict]]:
        """全量取出 (ids, documents, metadatas)，供 BM25 与上下文补全使用。"""
        data = self.collection.get(include=["documents", "metadatas"])
        return data["ids"], data["documents"], data["metadatas"]

    def stats(self) -> dict:
        if not self.exists():
            return {"collection": self.name, "count": 0, "exists": False}
        collection = self.collection
        data = collection.get(include=["metadatas"])
        metas = data["metadatas"] or []
        chapters = {m.get("chapter", "") for m in metas}
        tables = sum(1 for m in metas if m.get("has_table"))
        return {
            "collection": self.name,
            "count": collection.count(),
            "exists": True,
            "chapters": len(chapters),
            "chunks_with_table": tables,
            "path": str(self.db_dir),
        }
