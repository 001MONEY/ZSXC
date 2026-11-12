"""UM1850（HAL/LL 驱动说明书）-> 可检索块。

结构特点（实测 Rev 3）：
- 每个外设一章（如 "20 HAL GPIO Generic Driver"），章下三节：
  registers structures / API description / defines；
- API 章节的正文里，每个函数是一个条目：
      HAL_GPIO_Init
      Function name
      void HAL_GPIO_Init (GPIO_TypeDef * GPIOx, GPIO_InitTypeDef * GPIO_Init)
      Function description
      ...
      Parameters
      Return values
所以在这里做二次切分：先按书签切小节，再把 API 小节切成「一函数一块」。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

from .chunker import count_tokens, pack_text
from .config import UM_PDF_PATH, BuildConfig
from .loader import Section, load_sections

MAX_TOKENS = 480
OVERLAP_TOKENS = 64

_ENTRY_RE = re.compile(r"(?m)^(HAL_[A-Za-z0-9_]+|LL_[A-Za-z0-9_]+)\s*$\nFunction name\s*$")
_PERIPHERAL_RE = re.compile(r"\b(?:HAL|LL)\s+([A-Za-z0-9_]+)")
_PROTO_RE = re.compile(r"(?m)^(?:void|int|uint\d+_t|[A-Z]\w+)\s+\**\w+\s*\(")
_TYPE_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*_TypeDef|HAL_StatusTypeDef)\b")


def um_config() -> BuildConfig:
    """UM1850 的版心与前置页：页眉 y≈56~80，页脚 y≈820，正文 88~800。"""
    return BuildConfig(
        header_y=90.0,
        footer_y=800.0,
        skip_front_pages=2,
        include_figures=False,
        max_tokens=MAX_TOKENS,
        overlap_tokens=OVERLAP_TOKENS,
        max_table_rows=80,
        exclude_chapter_patterns=(r"Revision history",),
    )


def _peripheral(chapter: str, fallback: str = "") -> str:
    match = _PERIPHERAL_RE.search(chapter)
    return match.group(1) if match else fallback


def _first_sentence(text: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    match = re.match(r"(.+?[.:])\s", text)
    return (match.group(1) if match else text)[:limit]


def _kind_of(title: str, text: str = "") -> str:
    low = title.lower()
    if "registers structures" in low or "Data Fields" in text or "Field Documentation" in text:
        return "typedef"
    if "defines" in low:
        return "define"
    if "api description" in low or "detailed description" in low:
        return "overview"
    return "section"


def _base_meta(section: Section, periph: str, doc: str = "UM1850", rev: str = "3") -> dict:
    return {
        "source": UM_PDF_PATH.name,
        "doc": doc,
        "rev": rev,
        "module": periph,
        "file": f"UM1850 §{section.number or '-'}",
        "lines": f"p{section.start_page}-{section.end_page}",
        "chapter": section.chapter,
        "chapter_no": (re.match(r"^(\d+)", section.number or section.chapter) or [None, ""])[1],
        "path": section.path,
        "level": int(section.anchor.level),
        "page": int(section.start_page),
        "pages": f"{section.start_page}-{section.end_page}",
        "has_table": False,
        "merged": "",
    }


def _entry_documents(section: Section, head: str, entries: list[tuple[str, str]], periph: str) -> list[Document]:
    docs: list[Document] = []
    for symbol, body in entries:
        lines = [line for line in body.splitlines() if line.strip()]
        proto = lines[0].strip() if lines else ""
        desc_match = re.search(
            r"(?m)^Function description\s*\n(.+?)(?=\n(?:Parameters|Return values|Notes)\s*$|\Z)", body, re.S
        )
        desc = _first_sentence(desc_match.group(1)) if desc_match else ""
        summary = f"{periph} 模块的 HAL 函数 {symbol}" + (f"：{desc}" if desc else "")
        header = (
            f"### {symbol}\n"
            f"UM1850 §{section.number} | 模块 {periph} | 类型 function | p.{section.start_page}\n"
            f"原型: {proto}\n{summary}\n\n"
        )
        payload = body.strip()
        pieces = [payload] if count_tokens(payload) <= MAX_TOKENS else pack_text(payload, MAX_TOKENS, OVERLAP_TOKENS)
        for index, piece in enumerate(pieces):
            text = header + piece if index == 0 else f"[{symbol} 续 {index + 1}/{len(pieces)}]\n{piece}"
            meta = _base_meta(section, periph)
            meta.update(
                {
                    "kind": "function",
                    "symbol": symbol,
                    "title": symbol,
                    "section_id": f"{section.section_id}:{symbol}",
                    "chunk_index": int(index),
                    "n_chunks": int(len(pieces)),
                    "tokens": int(count_tokens(text)),
                    "symbols": ";".join(dict.fromkeys([symbol] + _TYPE_RE.findall(proto))),
                }
            )
            docs.append(
                Document(page_content=text, metadata=meta, id=f"um:{section.section_id}:{symbol}#{index}")
            )

    if head.strip() and count_tokens(head) > 60:
        meta = _base_meta(section, periph)
        meta.update(
            {
                "kind": "overview",
                "symbol": "",
                "title": section.title,
                "section_id": section.section_id,
                "chunk_index": 0,
                "n_chunks": 1,
                "tokens": int(count_tokens(head)),
                "symbols": ";".join(dict.fromkeys(_TYPE_RE.findall(head))),
            }
        )
        docs.append(Document(page_content=head.strip(), metadata=meta, id=f"um:{section.section_id}:overview"))
    return docs


def _section_documents(section: Section, periph: str, kind: str) -> list[Document]:
    docs: list[Document] = []
    pieces = pack_text(section.text, MAX_TOKENS, OVERLAP_TOKENS)
    for index, piece in enumerate(pieces):
        text = piece
        meta = _base_meta(section, periph)
        meta.update(
            {
                "kind": kind,
                "symbol": "",
                "title": section.title,
                "section_id": section.section_id,
                "chunk_index": int(index),
                "n_chunks": int(len(pieces)),
                "tokens": int(count_tokens(text)),
                "has_table": bool("|" in piece),
                "symbols": ";".join(dict.fromkeys(_TYPE_RE.findall(piece))),
            }
        )
        docs.append(Document(page_content=text, metadata=meta, id=f"um:{section.section_id}#{index}"))
    return docs


def um_documents(pdf_path: Path = UM_PDF_PATH, cfg: BuildConfig | None = None) -> list[Document]:
    cfg = cfg or um_config()
    if not Path(pdf_path).is_file():
        raise FileNotFoundError(f"找不到 UM1850：{pdf_path}")

    sections = load_sections(pdf_path, cfg)
    docs: list[Document] = []
    for section in sections:
        periph = _peripheral(section.chapter + " " + section.title, fallback=section.chapter)
        matches = list(_ENTRY_RE.finditer(section.text))
        if matches:
            head = section.text[: matches[0].start()]
            entries: list[tuple[str, str]] = []
            for i, match in enumerate(matches):
                end = matches[i + 1].start() if i + 1 < len(matches) else len(section.text)
                entries.append((match.group(1), section.text[match.end() : end]))
            docs.extend(_entry_documents(section, head, entries, periph))
        else:
            docs.extend(_section_documents(section, periph, _kind_of(section.title, section.text)))
    return docs


def summarize(docs: list[Document]) -> str:
    kinds = Counter(d.metadata["kind"] for d in docs)
    modules = Counter(d.metadata["module"] for d in docs)
    functions = sum(1 for d in docs if d.metadata["kind"] == "function" and d.metadata["chunk_index"] == 0)
    return (
        f"块数 {len(docs)}（函数 {functions} 个，其余 {len(docs) - functions} 块）；类型 {dict(kinds)}；"
        f"模块数 {len(modules)}，前 8: {modules.most_common(8)}"
    )
