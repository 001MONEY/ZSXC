"""Markdown -> 可检索块（MinerU 解析产物、手写笔记通用）。

为什么按标题切而不是按字符切：
- MinerU 输出保留了 `#/##/###` 层级，标题就是天然的语义边界；
- 表格（`| ... |`）与代码块（```)必须整块保留，从中间截断会让位表/示例失去意义。

元数据保留 `doc / section_id / title / path / level / tokens / has_table`，
和 PDF 通道的字段对齐，检索、重排、出处展示可以直接复用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from .chunker import count_tokens, pack_text

MAX_TOKENS = 480
OVERLAP_TOKENS = 64
MIN_TOKENS = 60

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_SYMBOL_RE = re.compile(
    r"\b(?:0x[0-9A-Fa-f]{1,8})\b"
    r"|\b[A-Z][A-Za-z0-9]{1,9}x?_[A-Z0-9_]{2,}\b"
    r"|\b(?:HAL|LL)_[A-Za-z0-9_]+\b"
)
_PAGE_LINE_RE = re.compile(r"^(?:.*\bpage\s+\d+\s*/\s*\d+\s*|.*-\s*Rev\s+\d+\s*)$", re.IGNORECASE)
# OCR/字体嵌入常见问题：STM32 -> "S T M 32"、HAL -> "H A L"，只合并「连续单字母/数字」的片段
_SPACED_LATIN_RE = re.compile(r"\b(?:[A-Za-z0-9]\s){2,}[A-Za-z0-9]\b")


@dataclass
class MdSection:
    title: str
    level: int
    path: str
    text: str


def clean_markdown(text: str) -> str:
    """去掉 MinerU 常见噪声：页眉页脚行、纯页码、HTML 换行、超长空行。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("<br>", "\n").replace("<br/>", "\n")
    kept: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            kept.append(line)
            continue
        if not in_fence:
            stripped = line.strip()
            if not stripped:
                kept.append("")
                continue
            if _PAGE_LINE_RE.match(stripped) or re.fullmatch(r"\d{1,4}", stripped):
                continue
            if stripped.startswith("<!--") and stripped.endswith("-->"):
                continue
        kept.append(line.rstrip())
    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _SPACED_LATIN_RE.sub(lambda m: m.group(0).replace(" ", ""), text)
    return text.strip()


def split_markdown(text: str) -> tuple[str, list[MdSection]]:
    """按标题切小节；代码块内的 `#` 不算标题。返回 (前言, sections)。"""
    lines = text.splitlines()
    preface: list[str] = []
    sections: list[MdSection] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if not body and not stack:
            preface.extend(buf)
        elif body or stack:
            title = stack[-1][1] if stack else "前言"
            sections.append(
                MdSection(
                    title=title,
                    level=stack[-1][0] if stack else 0,
                    path=" > ".join(t for _, t in stack) or title,
                    text=body,
                )
            )
        buf.clear()

    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            buf.append(line)
            continue
        match = None if in_fence else _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            continue
        buf.append(line)
    flush()

    if not sections and preface:
        sections.append(MdSection(title="全文", level=0, path="全文", text="\n\n".join(preface).strip()))
        preface = []
    return "\n".join(preface).strip(), sections


def _symbols(text: str) -> str:
    seen: list[str] = []
    for token in _SYMBOL_RE.findall(text):
        if token not in seen:
            seen.append(token)
        if len(seen) >= 24:
            break
    return ";".join(seen)


def _plain_images(text: str) -> str:
    """图片链接对文本检索没帮助，换成可读占位。"""
    return _IMAGE_RE.sub(lambda m: f"[图片: {m.group(1) or Path(m.group(2)).name}]", text)


def markdown_documents(
    md_path: str | Path,
    doc: str | None = None,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Document]:
    path = Path(md_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 Markdown：{path}")
    doc_name = doc or path.stem
    text = clean_markdown(path.read_text(encoding="utf-8", errors="ignore"))
    _, sections = split_markdown(text)

    docs: list[Document] = []
    for s_index, section in enumerate(sections):
        body = _plain_images(section.text)
        if count_tokens(body) < MIN_TOKENS and s_index + 1 < len(sections):
            # 太短的过渡小节并到下一节，避免占 top-k
            sections[s_index + 1].text = f"{body}\n\n{sections[s_index + 1].text}"
            continue
        pieces = pack_text(body, max_tokens, overlap_tokens) or [body]
        for index, piece in enumerate(pieces):
            chunk_text = piece if index == 0 else f"[{section.title}]\n{piece}"
            tokens = count_tokens(chunk_text)
            metadata = {
                "source": path.name,
                "doc": doc_name,
                "kind": "md_section",
                "symbol": "",
                "title": section.title,
                "section_id": f"md:{s_index}",
                "chapter": section.path.split(" > ")[0] if section.path else section.title,
                "chapter_no": "",
                "path": section.path,
                "level": int(section.level),
                "page": 0,
                "pages": "-",
                "chunk_index": int(index),
                "n_chunks": int(len(pieces)),
                "tokens": int(tokens),
                "has_table": bool(any(line.strip().startswith("|") for line in piece.splitlines())),
                "symbols": _symbols(chunk_text),
                "merged": "",
            }
            docs.append(
                Document(
                    page_content=chunk_text,
                    metadata=metadata,
                    id=f"md:{path.stem}:{s_index}#{index}",
                )
            )
    return docs


def summarize(docs: list[Document]) -> str:
    tables = sum(1 for d in docs if d.metadata.get("has_table"))
    tokens = sum(d.metadata.get("tokens", 0) for d in docs)
    return f"块数 {len(docs)}（含表格 {tables}），约 {tokens:,} tokens"
