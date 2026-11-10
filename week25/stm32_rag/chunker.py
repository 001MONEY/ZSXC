"""Section -> 可检索的 chunk。

为什么不用 `RecursiveCharacterTextSplitter` 了：
- 这本书的段落是英文长句，按字符硬切会把一句话切两半；
- 寄存器位表必须整张留着，不能从中间截断；
- 1632 个书签里有大量两三行的小节，单独成块会占满 top-k 名额。

所以改成：按 token 装包（段落为最小单位，表格原子不可分），短小节先并入上一块，
同一个小节切出的后续块补一行 [小节标题] 作为上下文前缀。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken
from langchain_core.documents import Document

from .config import PDF_PATH, BuildConfig
from .loader import Section, load_sections

_ENC = tiktoken.get_encoding("cl100k_base")

# 抽取寄存器名/位域名/地址，供关键词通道使用
_SYMBOL_RE = re.compile(
    r"\b(?:0x[0-9A-Fa-f]{1,8})\b"
    r"|\b[A-Z][A-Za-z0-9]{1,9}x?_[A-Z0-9_]{2,}\b"
    r"|\b[A-Z][A-Z0-9]{1,12}\[\d+(?::\d+)?\]\b"
)


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def _split_table(text: str, max_tokens: int) -> list[str]:
    """超长表格按「行」切，并在每片前面重复表头，绝不从一行中间断开。"""
    rows = text.split("\n")
    header, body = (rows[0], rows[1:]) if len(rows) > 1 else ("", rows)
    out: list[str] = []
    current: list[str] = []
    size = 0
    for row in body:
        n = count_tokens(row) + 1
        if current and size + n > max_tokens:
            out.append("\n".join(([header] if header else []) + current))
            current, size = [], 0
        current.append(row)
        size += n
    if current:
        out.append("\n".join(([header] if header else []) + current))
    return out


def _split_oversize(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """单个段落/表格超过块上限时切开。表格按行切，正文按 token 窗口切。"""
    if text.lstrip().startswith("|"):
        return _split_table(text.strip(), max_tokens)

    ids = _ENC.encode(text, disallowed_special=())
    out: list[str] = []
    step = max(1, max_tokens - overlap_tokens)
    for start in range(0, len(ids), step):
        piece = _ENC.decode(ids[start : start + max_tokens])
        if piece.strip():
            out.append(piece.strip())
        if start + max_tokens >= len(ids):
            break
    return out


def pack_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """按段落装包，块与块之间带重叠。

    重叠只回带**完整段落**：否则块边界正好落在寄存器位表中间时，
    下一块会以 "rw |" 这种半行开头。
    """
    units = [u.strip() for u in re.split(r"\n\s*\n", text) if u.strip()]
    chunks: list[str] = []
    current: list[str] = []
    sizes: list[int] = []
    current_tokens = 0

    for unit in units:
        n = count_tokens(unit)
        if n > max_tokens:  # 超长表格或超长段：先冲掉当前包，再单独切
            if current:
                chunks.append("\n\n".join(current))
                current, sizes, current_tokens = [], [], 0
            chunks.extend(_split_oversize(unit, max_tokens, overlap_tokens))
            continue

        if current and current_tokens + n > max_tokens:
            chunks.append("\n\n".join(current))
            keep: list[tuple[str, int]] = []
            budget = overlap_tokens
            for prev_unit, prev_size in zip(reversed(current), reversed(sizes)):
                if prev_size <= budget and prev_size * 2 <= max_tokens:
                    keep.insert(0, (prev_unit, prev_size))
                    budget -= prev_size
                else:
                    break
            current = [u for u, _ in keep]
            sizes = [s for _, s in keep]
            current_tokens = sum(sizes)

        current.append(unit)
        sizes.append(n)
        current_tokens += n

    if current:
        chunks.append("\n\n".join(current))
    return chunks


@dataclass
class _Merged:
    section: Section
    text: str
    extra_titles: list[str]


def _merge_short_sections(sections: list[Section], cfg: BuildConfig) -> list[_Merged]:
    """短小节前后合并。

    规则：只能并到「同级或更细」的小节上——把父级小节（如 9.2）拼到子级小节（如 9.1.11）
    的正文后面，读起来会是倒序的。父级短小节改为向下挂到它的第一个子小节前面。
    """
    merged: list[_Merged] = []
    carry: _Merged | None = None

    for section in sections:
        if carry is not None and carry.section.chapter != section.chapter:
            merged.append(carry)  # 换章了，挂不上去，单独成块
            carry = None

        text = section.text
        extra: list[str] = []
        if carry is not None:
            text = f"{carry.text}\n\n{text}"
            extra = carry.extra_titles + [carry.section.title]
            carry = None

        tokens = count_tokens(text)
        can_merge_back = bool(
            merged
            and tokens < cfg.min_tokens
            and merged[-1].section.chapter == section.chapter
            and section.anchor.level >= merged[-1].section.anchor.level
            and count_tokens(merged[-1].text) + tokens <= int(cfg.max_tokens * 1.4)
        )
        if can_merge_back:
            merged[-1].text += "\n\n" + text
            merged[-1].extra_titles.extend(extra + [section.title])
            continue

        item = _Merged(section=section, text=text, extra_titles=extra)
        if tokens < cfg.min_tokens and section.anchor.level <= 3:
            carry = item  # 可能是父级标题下的过渡性文字，等下一节来了拼到前面
            continue
        merged.append(item)

    if carry is not None:
        merged.append(carry)
    return merged


def _symbols(text: str) -> str:
    seen: list[str] = []
    for token in _SYMBOL_RE.findall(text):
        if token not in seen:
            seen.append(token)
        if len(seen) >= 24:
            break
    return ";".join(seen)


def _chapter_no(chapter: str) -> str:
    match = re.match(r"^(\d+)", chapter.strip())
    return match.group(1) if match else ""


def _density(chapter: str) -> str:
    """同一外设在手册里按器件密度分多章描述，标出来便于去重与回答。"""
    low = chapter.lower()
    if "connectivity line" in low:
        return "connectivity-line"
    if "density" in low:
        return "low-medium-high-xl-density"
    return ""


def chunk_sections(sections: list[Section], cfg: BuildConfig | None = None) -> list[Document]:
    cfg = cfg or BuildConfig()
    docs: list[Document] = []

    for item in _merge_short_sections(sections, cfg):
        section = item.section
        pieces = pack_text(item.text, cfg.max_tokens, cfg.overlap_tokens)
        for i, piece in enumerate(pieces):
            if i > 0:  # 后续块补上小节标题，避免脱离上下文
                body = f"[{section.title}]\n{piece}"
            else:
                body = piece
            metadata = {
                "source": PDF_PATH.name,
                "section_id": section.section_id,
                "title": section.title,
                "chapter": section.chapter,
                "chapter_no": _chapter_no(section.chapter),
                "density": _density(section.chapter),
                "path": section.path,
                "level": int(section.anchor.level),
                "page": int(section.start_page),
                "pages": f"{section.start_page}-{section.end_page}",
                "chunk_index": int(i),
                "n_chunks": int(len(pieces)),
                "tokens": int(count_tokens(body)),
                "has_table": bool("|" in piece),
                "symbols": _symbols(piece),
                "merged": ";".join(item.extra_titles),
            }
            docs.append(
                Document(
                    page_content=body,
                    metadata=metadata,
                    id=f"{section.section_id}#{i}",
                )
            )
    return docs


def build_documents(
    pdf_path=PDF_PATH, cfg: BuildConfig | None = None
) -> tuple[list[Section], list[Document]]:
    cfg = cfg or BuildConfig()
    sections = load_sections(pdf_path, cfg)
    return sections, chunk_sections(sections, cfg)
