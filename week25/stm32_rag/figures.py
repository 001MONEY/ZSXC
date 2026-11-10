"""把手册里的插图抠出来，变成可检索的一块。

这份 PDF 的图全是矢量绘制（0 张位图），所以不需要 OCR：
1. 在正文里找 `Figure N.` 题注（书签里没有 Figure，只有 Table，只能扫正文）；
2. 从题注往下取一串相邻的矢量元素作为图区，顺带把落进图区的文字块并进来
   —— 图里的标签往往就是文字块，不并进来会被裁掉；
3. 按 150 dpi 裁成 PNG，存到 chroma_db/figures/；
4. 生成 kind=figure 的文本块（题注 + 章节路径 + 页码 + 图片文件名），走正常嵌入流程。

图里的细节文字（信号名、分频比、数值）因为没有 OCR 仍然查不到，
命中后由 answer(with_figures=True) 把原图随消息发给视觉模型兜底。
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

from .chunker import count_tokens
from .config import FIGURES_DIR, PDF_PATH, BuildConfig
from .loader import build_anchors

_CAPTION_RE = re.compile(r"^\s*Figure\s+(\d+)\.\s*(.+)$", re.MULTILINE)
_DOT_TAIL_RE = re.compile(r"\s*(?:\.\s*){4,}\s*\d*\s*$")  # 目录里的点线尾巴


@dataclass
class Figure:
    number: int
    title: str
    page: int  # 印刷页码
    image_file: str  # 相对 FIGURES_DIR
    width: int
    height: int
    chapter: str = ""
    path: str = ""
    section_id: str = ""


def _manifest_path() -> Path:
    return FIGURES_DIR / "manifest.json"


def _cache_key(pdf_path: Path, cfg: BuildConfig) -> str:
    stat = pdf_path.stat()
    return (
        f"{stat.st_mtime_ns}|{stat.st_size}|{cfg.figure_dpi}"
        f"|{cfg.skip_front_pages}|{cfg.page_range}"
    )


def _section_index(anchors) -> tuple[list[tuple[int, float]], list[tuple[str, str, str]]]:
    """预处理书签：给每个锚点算出 (章标题, 章节路径, 小节 ID)。"""
    positions: list[tuple[int, float]] = []
    metas: list[tuple[str, str, str]] = []
    stack: list = []
    for anchor in anchors:
        while stack and stack[-1].level >= anchor.level:
            stack.pop()
        stack.append(anchor)
        positions.append((anchor.page, anchor.y))
        chapter = stack[0].title
        path = " > ".join(a.title for a in stack)
        section_id = anchor.number or f"L{anchor.level}@p{anchor.page + 1}"
        metas.append((chapter, path, section_id))
    return positions, metas


def _caption_blocks(page: pymupdf.Page, cfg: BuildConfig) -> list[tuple[int, str, pymupdf.Rect]]:
    found: list[tuple[int, str, pymupdf.Rect]] = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
        if y0 < cfg.header_y or y1 > cfg.footer_y:
            continue
        match = _CAPTION_RE.search(text)
        if not match:
            continue
        title = re.sub(r"\s+", " ", _DOT_TAIL_RE.sub("", match.group(2))).strip()
        found.append((int(match.group(1)), title, pymupdf.Rect(x0, y0, x1, y1)))
    found.sort(key=lambda item: item[2].y0)
    return found


def _figure_region(
    page: pymupdf.Page,
    caption: pymupdf.Rect,
    next_caption_y: float | None,
    cfg: BuildConfig,
) -> pymupdf.Rect | None:
    """题注下方的图区。取与题注相连的一串矢量元素，再把落在其中的文字块并进来。"""
    limit = next_caption_y if next_caption_y is not None else cfg.footer_y
    limit = min(limit, cfg.footer_y)

    drawings = [pymupdf.Rect(d["rect"]) for d in page.get_drawings()]
    below = [r for r in drawings if r.y0 >= caption.y1 - 2 and r.y0 <= limit]
    if not below:
        return None

    below.sort(key=lambda r: r.y0)
    used = [below[0]]
    top, bottom = below[0].y0, below[0].y1
    for rect in below[1:]:
        if rect.y0 - bottom > 36:  # 与当前图区断开，说明是下一张图或表格
            break
        used.append(rect)
        bottom = max(bottom, rect.y1)

    # 图里的标签常常是文字块，不并进来会被裁掉
    for block in page.get_text("blocks"):
        rect = pymupdf.Rect(block[0], block[1], block[2], block[3])
        if rect.y0 < top - 8 or rect.y0 > bottom + 8 or rect.y1 > limit:
            continue
        used.append(rect)
        bottom = max(bottom, rect.y1)

    region = pymupdf.Rect(
        min(r.x0 for r in used),
        min(top, caption.y1),
        max(r.x1 for r in used),
        bottom,
    )
    if not region.is_empty and region.height >= cfg.min_figure_height:
        return region + (-4, -4, 4, 4)  # 留一点边
    return None


def extract_figures(
    pdf_path: str | Path = PDF_PATH,
    cfg: BuildConfig | None = None,
    force: bool = False,
) -> list[Figure]:
    cfg = cfg or BuildConfig()
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"找不到 PDF：{pdf_path}")

    key = _cache_key(pdf_path, cfg)
    manifest = _manifest_path()
    if not force and manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if data.get("key") == key:
                figures = [Figure(**item) for item in data["figures"]]
                if figures and all((FIGURES_DIR / f.image_file).is_file() for f in figures):
                    return figures
        except Exception:
            pass

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figures: list[Figure] = []
    name_counter: dict[str, int] = {}

    with pymupdf.open(str(pdf_path)) as doc:
        positions, metas = _section_index(build_anchors(doc, cfg))

        first = cfg.skip_front_pages if cfg.page_range is None else max(cfg.skip_front_pages, cfg.page_range[0] - 1)
        last = doc.page_count if cfg.page_range is None else min(doc.page_count, cfg.page_range[1])

        for pno in range(first, last):
            page = doc[pno]
            captions = _caption_blocks(page, cfg)
            for index, (number, title, rect) in enumerate(captions):
                next_y = captions[index + 1][2].y0 if index + 1 < len(captions) else None
                region = _figure_region(page, rect, next_y, cfg)
                if region is None:
                    continue

                base = f"fig{number:03d}_p{pno + 1}"
                seen = name_counter.get(base, 0)
                name_counter[base] = seen + 1
                file_name = f"{base}.png" if seen == 0 else f"{base}_{seen + 1}.png"

                pix = page.get_pixmap(dpi=cfg.figure_dpi, clip=region)
                pix.save(str(FIGURES_DIR / file_name))

                idx = bisect_right(positions, (pno, rect.y0 + 2.0)) - 1
                chapter, path, section_id = metas[idx] if idx >= 0 else ("", "", "")
                figures.append(
                    Figure(
                        number=number,
                        title=title,
                        page=pno + 1,
                        image_file=file_name,
                        width=pix.width,
                        height=pix.height,
                        chapter=chapter,
                        path=path,
                        section_id=section_id,
                    )
                )

    payload = {"key": key, "figures": [asdict(f) for f in figures]}
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return figures


def figure_documents(figures: list[Figure], source: str) -> list[Document]:
    docs: list[Document] = []
    for fig in figures:
        title = f"Figure {fig.number}. {fig.title}"
        text = (
            f"### {title}\n\n"
            f"所在章节：{fig.path or fig.chapter}\n"
            f"页码：p.{fig.page}\n"
            f"图片：{fig.image_file}\n"
            "（这是手册中的插图；正文提取不到图内的信号名和数值，回答图内细节需要看原图。）"
        )
        metadata = {
            "source": source,
            "kind": "figure",
            "figure_no": int(fig.number),
            # 图用独立命名空间：否则会被“同小节只留最相关一块”的规则收敛掉，
            # 也会被邻居补全当成正文块去找前后块
            "section_id": f"fig{fig.number:03d}p{fig.page}",
            "title": title,
            "chapter": fig.chapter,
            "chapter_no": "",
            "density": "",
            "path": fig.path or fig.chapter,
            "level": 4,
            "page": int(fig.page),
            "pages": str(fig.page),
            "chunk_index": 0,
            "n_chunks": 1,
            "tokens": int(count_tokens(text)),
            "has_table": False,
            "symbols": f"Figure {fig.number}",
            "merged": "",
            "image_file": fig.image_file,
        }
        docs.append(
            Document(
                page_content=text,
                metadata=metadata,
                id=f"fig{fig.number:03d}_p{fig.page}",
            )
        )
    return docs


def summarize(figures: list[Figure]) -> str:
    if not figures:
        return "没有抽到插图"
    pages = sorted({f.page for f in figures})
    with_image = sum(1 for f in figures if (FIGURES_DIR / f.image_file).is_file())
    return f"插图 {len(figures)} 张（带 PNG {with_image} 张），分布 {pages[0]}-{pages[-1]} 页"
