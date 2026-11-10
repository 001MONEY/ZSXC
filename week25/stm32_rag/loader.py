"""PDF -> 章节化文本。

思路：
1. 用 `get_toc(simple=False)` 拿到 1623 条书签，每条都带「页码 + 标题在页面上的 y 坐标」，
   这就是天然的切节锚点。过滤掉 Table/Figure 书签（它们只是图表标题，不该切断寄存器小节）。
2. 逐页取文本块（按 y 裁掉页眉页脚），再用 `find_tables()` 把寄存器位域表转成 Markdown，
   并把被表格覆盖的原始文本块剔除——否则位表会被抽成「31 30 29 … rw rw」这种噪声。
3. 按块的 y 坐标二分查找归属小节，一个页面里出现多个小节也能分开。
"""

from __future__ import annotations

import hashlib
import pickle
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .cleaner import clean_text, normalize_space
from .config import DB_DIR, PDF_PATH, BuildConfig

_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.*)$")
_SKIP_TITLE_RE = re.compile(r"^(table|figure)\s+\d+", re.IGNORECASE)


@dataclass
class Anchor:
    """一个章节书签。"""

    level: int
    title: str
    page: int  # 0 基页码（PyMuPDF 内部索引）
    y: float  # 标题在页面上的纵坐标
    number: str  # "9.2.1"，没有编号则为空

    @property
    def printed_page(self) -> int:
        return self.page + 1


@dataclass
class Section:
    """一个小节（同一锚点下的全部正文）。"""

    anchor: Anchor
    title: str
    number: str
    chapter: str
    path: str
    start_page: int  # 印刷页码
    end_page: int
    text: str
    n_tables: int = 0
    # 建库内部唯一 ID：同一页可能挂着多个无编号小节，光靠 level@page 会撞车
    uid: str = ""

    @property
    def section_id(self) -> str:
        return getattr(self, "uid", "") or self.number or f"L{self.anchor.level}@p{self.start_page}"


def split_number(title: str) -> tuple[str, str]:
    """"9.2.1 Port configuration register low" -> ("9.2.1", "Port configuration ...")"""
    m = _NUM_RE.match(title)
    if m:
        return m.group(1), m.group(2).strip()
    return "", title


def build_anchors(doc: pymupdf.Document, cfg: BuildConfig) -> list[Anchor]:
    anchors: list[Anchor] = []
    for item in doc.get_toc(simple=False):
        level, title, page1 = item[0], item[1], item[2]
        dest = item[3] if len(item) > 3 else {}
        page = int(dest.get("page", page1 - 1))
        to = dest.get("to")
        y = float(to.y) if to is not None else 0.0

        title = normalize_space(title)
        if not title or _SKIP_TITLE_RE.match(title):
            continue
        if page < cfg.skip_front_pages:
            continue
        number, _ = split_number(title)
        anchors.append(Anchor(level=level, title=title, page=page, y=y, number=number))

    anchors.sort(key=lambda a: (a.page, a.y, a.level))

    # 同一坐标可能挂了两级书签，保留最后（最深）的一条
    deduped: list[Anchor] = []
    for anchor in anchors:
        if deduped and deduped[-1].page == anchor.page and abs(deduped[-1].y - anchor.y) < 1.0:
            if anchor.level >= deduped[-1].level:
                deduped[-1] = anchor
            continue
        deduped.append(anchor)
    return deduped


def _intersection_ratio(block: tuple, bbox: tuple) -> float:
    x0 = max(block[0], bbox[0])
    y0 = max(block[1], bbox[1])
    x1 = min(block[2], bbox[2])
    y1 = min(block[3], bbox[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    area = max(1e-6, (block[2] - block[0]) * (block[3] - block[1]))
    return (x1 - x0) * (y1 - y0) / area


def _cell(value) -> str:
    if value is None:
        return ""
    return normalize_space(str(value).replace("|", "/"))


def table_to_markdown(table, cfg: BuildConfig) -> str:
    """把一个 PyMuPDF Table 转成 Markdown。转不了就退回逐行拼接。"""
    try:
        if table.row_count < 2 or table.col_count < 2:
            return ""
        rows = table.extract()
    except Exception:
        return ""
    rows = [r for r in rows if any(_cell(c) for c in r)][: cfg.max_table_rows]
    if len(rows) < 2:
        return ""
    lines = ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def page_pieces(page: pymupdf.Page, cfg: BuildConfig) -> list[tuple[float, str, bool]]:
    """返回 [(y0, 文本, 是否表格)]，已裁掉页眉页脚并把表格替换成 Markdown。"""
    pieces: list[tuple[float, str, bool]] = []

    table_bboxes: list[tuple] = []
    try:
        finder = page.find_tables()
    except Exception:
        finder = None
    if finder is not None:
        for table in finder.tables:
            bbox = tuple(table.bbox)
            if bbox[1] < cfg.header_y or bbox[3] > cfg.footer_y:
                continue  # 页眉页脚里被误识别的小表
            markdown = table_to_markdown(table, cfg)
            if not markdown:
                continue
            table_bboxes.append(bbox)
            pieces.append((bbox[1], markdown, True))

    for block in page.get_text("blocks"):
        x0, y0, x1, y1, raw = block[0], block[1], block[2], block[3], block[4]
        if y0 < cfg.header_y or y1 > cfg.footer_y:
            continue
        if not raw.strip():
            continue
        if any(_intersection_ratio((x0, y0, x1, y1), tb) > 0.3 for tb in table_bboxes):
            continue  # 已被表格覆盖，丢掉抽得稀碎的原文
        pieces.append((y0, raw, False))

    pieces.sort(key=lambda p: p[0])
    return pieces


def _cache_file(pdf_path: Path, cfg: BuildConfig) -> Path:
    stat = pdf_path.stat()
    raw = (
        f"{pdf_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
        f"|{cfg.header_y}|{cfg.footer_y}|{cfg.skip_front_pages}"
        f"|{cfg.page_range}|{cfg.max_table_rows}|v2"
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return DB_DIR / "_parse_cache" / f"sections_{digest}.pkl"


def _assign_uids(sections: list[Section]) -> list[Section]:
    """给每个小节分配唯一 ID（有编号用编号，重名或没编号就加序号后缀）。"""
    counts: dict[str, int] = {}
    for section in sections:
        base = section.number or f"L{section.anchor.level}@p{section.start_page}"
        occurrence = counts.get(base, 0)
        counts[base] = occurrence + 1
        section.uid = base if occurrence == 0 else f"{base}~{occurrence}"
    return sections


def _apply_filters(sections: list[Section], cfg: BuildConfig) -> list[Section]:
    """按配置丢掉整章噪声（如 Revision history）。对缓存结果也要生效。"""
    if not cfg.exclude_chapter_patterns:
        return sections
    patterns = [re.compile(p, re.IGNORECASE) for p in cfg.exclude_chapter_patterns]
    return [s for s in sections if not any(p.search(s.chapter) for p in patterns)]


def load_sections(
    pdf_path: str | Path = PDF_PATH,
    cfg: BuildConfig | None = None,
    use_cache: bool = True,
) -> list[Section]:
    """整本 PDF -> Section 列表（按出现顺序）。结果会缓存到磁盘，二次调用秒开。"""
    cfg = cfg or BuildConfig()
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"找不到 PDF：{pdf_path}")

    cache = _cache_file(pdf_path, cfg)
    if use_cache and cache.is_file():
        try:
            with cache.open("rb") as handle:
                return _apply_filters(_assign_uids(pickle.load(handle)), cfg)
        except Exception:
            pass

    with pymupdf.open(str(pdf_path)) as doc:
        anchors = build_anchors(doc, cfg)
        if not anchors:
            raise ValueError("没有解析到章节书签，请检查 PDF 是否自带目录。")

        positions = [(a.page, a.y) for a in anchors]
        buffers: dict[int, list[str]] = {}
        tables: dict[int, int] = {}
        end_pages: dict[int, int] = {}

        # 只处理范围内的页，但锚点全集都保留：否则 --pages 会丢掉父级章节，
        # chapter 字段就变成 "9.1.1 …" 这种小标题了。
        lo = 1 if cfg.page_range is None else cfg.page_range[0]
        hi = doc.page_count if cfg.page_range is None else cfg.page_range[1]
        start = max(0, lo - 1)
        stop = min(doc.page_count, hi)

        for pno in range(start, stop):
            page = doc[pno]
            for y0, text, is_table in page_pieces(page, cfg):
                idx = bisect_right(positions, (pno, y0 + 2.0)) - 1
                if idx < 0:
                    continue
                anchor = anchors[idx]
                # 块本身就是标题：正文已经用 "### 标题" 形式写在块首，不再重复
                if anchor.page == pno and abs(anchor.y - y0) <= 4.5 and not is_table:
                    continue
                buffers.setdefault(idx, []).append(text)
                if is_table:
                    tables[idx] = tables.get(idx, 0) + 1
                end_pages[idx] = max(end_pages.get(idx, 0), pno + 1)

        sections: list[Section] = []
        stack: list[Anchor] = []
        for idx, anchor in enumerate(anchors):
            while stack and stack[-1].level >= anchor.level:
                stack.pop()
            stack.append(anchor)
            chapter = stack[0].title
            path = " > ".join(a.title for a in stack)

            pieces = buffers.pop(idx, None)
            if not pieces:
                continue
            body = clean_text("\n\n".join(pieces))
            if not body:
                continue
            section_end = end_pages.get(idx, anchor.printed_page)
            if section_end < lo or anchor.printed_page > hi:
                continue  # 与请求页范围不相交

            sections.append(
                Section(
                    anchor=anchor,
                    title=anchor.title,
                    number=anchor.number,
                    chapter=chapter,
                    path=path,
                    start_page=anchor.printed_page,
                    end_page=section_end,
                    text=f"### {anchor.title}\n\n{body}",
                    n_tables=tables.get(idx, 0),
                )
            )

    if use_cache:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("wb") as handle:
                pickle.dump(sections, handle)
        except Exception:
            pass
    return _apply_filters(_assign_uids(sections), cfg)


def summarize(sections: list[Section]) -> str:
    """建库前先看这行日志。"""
    chars = sum(len(s.text) for s in sections)
    tables = sum(s.n_tables for s in sections)
    return (
        f"小节数 {len(sections)}，正文 {chars:,} 字符，表格 {tables} 张，"
        f"页码范围 {sections[0].start_page}-{sections[-1].end_page}"
    )
