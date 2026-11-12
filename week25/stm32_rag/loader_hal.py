"""HAL/CMSIS 源码与 Nucleo 例程 -> 可检索块。

和 PDF 切法完全不同：代码要按**语法单元**切，而不是按字符/段落。
- 函数：前置 Doxygen 注释 + 签名 + 函数体，一块一个函数；
- 结构体/枚举：typedef 整块；
- 宏：连续的 #define 段成组（跳过 include guard 这种噪声）。

每块开头补一行自然语言摘要（"GPIO 模块的函数 HAL_GPIO_Init：..."），
否则中文问题很难在纯 C 代码上取得稠密召回。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from .chunker import count_tokens, pack_text
from .config import HAL_DRIVER_DIR, HAL_EXAMPLES_DIR, HAL_VERSION, BuildConfig

MAX_TOKENS = 480
OVERLAP_TOKENS = 64
MIN_TOKENS = 40  # 太短的宏/函数片段并成一块，避免占满 top-k

_SIG_RE = re.compile(r"^(?!\s)(?:static\s+)?[\w\s\*]*?\b(HAL_[A-Za-z0-9_]+|LL_[A-Za-z0-9_]+)\s*\(")
_ANY_SIG_RE = re.compile(r"^(?!\s)(?:static\s+)?[\w\s\*]*?\b(\w+)\s*\(")
_NO_VALUE_RE = re.compile(r"^#define\s+(\w+)\s*$")
_SKIP_FILES = {"stm32_assert_template.h"}
_TYPE_RE = re.compile(r"\b(GPIO_TypeDef|GPIO_InitTypeDef|HAL_StatusTypeDef|[A-Z][A-Za-z0-9]*_TypeDef)\b")
_MACRO_GROUP_TOKENS = 380  # 多个宏合并成一块时的 token 预算


@dataclass
class _Unit:
    kind: str  # function / typedef / macro
    symbol: str
    start: int  # 0 基行号
    end: int
    comment_start: int | None = None
    count: int = 1  # 宏组里包含的宏个数


def _module(path: Path) -> str:
    """stm32f1xx_hal_gpio.c -> GPIO，stm32f1xx_ll_gpio.c -> LL_GPIO，其它取其本身。"""
    stem = path.stem
    if stem.startswith("stm32f1xx_ll_"):
        return "LL_" + stem[len("stm32f1xx_ll_") :].upper()
    if stem.startswith("stm32f1xx_hal_"):
        return stem[len("stm32f1xx_hal_") :].upper()
    return stem.upper()


def _scan_units(lines: list[str], name_re: re.Pattern[str] = _SIG_RE) -> list[_Unit]:
    units: list[_Unit] = []
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]

        if line.startswith("typedef struct") or line.startswith("typedef enum"):
            j = i
            while j < n and not re.match(r"^\}\s*\w+\s*;", lines[j]):
                j += 1
            if j < n:
                name = re.match(r"^\}\s*(\w+)\s*;", lines[j]).group(1)
                units.append(_Unit("typedef", name, i, j, _comment_above(lines, i)))
                i = j + 1
                continue

        if line.startswith("#define"):
            j = i
            while j + 1 < n and (lines[j + 1].startswith("#define") or lines[j].rstrip().endswith("\\")):
                j += 1
            names = [re.match(r"^#define\s+(\w+)", x).group(1) for x in lines[i : j + 1] if x.startswith("#define")]
            # 无值的 "#define XXX_H" 基本都是 include guard，丢掉
            names = [
                x for x in names
                if not (x.endswith("_H") and any(_NO_VALUE_RE.match(y) and _NO_VALUE_RE.match(y).group(1) == x
                                                 for y in lines[i : j + 1]))
            ]
            if names:
                units.append(_Unit("macro", names[0], i, j, _comment_above(lines, i), len(names)))
            i = j + 1
            continue

        if line[:1].isalpha() and not line.startswith("#"):
            match = name_re.match(line)
            if match:
                # 签名可能跨行：向后最多 8 行找 '{' 或 ';'
                j = i
                while j < n and "{" not in lines[j] and ";" not in lines[j] and j - i < 8:
                    j += 1
                if j < n and "{" in lines[j] and ";" not in lines[j]:
                    k = j
                    while k < n and lines[k].strip() != "}":
                        k += 1
                    if k < n:
                        units.append(_Unit("function", match.group(1), i, k, _comment_above(lines, i)))
                        i = k + 1
                        continue
        i += 1
    return units


def _merge_macro_units(units: list[_Unit], lines: list[str], max_gap: int = 25) -> list[_Unit]:
    """把相邻的宏单元合并成不超过 token 预算的块。

    HAL 头文件里一个宏常跟着一段注释，如果每个宏单独成块，会产生上千个碎片。
    """
    merged: list[_Unit] = []
    current: _Unit | None = None

    for unit in units:
        if unit.kind != "macro":
            if current is not None:
                merged.append(current)
                current = None
            merged.append(unit)
            continue

        if current is None:
            current = _Unit("macro", unit.symbol, unit.start, unit.end, unit.comment_start, unit.count)
            continue

        span = "\n".join(lines[current.start : unit.end + 1])
        gap = unit.start - current.end
        if gap <= max_gap and count_tokens(span) <= _MACRO_GROUP_TOKENS:
            current.end = unit.end
            current.count += unit.count
        else:
            merged.append(current)
            current = _Unit("macro", unit.symbol, unit.start, unit.end, unit.comment_start, unit.count)

    if current is not None:
        merged.append(current)
    return merged


def _comment_above(lines: list[str], index: int, max_lines: int = 80) -> int | None:
    """函数签名上方紧邻的 Doxygen 注释块起点；没有就返回 None。"""
    j = index - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    if j < 0:
        return None
    tail = lines[j].strip()
    if not (tail.endswith("*/") or tail.startswith("*")):
        return None

    floor = max(0, index - max_lines)
    start = j
    while start >= floor:
        text = lines[start].strip()
        if text.startswith("/**") or text.startswith("/*"):
            return start
        if text.startswith("#") or text.startswith("}"):
            return None
        start -= 1
    return None


def _brief(comment: str) -> str:
    """从 Doxygen 注释里抽 @brief（可能跨行，遇到下一个 @ 标签为止）。"""
    match = re.search(r"@brief\s+(.+?)(?=\n\s*\*\s*@|\Z)", comment, re.S)
    text = match.group(1) if match else ""
    text = re.sub(r"(?m)^\s*\*\s?", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 3 and re.search(r"\w", text):
        return text
    # 没有 @brief 时退回第一句正文
    for raw in comment.splitlines():
        body = raw.strip().lstrip("*").strip()
        if body and not body.startswith("/**") and not body.startswith("*/") and not body.startswith("@"):
            return body
    return ""


def _meta(
    path: Path,
    root: Path,
    unit: _Unit,
    module: str,
    kind: str,
    title: str,
    doc: str,
    chunk_index: int = 0,
    n_chunks: int = 1,
    tokens: int = 0,
) -> dict:
    rel = path.relative_to(root).as_posix()
    lines = f"{unit.start + 1}-{unit.end + 1}"
    return {
        "source": path.name,
        "doc": doc,
        "kind": kind,
        "symbol": unit.symbol,
        "module": module,
        "file": rel,
        "lines": lines,
        "title": title,
        "section_id": f"{rel}:{unit.start + 1}",
        "chapter": f"{doc} {module}",
        "chapter_no": module,
        "path": rel,
        "level": 3,
        "page": 0,
        "pages": "-",
        "chunk_index": int(chunk_index),
        "n_chunks": int(n_chunks),
        "tokens": int(tokens),
        "has_table": False,
        "symbols": ";".join(dict.fromkeys([unit.symbol] + _TYPE_RE.findall(title))),
        "merged": "",
        "version": HAL_VERSION,
    }


def _emit(
    path: Path,
    root: Path,
    unit: _Unit,
    lines: list[str],
    module: str,
    summary: str,
    doc: str,
    cfg: BuildConfig | None = None,
) -> list[Document]:
    """把单个语法单元切成 1..n 块，首块带标题+摘要。"""
    cfg = cfg or BuildConfig(max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS)
    comment = ""
    if unit.comment_start is not None:
        comment = "\n".join(lines[unit.comment_start : unit.start])
    body = "\n".join(lines[unit.start : unit.end + 1]).strip()
    payload = (comment.strip() + "\n" + body).strip() if comment.strip() else body

    header = f"### {unit.symbol}\n// {path.relative_to(root).as_posix()}:{unit.start + 1}-{unit.end + 1}\n// {doc} | 模块 {module} | 类型 {unit.kind}\n{summary}\n\n"
    pieces = [payload] if count_tokens(payload) <= cfg.max_tokens else pack_text(
        payload, cfg.max_tokens, cfg.overlap_tokens
    )
    docs: list[Document] = []
    for index, piece in enumerate(pieces):
        text = header + piece if index == 0 else f"[{unit.symbol} 续 {index + 1}/{len(pieces)}]\n{piece}"
        title = unit.symbol if unit.kind != "macro" else f"{module} 宏定义组（{unit.symbol} 等）"
        docs.append(
            Document(
                page_content=text,
                metadata=_meta(path, root, unit, module, unit.kind, title, doc, index, len(pieces), count_tokens(text)),
                id=f"{doc.lower()}:{path.relative_to(root).as_posix()}:{unit.start + 1}#{index}",
            )
        )
    return docs


def hal_documents(
    driver_dir: Path = HAL_DRIVER_DIR,
    cfg: BuildConfig | None = None,
) -> list[Document]:
    """HAL 驱动 .c/.h -> Documents。"""
    if not driver_dir.is_dir():
        raise FileNotFoundError(f"找不到 HAL 驱动目录：{driver_dir}")

    docs: list[Document] = []
    for path in sorted(list((driver_dir / "Src").glob("*.c")) + list((driver_dir / "Inc").glob("*.h"))):
        if path.name in _SKIP_FILES:
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        module = _module(path)
        for unit in _merge_macro_units(_scan_units(lines), lines):
            comment = "\n".join(lines[unit.comment_start : unit.start]) if unit.comment_start is not None else ""
            brief = _brief(comment)
            if unit.kind == "function":
                summary = f"{module} 模块的函数 {unit.symbol}" + (f"：{brief}" if brief else "")
            elif unit.kind == "typedef":
                summary = f"{module} 模块的结构体/枚举 {unit.symbol}" + (f"：{brief}" if brief else "")
            else:
                summary = f"{module} 模块的宏定义组（{unit.count} 个宏）" + (f"：{brief}" if brief else "")
            docs.extend(_emit(path, driver_dir, unit, lines, module, summary, "HAL", cfg))
    return docs


def examples_documents(
    examples_dir: Path = HAL_EXAMPLES_DIR,
    cfg: BuildConfig | None = None,
) -> list[Document]:
    """Nucleo 例程的 main.c -> Documents（按函数切）。"""
    if not examples_dir.is_dir():
        raise FileNotFoundError(f"找不到例程目录：{examples_dir}")

    docs: list[Document] = []
    for path in sorted(examples_dir.rglob("main.c")):
        rel_parts = path.relative_to(examples_dir).parts
        category = rel_parts[0] if rel_parts else "?"
        project = rel_parts[1] if len(rel_parts) > 1 else path.parent.name
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for unit in _scan_units(lines, _ANY_SIG_RE):
            if unit.kind != "function":
                continue
            comment = "\n".join(lines[unit.comment_start : unit.start]) if unit.comment_start is not None else ""
            brief = _brief(comment)
            summary = f"例程 {category}/{project} 的 {unit.symbol} 函数" + (f"：{brief}" if brief else "")
            docs.extend(_emit(path, examples_dir, unit, lines, category, summary, "HAL_EXAMPLE", cfg))
    return docs


def summarize(docs: list[Document]) -> str:
    from collections import Counter

    kinds = Counter(d.metadata["kind"] for d in docs)
    modules = Counter(d.metadata["module"] for d in docs)
    tokens = sum(d.metadata["tokens"] for d in docs)
    return (
        f"块数 {len(docs)}，约 {tokens:,} tokens；类型 {dict(kinds)}；"
        f"模块数 {len(modules)}，前 8: {modules.most_common(8)}"
    )
