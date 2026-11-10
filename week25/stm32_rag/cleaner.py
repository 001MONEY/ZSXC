"""英文技术手册的文本清洗。

与中文讲解稿的区别：
1. 中文靠「汉字之间不该换行」来合行；英文靠「行尾没有句末标点就换行」，还要处理连字符断词。
2. 页眉页脚是固定字符串（DocID / RM0008 / 页码），且每页重复上千次，必须清掉。
3. 表格已转成 Markdown，属于结构化内容，整体跳过清洗。
"""

from __future__ import annotations

import re

# 连字符、不换行空格、零宽字符、项目符号字体映射
_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u00ad"), None)
_BULLET_CHARS = "\uf0b7\u00b7\u25cf\u25aa\u25a0"

PAGE_NO_RE = re.compile(r"^\s*\d{1,4}\s*/\s*\d{3,4}\s*$")
DOCID_RE = re.compile(r"^\s*DocID\s*\d+\s+Rev\.?\s*\d+\s*$", re.IGNORECASE)
RM_RE = re.compile(r"^\s*RM\d{4,5}\s*$", re.IGNORECASE)

# 句末标点：出现在行尾说明这段话说完了，不应该和下一条合并
_SENTENCE_END = (".", ":", ";", "!", "?", "•", "|", ")", "]", "%")
_JOIN_START_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789([{")
# 00: / 1. / 2) / a) 开头是列表项，不能和上一行合并
_LIST_ITEM_RE = re.compile(r"^\s*(?:\d{1,3}\s*[:.)]|[a-zA-Z][.)])\s")


def normalize_space(text: str) -> str:
    """把连续空白压成一个空格（用于标题、表格单元）。"""
    return re.sub(r"\s+", " ", text or "").strip()


def is_noise_line(line: str) -> bool:
    """页眉页脚、孤立的页码行。"""
    s = line.strip()
    if not s:
        return True
    return bool(PAGE_NO_RE.match(s) or DOCID_RE.match(s) or RM_RE.match(s))


def _merge_lines(lines: list[str]) -> str:
    """把同一段里被排版折断的行接回去。"""
    out: list[str] = []
    for raw in lines:
        line = raw.rstrip()
        if not line:
            continue
        if not out:
            out.append(line)
            continue

        prev = out[-1]
        if prev.startswith("#") or line.startswith("#"):
            out.append(line)
            continue
        if not _should_join(prev, line):
            out.append(line)
            continue

        if prev.endswith("-"):  # 连字符断词：inter-\nrupt -> interrupt
            out[-1] = prev[:-1] + line.lstrip()
        else:
            out[-1] = prev + " " + line.lstrip()
    return "\n".join(out)


def _should_join(prev: str, line: str) -> bool:
    if not prev or not line:
        return False
    if prev.rstrip().endswith(_SENTENCE_END):
        return False
    if line.lstrip()[:1] in {"•", "-", "|", "#"}:  # 新项目符号 / 表格行 / 标题，另起
        return False
    if re.match(r"^(table|figure)\s+\d", line, re.IGNORECASE):
        return False
    if _LIST_ITEM_RE.match(line):  # 00: Analog mode / 1. xxx / a) xxx 都是新条目
        return False
    first = line.lstrip()[:1]
    if first in _JOIN_START_CHARS:
        return True
    # 技术手册里大量续行以大写开头（寄存器名、APB2、USART…），
    # 上一行已经长到不像标题时也认为是续行。
    return len(prev) >= 60


def clean_text(text: str) -> str:
    """块级文本 -> 干净段落。表格段落（以 | 开头）原样保留。"""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_INVISIBLE)
    text = text.replace("\u00a0", " ")
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    for ch in _BULLET_CHARS:
        text = text.replace(ch, "•")

    paragraphs: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        if not para.strip():
            continue
        if para.lstrip().startswith("|"):  # Markdown 表格：保持原样
            paragraphs.append(para.strip())
            continue
        lines = [ln for ln in para.split("\n") if not is_noise_line(ln)]
        if not lines:
            continue
        merged = _merge_lines(lines)
        merged = re.sub(r"[ \t]{2,}", " ", merged)
        for piece in merged.split("\n"):
            piece = piece.strip()
            if piece:
                paragraphs.append(piece)
    return "\n\n".join(paragraphs)
