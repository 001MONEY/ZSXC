"""把检索结果拼成 Prompt，调用大模型生成带出处的答案。"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path

from .config import FIGURES_DIR, get_llm
from .retrieve import Hit, HybridRetriever

SYSTEM_PROMPT = """你是 STM32F10xxx 参考手册（RM0008）的检索问答助手。

规则：
1. 只依据用户消息里【资料】的内容回答，不要引入手册之外的知识。
2. 每条结论后用 [编号] 标注出处，编号对应资料片段的序号。
3. 寄存器名、位域名、地址偏移、复位值必须原样照抄，不要翻译、不要改写大小写。
4. 位域编码（如 PLLMUL=0111 表示 x9、MODE=11 表示 50MHz）必须从资料表格里原样摘录；
   资料里没列出对应编码就写“资料中未给出该编码”，绝对不要自己推算或猜测。
5. 手册按器件密度分章（例如“7 Low-, medium-, high- and XL-density”与“8 Connectivity line devices”
   对同一寄存器各有一套描述）。资料同时给出多个版本时：用户说明了型号/密度就只讲对应版本；
   没说就用一句话点明差异，再给通用结论，不要把两版描述各抄一遍。
6. 只引用真正支撑结论的片段；同一小节的多个片段合并引用，不要逐条复述全部资料。
7. 直接给结论和配置方法，不要输出分析、推理、自我修正的过程，也不要怀疑资料是否被截断；
   资料里没写到的部分就说“资料中未给出”，不要推测性的标注。
8. 型号与章节对应：第 7 章（Low-, medium-, high- and XL-density）适用于 STM32F101/102/103/104，
   第 8 章（Connectivity line）适用于带以太网/USB OTG 的 STM32F105/107。
   问题给出型号时，只用以对应版本的资料作答；资料里只有另一版本时才说明“这是 F105/107 的描述”。
9. 资料不足以回答时，先直接说“手册的检索结果中没有找到……”，再说明手册里可能相关的章节，不要猜测。
10. 默认用中文回答；寄存器/位域/缩写保留英文原文，必要时在括号里给中文解释。"""


@dataclass
class RagAnswer:
    question: str
    text: str
    hits: list[Hit]

    def sources(self) -> list[str]:
        """按小节合并出处：同节的多个块显示成一行，引用编号并列。"""
        groups: list[dict] = []
        lookup: dict[str, dict] = {}
        for i, hit in enumerate(self.hits, start=1):
            meta = hit.metadata
            path = str(meta.get("path") or meta.get("title", ""))
            group = lookup.get(path)
            if group is None:
                group = {"path": path, "ids": [], "pages": []}
                lookup[path] = group
                groups.append(group)
            group["ids"].append(i)
            group["pages"].extend(_page_numbers(str(meta.get("pages", ""))))

        lines: list[str] = []
        for group in groups:
            ids = ",".join(str(i) for i in group["ids"])
            pages = group["pages"]
            if pages:
                span = f"p.{min(pages)}-{max(pages)}" if len(pages) > 1 else f"p.{pages[0]}"
                page_text = f"｜{span}"
            else:
                page_text = ""
            merged = f"（同节 {len(group['ids'])} 块）" if len(group["ids"]) > 1 else ""
            lines.append(f"[{ids}] {group['path']}{page_text}{merged}")
        return lines


def _page_numbers(pages: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", pages or "")]


def format_context(hits: list[Hit]) -> str:
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        meta = hit.metadata
        header = (
            f"[{i}] 小节：{meta.get('title', '')}"
            f"｜路径：{meta.get('path', '')}"
            f"｜页码：{meta.get('pages', '')}"
        )
        blocks.append(f"{header}\n{hit.text}")
    return "\n\n".join(blocks)


def answer(
    question: str,
    retriever: HybridRetriever | None = None,
    llm=None,
    hits: list[Hit] | None = None,
    with_figures: bool = False,
    max_figures: int = 2,
) -> RagAnswer:
    retriever = retriever or HybridRetriever()
    if hits is None:
        hits = retriever.search(question)
    if not hits:
        return RagAnswer(question=question, text="向量库里没有检索到相关内容。", hits=[])

    model = llm or retriever.llm or get_llm()
    user_message = f"【资料】\n{format_context(hits)}\n\n【问题】{question}"

    content: object = user_message
    if with_figures:
        images = _figure_parts(hits, max_figures)
        if images:
            content = [
                {
                    "type": "text",
                    "text": user_message
                    + "\n\n【提示】随消息附上的是手册原图，图内的信号名、分频比、数值以图为准，"
                    "回答时注明对应的 Figure 编号。",
                }
            ] + images

    response = model.invoke([("system", SYSTEM_PROMPT), ("user", content)])
    return RagAnswer(question=question, text=(response.content or "").strip(), hits=hits)


def _data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"


def _figure_parts(hits: list[Hit], limit: int) -> list[dict]:
    """把命中的插图读成 data URL，最多 limit 张。"""
    parts: list[dict] = []
    for hit in hits:
        if hit.metadata.get("kind") != "figure":
            continue
        path = FIGURES_DIR / str(hit.metadata.get("image_file", ""))
        if not path.is_file():
            continue
        parts.append({"type": "image_url", "image_url": {"url": _data_url(path)}})
        if len(parts) >= limit:
            break
    return parts
