"""多语料路由检索：按问题选集合 -> 各自混合召回 -> 跨集合 RRF -> 统一重排。

三份语料回答的是不同层面的问题，混在一个集合里会互相污染：
- RM0008：寄存器/位域/复位值（"RCC_APB2ENR 的 IOPAEN 是哪一位"）
- UM1850：HAL/LL API 的用途、参数、返回值（"HAL_GPIO_Init 的参数是什么"）
- HAL 源码：实现细节、`@retval` 分支、宏展开（"__HAL_RCC_GPIOA_CLK_ENABLE 展开成什么"）
- 例程：标准用法（"ADC 轮询怎么启动"）
- 数据手册：电气参数、引脚、外设数量（"F103C8T6 有多少个定时器"）

路由用规则而不是分类模型：符号名和寄存器名是强特征，规则准确且零延迟。
"""

from __future__ import annotations

import re
from collections import defaultdict
from types import SimpleNamespace

from .config import SOURCE_COLLECTIONS, RetrieveConfig, get_llm
from .index import Stm32Index
from .retrieve import Hit, HybridRetriever, rerank_hits

_SYMBOL_RE = re.compile(r"\b(?:HAL|LL)_[A-Za-z0-9_]+\b")
_REGISTER_RE = re.compile(
    r"寄存器|位域|位定义|哪一位|复位值|地址|偏移|offset|reset value|"
    r"\b0x[0-9A-Fa-f]{4,}\b|\b[A-Z]{2,10}x?_(?:CR[LH1-4]?|SR|DR|BRR|ARR|CCR|CCMR\d?|PSC|BSRR|ODR|IDR|APB\d?ENR|CFGR|ACR|CSR)\b"
)
_DATASHEET_RE = re.compile(
    r"电压|电流|功耗|温度|封装|引脚|管脚|主频|Flash|RAM|电气|特性表|绝对最大|"
    r"数据手册|驱动能力|VDD|VSS|VIL|VIH|MHz|MSPS|Kbyte|字节|"
    r"多少个|多少个定时器|外设数量",
    re.IGNORECASE,
)
_CHIP_RE = re.compile(r"\bSTM32[A-Z]\d{3}[A-Z0-9]{2,}\b", re.IGNORECASE)
_CODE_API_RE = re.compile(
    r"函数|回调|宏|原型|参数|返回值|怎么用|如何调用|例程|示例|API|使能|初始化|配置步骤|"
    r"怎么配置|如何配置|配置流程|使用步骤",
    re.IGNORECASE,
)
_FUNC_Q = re.compile(r"函数|哪个函数|回调|调用|实现|返回值")
_DEF_Q = re.compile(r"结构体|成员|字段|类型定义|枚举")
_MACRO_Q = re.compile(r"宏|define", re.IGNORECASE)

# 中文问法 -> 英文 API 关键词：代码语料是英文的，不做这层映射，
# “高电平”和“TogglePin”在向量/BM25 里几乎无法区分。
_EXPANSIONS = (
    ("高电平", ("SET", "high")),
    ("低电平", ("RESET", "low")),
    ("翻转", ("Toggle",)),
    ("读取", ("Read", "GetState")),
    ("输入", ("Read", "input")),
    ("输出", ("Write", "output")),
    ("外部中断", ("EXTI", "IRQHandler", "Callback")),
    ("中断", ("IRQHandler", "Callback")),
    ("串口", ("UART", "USART")),
    ("定时器", ("TIM", "Timer")),
    ("看门狗", ("IWDG", "WWDG")),
    ("时钟使能", ("CLK_ENABLE",)),
    ("初始化", ("Init",)),
    ("复位", ("DeInit", "RESET")),
)


def expand_query(query: str) -> str:
    """给中文问题补上英文 API 关键词（只用于召回与重排）。"""
    extra: list[str] = []
    for zh, words in _EXPANSIONS:
        if zh in query:
            extra.extend(words)
    return query + (" " + " ".join(dict.fromkeys(extra)) if extra else "")


def _kind_bonus(query: str, hit: Hit) -> float:
    """重排后的规则分：问函数就抬函数块，问结构体就抬 typedef，避免同类噪声抢位。"""
    kind = str(hit.metadata.get("kind", ""))
    bonus = 0.0
    if _FUNC_Q.search(query) and kind == "function":
        bonus += 2.0
    if _DEF_Q.search(query) and kind == "typedef":
        bonus += 2.0
    if _MACRO_Q.search(query) and kind in ("macro", "define"):
        bonus += 2.0
    return bonus

DEFAULT_SOURCES = ("rm", "ds", "um", "hal", "examples")


def route(query: str) -> list[str]:
    """按问题特征给出优先级顺序；调用方再按集合是否存在过滤。"""
    if _SYMBOL_RE.search(query):
        return ["hal", "um", "examples"]
    if _REGISTER_RE.search(query):
        return ["rm", "hal"]
    if _CHIP_RE.search(query) or _DATASHEET_RE.search(query):
        return ["ds", "rm", "hal"]
    if _CODE_API_RE.search(query):
        return ["hal", "um", "examples"]
    return ["rm", "hal", "ds"]


def _collapse_sections(hits: list[Hit]) -> list[Hit]:
    """同一小节只留最相关的一块（其余由邻居补全负责）。"""
    seen: set[str] = set()
    kept: list[Hit] = []
    for hit in hits:
        section_id = str(hit.metadata.get("section_id", ""))
        if section_id in seen:
            continue
        seen.add(section_id)
        kept.append(hit)
    return kept


class MultiSourceRetriever:
    """按规则路由到多个集合，召回后跨集合 RRF 融合再统一重排。"""

    def __init__(
        self,
        sources: list[str] | None = None,
        cfg: RetrieveConfig | None = None,
        llm=None,
        db_dir=None,
    ) -> None:
        self.sources = list(sources or DEFAULT_SOURCES)
        self.cfg = cfg or RetrieveConfig()
        self._llm = llm
        self._db_dir = db_dir
        self._retrievers: dict[str, HybridRetriever] = {}
        self._unavailable: set[str] = set()

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def _retriever(self, name: str) -> HybridRetriever | None:
        if name in self._retrievers:
            return self._retrievers[name]
        if name in self._unavailable or name not in SOURCE_COLLECTIONS:
            return None
        index = Stm32Index(collection=SOURCE_COLLECTIONS[name], db_dir=self._db_dir) if self._db_dir else Stm32Index(
            collection=SOURCE_COLLECTIONS[name]
        )
        if not index.exists() or index.count() == 0:
            self._unavailable.add(name)
            return None
        retriever = HybridRetriever(index=index, cfg=self.cfg, llm=self._llm)
        self._retrievers[name] = retriever
        return retriever

    # ------------------------------------------------------------ 路由
    def plan(self, query: str) -> list[str]:
        names = [n for n in route(query) if self._retriever(n) is not None]
        if not names:  # 规则命中的集合都没建，退回全部可用集合
            names = [n for n in self.sources if self._retriever(n) is not None]
        return names

    def ensure_ready(self):
        """把各集合语料拼起来，供 eval 的锚点校验使用。"""
        texts: list[str] = []
        ids: list[str] = []
        metas: list[dict] = []
        for name in self.sources:
            retriever = self._retriever(name)
            if retriever is None:
                continue
            corpus = retriever.ensure_ready()
            ids.extend(corpus.ids)
            texts.extend(corpus.texts)
            metas.extend(corpus.metas)
        return SimpleNamespace(ids=ids, texts=texts, metas=metas)

    def search(self, query: str, cfg: RetrieveConfig | None = None) -> list[Hit]:
        cfg = cfg or self.cfg
        names = self.plan(query)
        search_query = expand_query(query)

        bonus: dict[str, float] = defaultdict(float)
        merged: dict[str, Hit] = {}
        # 路由顺序即重要程度：第一位 1.0，之后依次衰减，
        # 否则例程库的函数块会把 HAL 源码的函数挤出重排池。
        weights = [1.0, 0.6, 0.4, 0.3, 0.2]
        for position, name in enumerate(names):
            weight = weights[min(position, len(weights) - 1)]
            retriever = self._retrievers[name]
            ranked = retriever.recall(search_query, cfg)[: cfg.rerank_pool]
            for rank, hit in enumerate(ranked, start=1):
                hit.metadata = {**hit.metadata, "source_key": name}
                merged.setdefault(hit.id, hit)
                # 规则加权：问函数时函数块轻微上提（量级约等于一个名次贡献）
                bonus[hit.id] += weight * (1.0 / (cfg.rrf_k + rank) + 0.008 * _kind_bonus(query, hit))

        candidates = sorted(merged.values(), key=lambda h: bonus[h.id], reverse=True)[: cfg.rerank_pool]
        for hit in candidates:
            hit.score = bonus[hit.id]
        if cfg.use_rerank:
            # 扩展词只用于召回：给重排模型看原始问题，否则附加的英文关键词会干扰判断
            candidates = rerank_hits(self.llm, query, candidates)
            candidates = sorted(
                candidates,
                key=lambda h: (h.rerank_score if h.rerank_score is not None else 0.0) + _kind_bonus(query, h),
                reverse=True,
            )

        top = _collapse_sections(candidates)[: cfg.final_k]
        if cfg.expand_neighbors:
            top = self._expand(top, cfg)
        return top

    def _expand(self, hits: list[Hit], cfg: RetrieveConfig) -> list[Hit]:
        """同节邻居补全：寄存器编码表/枚举常常就落在相邻块里。"""
        result: list[Hit] = []
        used = 0
        seen: set[str] = set()
        for hit in hits:
            size = int(hit.metadata.get("tokens", 0) or 0)
            if result and used + size > cfg.max_context_tokens:
                break
            result.append(hit)
            seen.add(hit.id)
            used += size

            source = hit.metadata.get("source_key")
            retriever = self._retrievers.get(source)
            if retriever is None:
                continue
            for extra in retriever._expand([hit], cfg.max_context_tokens - used):
                if extra.id in seen:
                    continue
                extra.metadata = {**extra.metadata, "source_key": source}
                extra_size = int(extra.metadata.get("tokens", 0) or 0)
                if used + extra_size > cfg.max_context_tokens:
                    continue
                seen.add(extra.id)
                result.append(extra)
                used += extra_size
        return result

    def describe(self) -> str:
        available = [n for n in self.sources if self._retriever(n) is not None]
        return "可用集合：" + ("、".join(available) if available else "（无）")
