"""STM32F10xxx 参考手册 RAG —— 全局配置与模型工厂。

所有可调参数集中在这里，密钥一律从 week23/.env 读取，代码里不出现明文 Key。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- 路径
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent  # week25
DEFAULT_LAB_CONFIG_DIR = PROJECT_DIR.parent / "week23" / "20261027"

LAB_CONFIG_DIR = Path(
    os.getenv("LAB_CONFIG_DIR", str(DEFAULT_LAB_CONFIG_DIR))
).expanduser().resolve()

# 先加载课堂的 .env，否则下面这些模块级常量拿到的是空值。
# override=False：shell 里显式设置的变量优先，.env 只补缺。
try:
    from dotenv import load_dotenv  # noqa: PLC0415

    _ENV_FILE = LAB_CONFIG_DIR.parent / ".env"
    if _ENV_FILE.is_file():
        load_dotenv(_ENV_FILE, override=False)
except Exception:  # dotenv 没装也不能影响建库/检索
    pass

PDF_PATH = Path(
    os.getenv(
        "STM32_PDF",
        # 优先用 week25 下的本地副本，没有时回退到网盘路径
        str(PROJECT_DIR / "STM32F10xxx参考手册（英文）.pdf")
        if (PROJECT_DIR / "STM32F10xxx参考手册（英文）.pdf").is_file()
        else r"G:\BaiduNetdiskDownload\STM32入门教程资料\参考文档\STM32F10xxx参考手册（英文）.pdf",
    )
).expanduser()

# ---------------------------------------------------------------- 多语料（HAL 全家桶）
DOCS_DIR = Path(os.getenv("STM32_DOCS_DIR", str(PROJECT_DIR))).expanduser()
DS_PDF_PATH = Path(
    os.getenv("STM32_DS_PDF", str(DOCS_DIR / "STM32F103x8B数据手册（英文）.pdf"))
).expanduser()
UM_PDF_PATH = Path(
    os.getenv(
        "STM32_UM_PDF",
        str(DOCS_DIR / "um1850-description-of-stm32f1-hal-and-lowlayer-drivers-stmicroelectronics.pdf"),
    )
).expanduser()
HAL_ROOT = Path(os.getenv("STM32_HAL_ROOT", str(PROJECT_DIR / "vendor" / "STM32CubeF1"))).expanduser()
HAL_DRIVER_DIR = HAL_ROOT / "Drivers" / "STM32F1xx_HAL_Driver"
HAL_EXAMPLES_DIR = HAL_ROOT / "Projects" / "STM32F103RB-Nucleo" / "Examples"
HAL_VERSION = os.getenv("STM32_HAL_VERSION", "v1.1.10")

DS_COLLECTION = os.getenv("STM32_DS_COLLECTION", "stm32_ds_f103xb_text_embedding_v4")
UM_COLLECTION = os.getenv("STM32_UM_COLLECTION", "stm32_um1850_v3_text_embedding_v4")
HAL_COLLECTION = os.getenv("STM32_HAL_COLLECTION", "stm32f1_hal_v1110_text_embedding_v4")
EXAMPLES_COLLECTION = os.getenv("STM32_EXAMPLES_COLLECTION", "stm32f1_hal_examples_text_embedding_v4")


DB_DIR = Path(os.getenv("STM32_DB_DIR", str(PROJECT_DIR / "chroma_db"))).expanduser()
FIGURES_DIR = DB_DIR / "figures"  # 插图 PNG 与 manifest.json

# ---------------------------------------------------------------- 模型
EMBED_MODEL = os.getenv("STM32_EMBED_MODEL", "text-embedding-v4")
EMBED_DIM = int(os.getenv("STM32_EMBED_DIM", "1024"))
EMBED_BATCH = int(os.getenv("STM32_EMBED_BATCH", "10"))  # 百炼单请求上限
COLLECTION_NAME = os.getenv("STM32_COLLECTION", "stm32_rm0008_text_embedding_v4")

# 来源 -> 集合名：CLI / 路由 / 多语料检索共用
SOURCE_COLLECTIONS = {
    "rm": COLLECTION_NAME,
    "ds": DS_COLLECTION,
    "um": UM_COLLECTION,
    "hal": HAL_COLLECTION,
    "examples": EXAMPLES_COLLECTION,
}
# 默认 4096：带图问答时容易撞到上限，留足余量
ANSWER_MAX_TOKENS = int(os.getenv("STM32_ANSWER_MAX_TOKENS", "4096"))

# 生成/重排用哪个对话模型。嵌入模型（EMBED_MODEL）与此无关，不能换。
CHAT_MODEL = (
    os.getenv("STM32_CHAT_MODEL")
    or os.getenv("CHAT_MODEL")
    or "kimi-k3.1-pro"
)


def chat_model() -> str:
    """调用时才解析模型名：lab_config 导入后会再 load 一次 .env。"""
    _lab_config()
    return os.getenv("STM32_CHAT_MODEL") or os.getenv("CHAT_MODEL") or CHAT_MODEL


def thinking_enabled() -> bool:
    """kimi 默认会先输出一大段 reasoning；带图问答时很容易把 max_tokens 吃光导致正文为空。"""
    return (os.getenv("STM32_ENABLE_THINKING") or "0").strip() == "1"


# ---------------------------------------------------------------- 建库参数
@dataclass
class BuildConfig:
    """加载 + 切块参数。"""

    # 版心裁剪：实测页眉固定在 y≈58~70，页脚 y≈743~768，正文从 88 开始
    header_y: float = 74.0
    footer_y: float = 740.0

    # 切块：cl100k_base token 计数
    max_tokens: int = 480
    overlap_tokens: int = 64
    min_tokens: int = 70  # 小于此值的相邻小节并入上一块

    # 跳过 PDF 前 39 页的目录表（印刷页 1~39），它整页都是点线目录，属噪声
    skip_front_pages: int = 39

    # 只处理某段印刷页码（含首尾），None 表示全书。用于小范围试跑
    page_range: tuple[int, int] | None = None

    # 表格最多保留多少行，避免超长表把块撑爆
    max_table_rows: int = 60

    # 整章排除：Revision history 是 24 页的改版记录，对问答只有干扰
    exclude_chapter_patterns: tuple[str, ...] = (r"Revision history",)

    # 插图：把 Figure 题注下的矢量图抠成 PNG，并建成 kind=figure 的文本块
    include_figures: bool = True
    figure_dpi: int = 150
    min_figure_height: float = 60.0  # 图区太矮的当题注或表格线，跳过


# ---------------------------------------------------------------- 检索参数
@dataclass
class RetrieveConfig:
    """检索与重排参数。"""

    vector_k: int = 30  # 向量召回条数
    bm25_k: int = 30  # 关键词召回条数
    rrf_k: int = 60  # RRF 平滑常数
    rerank_pool: int = 14  # 送入重排的候选数
    final_k: int = 6  # 最终进 Prompt 的小节数（每个小节再补前后块）
    max_per_section: int = 3  # 同一小节最多保留几块：寄存器位定义表常常跨 3 块
    max_figures_in_context: int = 2  # 上下文里最多保留几张插图（问题点名的图不受限）
    device_boost: bool = True  # 问题里出现型号时，把召回向对应的密度章节偏一点
    use_rerank: bool = True  # LLM 重排开关
    expand_neighbors: bool = True  # 同一小节相邻块补全
    max_context_tokens: int = 6000  # 拼进 Prompt 的正文上限


# ---------------------------------------------------------------- 模型工厂
def _lab_config():
    """加载 week23/20261027/lab_config.py（课堂统一模型配置）。"""
    if not (LAB_CONFIG_DIR / "lab_config.py").is_file():
        raise FileNotFoundError(f"找不到模型配置：{LAB_CONFIG_DIR / 'lab_config.py'}")
    if str(LAB_CONFIG_DIR) not in sys.path:
        sys.path.insert(0, str(LAB_CONFIG_DIR))
    import lab_config  # noqa: PLC0415

    if lab_config.PROVIDER != "qwen":
        raise ValueError(
            f"本模块需要 LLM_PROVIDER=qwen（当前 {lab_config.PROVIDER}），"
            "因为嵌入模型走百炼 text-embedding-v4。"
        )
    return lab_config


def get_embeddings():
    """云端嵌入模型。维度必须与建库时一致，否则查询会报维度不符。"""
    lab = _lab_config()
    from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415

    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        base_url=lab.BASE_URL,
        api_key=lab.API_KEY,
        check_embedding_ctx_length=False,  # 让接口自己按字符切，不做本地 token 校验
        dimensions=EMBED_DIM,
        chunk_size=EMBED_BATCH,
    )


def get_llm(**overrides):
    """对话模型：走百炼兼容模式，默认 kimi（.env 里的 CHAT_MODEL）。

    不走 lab_config.get_model()，因为那是 week23 课堂的默认模型（qwen3.8-flash）；
    嵌入仍然用同一个百炼 key 的 text-embedding-v4。
    """
    lab = _lab_config()
    from langchain_openai import ChatOpenAI  # noqa: PLC0415

    params = {
        "model": chat_model(),
        "base_url": lab.DASHSCOPE_BASE_URL,
        "api_key": lab.DASHSCOPE_API_KEY,
        "temperature": 0,
        "max_tokens": ANSWER_MAX_TOKENS,
    }
    if not thinking_enabled():
        params["extra_body"] = {"enable_thinking": False}
    params.update(overrides)
    return ChatOpenAI(**params)


def collection_path(collection: str = COLLECTION_NAME) -> Path:
    return DB_DIR / collection


def describe() -> str:
    try:
        model = chat_model()
    except Exception:
        model = CHAT_MODEL
    return (
        f"PDF        : {PDF_PATH}\n"
        f"数据库目录 : {DB_DIR}\n"
        f"集合名     : {COLLECTION_NAME}\n"
        f"嵌入模型   : {EMBED_MODEL} / {EMBED_DIM} 维\n"
        f"对话模型   : {model}\n"
        f"插图目录   : {FIGURES_DIR}\n"
        f"模型配置   : {LAB_CONFIG_DIR}"
    )
