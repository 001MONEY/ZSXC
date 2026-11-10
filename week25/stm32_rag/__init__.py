"""STM32F10xxx 参考手册（RM0008）RAG 模块。

典型用法：

    from stm32_rag import Stm32Index, HybridRetriever, answer

    idx = Stm32Index()
    idx.build()                       # 首次建库（已写入的块会自动跳过）

    retriever = HybridRetriever()
    result = answer("GPIOx_CRL 的复位值是多少？", retriever=retriever)
    print(result.text)
    print(*result.sources(), sep="\n")
"""

from .answer import RagAnswer, answer, format_context
from .chunker import build_documents, chunk_sections, count_tokens, pack_text
from .config import BuildConfig, RetrieveConfig, describe, get_embeddings, get_llm
from .evaluate import check_anchors, load_cases, run_eval
from .figures import Figure, extract_figures, figure_documents
from .index import Stm32Index
from .loader import Section, load_sections
from .retrieve import BM25, Hit, HybridRetriever, tokenize

__all__ = [
    "BM25",
    "BuildConfig",
    "Figure",
    "Hit",
    "HybridRetriever",
    "RagAnswer",
    "RetrieveConfig",
    "Section",
    "Stm32Index",
    "answer",
    "build_documents",
    "check_anchors",
    "chunk_sections",
    "count_tokens",
    "describe",
    "extract_figures",
    "figure_documents",
    "format_context",
    "get_embeddings",
    "get_llm",
    "load_cases",
    "load_sections",
    "pack_text",
    "run_eval",
    "tokenize",
]
