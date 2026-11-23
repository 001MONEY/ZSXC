"""STM32F10xxx 参考手册（RM0008）RAG + AI 开发助手。

知识库用法（检索问答）：

    from stm32_rag import Stm32Index, HybridRetriever, answer

    idx = Stm32Index()
    idx.build()                       # 首次建库（已写入的块会自动跳过）

    retriever = HybridRetriever()
    result = answer("GPIOx_CRL 的复位值是多少？", retriever=retriever)
    print(result.text)
    print(*result.sources(), sep="\\n")

开发助手（需求 -> 可编译工程）：

    from stm32_rag import develop, DevAssistant

    delivery = develop("配置 PA8 输出 1MHz PWM")   # 确定性链路，含编译
    print(delivery.summary())

    assistant = DevAssistant()
    result = assistant.run("配置 PA8 输出 1MHz PWM")  # LLM 编排，失败自动回退
"""

from .core.chunker import build_documents, chunk_sections, count_tokens, pack_text
from .core.config import BuildConfig, RetrieveConfig, describe, get_embeddings, get_llm

from .corpus.figures import Figure, extract_figures, figure_documents
from .corpus.index import Stm32Index
from .corpus.loader import Section, load_sections

from .rag.answer import RagAnswer, answer, format_context
from .rag.evaluate import check_anchors, load_cases, run_eval
from .rag.retrieve import BM25, Hit, HybridRetriever, tokenize

from .graphs.graph import GraphResult, GraphStore, graph_available
from .graphs.graph_docs import DocGraphResult, DocGraphStore, doc_graph_available

from .gen.codegen import (
    AdcSpec,
    AppSpec,
    BuzzerSpec,
    CodePlan,
    GpioSpec,
    KeySpec,
    OledSpec,
    PwmSpec,
    SerialCmdSpec,
    TimerSpec,
    UartSpec,
    plan_application,
)
from .gen.cubemx import CubeMXInfo, PatchResult, apply_to_project, parse_ioc, find_ioc
from .gen.mcu import Device, Pin, pwm_timing, resolve_device, timer_clock, uart_brr
from .gen.project import KeilProject, ProjectFile, find_project
from .gen.scaffold import scaffold_project

from .build.keil import BuildIssue, BuildResult, KeilBuilder, build_project, parse_log, suggest_fixes
from .build.pipeline import Delivery, develop, parse_requirement

from .assistant.agent import AgentResult, DevAssistant
from .assistant.tools import WorkspaceSession

__all__ = [
    "AdcSpec",
    "AgentResult",
    "AppSpec",
    "BM25",
    "BuildConfig",
    "BuildIssue",
    "BuildResult",
    "BuzzerSpec",
    "CodePlan",
    "CubeMXInfo",
    "Delivery",
    "DevAssistant",
    "Device",
    "DocGraphResult",
    "DocGraphStore",
    "Figure",
    "GpioSpec",
    "GraphResult",
    "GraphStore",
    "Hit",
    "HybridRetriever",
    "KeilBuilder",
    "KeilProject",
    "KeySpec",
    "OledSpec",
    "Pin",
    "ProjectFile",
    "PatchResult",
    "PwmSpec",
    "RagAnswer",
    "RetrieveConfig",
    "Section",
    "SerialCmdSpec",
    "Stm32Index",
    "TimerSpec",
    "UartSpec",
    "WorkspaceSession",
    "answer",
    "apply_to_project",
    "build_documents",
    "build_project",
    "check_anchors",
    "chunk_sections",
    "count_tokens",
    "describe",
    "develop",
    "doc_graph_available",
    "extract_figures",
    "figure_documents",
    "find_ioc",
    "find_project",
    "format_context",
    "get_embeddings",
    "get_llm",
    "graph_available",
    "load_cases",
    "load_sections",
    "pack_text",
    "parse_ioc",
    "parse_log",
    "parse_requirement",
    "plan_application",
    "pwm_timing",
    "resolve_device",
    "run_eval",
    "scaffold_project",
    "suggest_fixes",
    "timer_clock",
    "tokenize",
    "uart_brr",
]
