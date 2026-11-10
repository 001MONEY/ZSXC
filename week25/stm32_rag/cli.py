"""命令行入口：

    python -m stm32_rag inspect                 # 体检 PDF + 预览切块
    python -m stm32_rag build --rebuild         # 全量建库
    python -m stm32_rag build --pages 160-200   # 只处理某几页（试跑）
    python -m stm32_rag stats                   # 看库里有多少块
    python -m stm32_rag ask "GPIOx_CRL 的复位值是多少？"
    python -m stm32_rag eval                    # 跑锚点评测
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from .config import COLLECTION_NAME, PDF_PATH, BuildConfig, RetrieveConfig, describe
from .index import Stm32Index


def parse_pages(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    if "-" in value:
        lo, hi = value.split("-", 1)
        return int(lo), int(hi)
    page = int(value)
    return page, page


def cmd_inspect(args) -> int:
    from .chunker import build_documents
    from .loader import summarize

    print(describe())
    cfg = BuildConfig(page_range=parse_pages(args.pages))
    sections, documents = build_documents(PDF_PATH, cfg)
    print("\n" + summarize(sections) + f"，切块 {len(documents)} 个")

    chapter_counts = Counter(s.chapter for s in sections)
    print(f"\n覆盖 {len(chapter_counts)} 章：")
    for chapter, count in chapter_counts.most_common(12):
        print(f"  {count:4d} 节  {chapter[:70]}")

    if args.show:
        for doc in documents[: args.show]:
            meta = doc.metadata
            print("\n" + "─" * 78)
            print(f"{doc.id} | {meta['title'][:64]} | p.{meta['pages']} | {meta['tokens']} tok | 表={meta['has_table']}")
            print(f"symbols: {meta['symbols'][:100]}")
            print(doc.page_content[:700])
    return 0


def cmd_build(args) -> int:
    index = Stm32Index(collection=args.collection)
    cfg = BuildConfig(page_range=parse_pages(args.pages), include_figures=not args.no_figures)
    result = index.build(PDF_PATH, cfg=cfg, rebuild=args.rebuild, workers=args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(index.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_figures(args) -> int:
    from .figures import extract_figures, summarize

    cfg = BuildConfig(page_range=parse_pages(args.pages), figure_dpi=args.dpi)
    figures = extract_figures(PDF_PATH, cfg, force=args.force)
    print(summarize(figures))
    for fig in figures[: args.show]:
        print(f"  Figure {fig.number:<3} p.{fig.page:<5} {fig.width}x{fig.height}  {fig.title[:58]}")
    return 0


def cmd_stats(args) -> int:
    index = Stm32Index(collection=args.collection)
    print(describe())
    print(json.dumps(index.stats(), ensure_ascii=False, indent=2))
    return 0


def _retriever(args):
    from .retrieve import HybridRetriever

    index = Stm32Index(collection=args.collection)
    cfg = RetrieveConfig()
    if getattr(args, "no_rerank", False):
        cfg.use_rerank = False
    if getattr(args, "no_expand", False):
        cfg.expand_neighbors = False
    if getattr(args, "k", None):
        cfg.final_k = args.k
    if getattr(args, "vector_k", None):
        cfg.vector_k = args.vector_k
    return HybridRetriever(index=index, cfg=cfg)


def cmd_ask(args) -> int:
    from .answer import answer

    retriever = _retriever(args)
    result = answer(args.question, retriever=retriever, with_figures=args.with_figures)
    print("\n【答】\n" + result.text)
    print("\n【出处】")
    for line in result.sources():
        print("  " + line)
    if args.show_context:
        print("\n【送入模型的资料】")
        for i, hit in enumerate(result.hits, start=1):
            print(f"\n[{i}] {hit.label}  向量排名={hit.vector_rank} 关键词排名={hit.bm25_rank} 重排={hit.rerank_score}")
            print(hit.text[:800])
    return 0


def cmd_eval(args) -> int:
    from .evaluate import check_anchors, load_cases, run_eval

    cases = load_cases(args.cases) if args.cases else load_cases()
    retriever = _retriever(args)
    if args.check:
        corpus = retriever.ensure_ready()
        missing = check_anchors(cases, corpus.texts)
        if missing:
            print(f"有 {len(missing)} 个锚点在语料里找不到（题目写错了或该书没这段）：")
            for question, anchor in missing:
                print(f"  - {question} / {anchor!r}")
        else:
            print("全部锚点都能在语料中命中。")
    summary = run_eval(retriever, cases, k=args.k)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 GBK，输出里的符号会直接报 UnicodeEncodeError
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(prog="stm32_rag", description="STM32F10xxx 参考手册 RAG")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Chroma 集合名")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="体检 PDF 并预览切块")
    p_inspect.add_argument("--pages", help="印刷页码范围，如 160-200")
    p_inspect.add_argument("--show", type=int, default=0, help="打印前 N 个块")
    p_inspect.set_defaults(func=cmd_inspect)

    p_build = sub.add_parser("build", help="建库")
    p_build.add_argument("--pages", help="印刷页码范围，如 160-200")
    p_build.add_argument("--rebuild", action="store_true", help="重建集合（先写临时集合，成功后再替换）")
    p_build.add_argument("--workers", type=int, default=4, help="并发嵌入线程数")
    p_build.add_argument("--no-figures", action="store_true", help="不建插图块（只建正文）")
    p_build.set_defaults(func=cmd_build)

    p_fig = sub.add_parser("figures", help="抽插图（调试用）")
    p_fig.add_argument("--pages", help="印刷页码范围")
    p_fig.add_argument("--dpi", type=int, default=150)
    p_fig.add_argument("--force", action="store_true", help="忽略 manifest 缓存重抽")
    p_fig.add_argument("--show", type=int, default=5, help="打印前 N 张")
    p_fig.set_defaults(func=cmd_figures)

    p_stats = sub.add_parser("stats", help="查看集合状态")
    p_stats.set_defaults(func=cmd_stats)

    p_ask = sub.add_parser("ask", help="提问")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=None, help="最终片段数")
    p_ask.add_argument("--vector-k", type=int, default=None)
    p_ask.add_argument("--no-rerank", action="store_true")
    p_ask.add_argument("--no-expand", action="store_true")
    p_ask.add_argument("--with-figures", action="store_true", help="命中插图时把原图随消息发给视觉模型")
    p_ask.add_argument("--show-context", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="锚点评测")
    p_eval.add_argument("--cases", help="评测集路径")
    p_eval.add_argument("-k", type=int, default=None)
    p_eval.add_argument("--vector-k", type=int, default=None)
    p_eval.add_argument("--no-rerank", action="store_true")
    p_eval.add_argument("--no-expand", action="store_true")
    p_eval.add_argument("--check", action="store_true", help="先校验锚点是否在语料中")
    p_eval.add_argument("--json", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
