"""命令行入口：

    python -m stm32_rag inspect                 # 体检 PDF + 预览切块
    python -m stm32_rag build --rebuild         # 全量建库
    python -m stm32_rag build --pages 160-200   # 只处理某几页（试跑）
    python -m stm32_rag stats                   # 看库里有多少块
    python -m stm32_rag ask "GPIOx_CRL 的复位值是多少？"
    python -m stm32_rag eval                    # 跑锚点评测

开发助手：

    python -m stm32_rag scaffold --dest workspaces/demo          # 生成可编译工程
    python -m stm32_rag codegen "配置 PA8 输出 1MHz PWM"         # 只看规格与代码
    python -m stm32_rag develop "配置 PA8 输出 1MHz PWM"         # 确定性全链路（含编译）
    python -m stm32_rag agent "配置 PA8 输出 1MHz PWM"           # LLM 编排（失败自动回退）
    python -m stm32_rag compile --workspace workspaces/demo      # 只编译（别名 keil）
    python -m stm32_rag project --workspace workspaces/demo      # 看工程结构
    python -m stm32_rag graph --build                            # 建器件知识图谱（Neo4j）
    python -m stm32_rag graph --pin PA8                          # 查引脚复用 / DMA / 中断
    python -m stm32_rag graph-docs --build                       # 建文档关系图（GraphRAG）
    python -m stm32_rag ask "问题" --graph-expand                # 检索时做跨语料扩展
    python -m stm32_rag mcp --workspace workspaces/demo          # 启动 Keil MCP Server
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path

from .core.config import (
    DS_PDF_PATH,
    PDF_PATH,
    SOURCE_COLLECTIONS,
    BuildConfig,
    RetrieveConfig,
    describe,
)
from .corpus.index import Stm32Index


def ds_config(page_range=None) -> BuildConfig:
    """数据手册：无目录页噪点，页脚在 743；不抽插图（与 RM0008 共用图目录会撞 Figure 编号）。"""
    return BuildConfig(
        skip_front_pages=1,
        header_y=74.0,
        footer_y=740.0,
        include_figures=False,
        exclude_chapter_patterns=(r"Revision history",),
        page_range=page_range,
    )


def source_documents(source: str, pages=None, no_figures: bool = False):
    """按来源产出 (小节/统计文案, Document 列表)。"""
    if source == "rm":
        from .core.chunker import build_documents

        cfg = BuildConfig(page_range=pages, include_figures=not no_figures)
        sections, docs = build_documents(PDF_PATH, cfg, source_name=PDF_PATH.name, doc="RM0008")
        return f"RM0008：小节 {len(sections)}", docs
    if source == "ds":
        from .core.chunker import build_documents

        cfg = ds_config(pages)
        sections, docs = build_documents(DS_PDF_PATH, cfg, source_name=DS_PDF_PATH.name, doc="DS")
        return f"DS：小节 {len(sections)}", docs
    if source == "um":
        from .corpus.loader_um import summarize, um_config, um_documents

        docs = um_documents(cfg=dataclasses.replace(um_config(), page_range=pages))
        return summarize(docs), docs
    if source == "hal":
        from .corpus.loader_hal import hal_documents, summarize

        docs = hal_documents()
        return summarize(docs), docs
    if source == "examples":
        from .corpus.loader_hal import examples_documents, summarize

        docs = examples_documents()
        return summarize(docs), docs
    raise ValueError(f"未知来源：{source}")


def parse_pages(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    if "-" in value:
        lo, hi = value.split("-", 1)
        return int(lo), int(hi)
    page = int(value)
    return page, page


def cmd_inspect(args) -> int:
    print(describe())
    summary, documents = source_documents(args.source, parse_pages(args.pages), args.no_figures)
    print(f"\n{summary}，切块 {len(documents)} 个")

    chapter_counts = Counter(d.metadata.get("chapter", "") for d in documents)
    print(f"\n覆盖 {len(chapter_counts)} 章/模块：")
    for chapter, count in chapter_counts.most_common(12):
        print(f"  {count:4d} 块  {chapter[:70]}")

    if args.show:
        for doc in documents[: args.show]:
            meta = doc.metadata
            print("\n" + "─" * 78)
            print(f"{doc.id} | {meta.get('title', '')[:64]} | {meta.get('kind')} | {meta.get('tokens')} tok")
            print(f"symbols: {meta.get('symbols', '')[:100]}")
            print(doc.page_content[:700])
    return 0


def cmd_build(args) -> int:
    source = args.source
    collection = args.collection or SOURCE_COLLECTIONS[source]
    index = Stm32Index(collection=collection)

    if source in ("rm", "ds"):
        pdf = PDF_PATH if source == "rm" else DS_PDF_PATH
        cfg = (
            BuildConfig(page_range=parse_pages(args.pages), include_figures=not args.no_figures)
            if source == "rm"
            else ds_config(parse_pages(args.pages))
        )
        result = index.build(
            pdf,
            cfg=cfg,
            rebuild=args.rebuild,
            workers=args.workers,
            source_name=pdf.name,
            doc="RM0008" if source == "rm" else "DS",
        )
    else:
        _, documents = source_documents(source, parse_pages(args.pages), args.no_figures)
        result = index.add_documents(documents, rebuild=args.rebuild, workers=args.workers)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(index.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_figures(args) -> int:
    from .corpus.figures import extract_figures, summarize

    cfg = BuildConfig(page_range=parse_pages(args.pages), figure_dpi=args.dpi)
    figures = extract_figures(PDF_PATH, cfg, force=args.force)
    print(summarize(figures))
    for fig in figures[: args.show]:
        print(f"  Figure {fig.number:<3} p.{fig.page:<5} {fig.width}x{fig.height}  {fig.title[:58]}")
    return 0


def cmd_stats(args) -> int:
    if args.collection:
        collection = args.collection
    elif args.source in SOURCE_COLLECTIONS:
        collection = SOURCE_COLLECTIONS[args.source]
    else:
        from .rag.ingest import load_registry

        collection = str(load_registry().get(args.source, {}).get("collection", "")) or args.source
    index = Stm32Index(collection=collection)
    print(describe())
    print(json.dumps(index.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_ingest(args) -> int:
    from .rag.ingest import ingest, ingest_batch

    pdfs = args.pdf or []
    common = dict(
        collection=args.collection,
        tier=args.tier,
        rebuild=args.rebuild,
        workers=args.workers,
        pages=args.pages,
        force=args.force,
    )
    if args.md:
        result = ingest(md=args.md, **common)
    elif len(pdfs) == 1:
        result = ingest(pdf=pdfs[0], **common)
    elif pdfs:
        result = ingest_batch(
            pdfs,
            tier=args.tier,
            pages=args.pages,
            rebuild=args.rebuild,
            workers=args.workers,
            force=args.force,
        )
    else:
        raise SystemExit("请提供 PDF，或用 --md 指定 Markdown。")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _retriever(args):
    from .rag.retrieve import HybridRetriever

    cfg = RetrieveConfig()
    if getattr(args, "no_rerank", False):
        cfg.use_rerank = False
    if getattr(args, "no_expand", False):
        cfg.expand_neighbors = False
    if getattr(args, "graph_expand", False):
        cfg.graph_expand = True
    if getattr(args, "k", None):
        cfg.final_k = args.k
    if getattr(args, "vector_k", None):
        cfg.vector_k = args.vector_k

    spec = (getattr(args, "sources", None) or "auto").strip()
    if spec == "auto":
        from .rag.router import MultiSourceRetriever

        return MultiSourceRetriever(cfg=cfg)

    names = [s.strip() for s in spec.split(",") if s.strip()]
    if len(names) == 1:
        return HybridRetriever(
            index=Stm32Index(collection=SOURCE_COLLECTIONS.get(names[0], names[0])), cfg=cfg
        )

    from .rag.router import MultiSourceRetriever

    return MultiSourceRetriever(sources=names, cfg=cfg)


def cmd_ask(args) -> int:
    from .rag.answer import answer

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
    from .rag.evaluate import check_anchors, load_cases, run_eval

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


def cmd_eval_graph(args) -> int:
    """跨语料 A/B：看「期望来源」有没有被图谱扩展拉进 top-k。"""
    from .rag.evaluate import load_function_cases, run_cross_source_eval
    from .rag.router import MultiSourceRetriever

    cases = load_function_cases(args.cases) if args.cases else load_function_cases()

    def make_retriever(cfg):
        return MultiSourceRetriever(cfg=cfg)

    base = run_cross_source_eval(make_retriever, cases, graph_expand=False, k=args.k)
    graph = run_cross_source_eval(make_retriever, cases, graph_expand=True, k=args.k)
    print(
        f"跨语料命中（期望来源进 top-{args.k}）：{len(cases)} 题 | "
        f"基线 {base['rate']:.1%} → 图扩展 {graph['rate']:.1%} | 平均新增块 {graph['avg_added']}"
    )
    for before, after in zip(base["rows"], graph["rows"]):
        mark = "OK" if after["hit"] else "~ "
        print(
            f"{mark} +{after['added']} | {before['q'][:36]:38s} "
            f"基线 {','.join(before['sources'])} → 图 {','.join(after['sources'])}"
        )
    if args.json:
        print(json.dumps({"base": base, "graph": graph}, ensure_ascii=False, indent=2))
    return 0


def cmd_agent_eval(args) -> int:
    """Agent 模式评测：成功率 / 自修复轮数 / 回退比例 / 耗时 / token。"""
    from .assistant.eval_agent import format_report, run_agent_eval

    only = [item.strip() for item in (args.only or "").split(",") if item.strip()]
    report = run_agent_eval(
        cases=args.cases,
        limit=args.limit,
        only=only or None,
        build=not args.no_build,
        max_steps=args.max_steps,
        keep_previous=args.keep,
        case_timeout=args.case_timeout,
    )
    print("\n" + format_report(report))
    if args.out:
        Path(args.out).expanduser().write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n明细已写入：{args.out}")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------- 开发助手

def cmd_scaffold(args) -> int:
    from .gen.scaffold import scaffold_project

    modules = tuple(item.strip() for item in args.modules.split(",") if item.strip())
    result = scaffold_project(args.dest, device=args.device, name=args.name, modules=modules)
    print(f"工程：{result['project']}")
    print(f"器件：{result['device']}")
    print(f"文件：{len(result['files'])} 个")
    return 0


def cmd_codegen(args) -> int:
    from .build.pipeline import parse_requirement
    from .gen.codegen import plan_application, spec_summary

    spec, notes, points = parse_requirement(args.requirement, device=args.device)
    plan = plan_application(spec)
    if args.json:
        print(json.dumps(
            {
                "spec": spec_summary(spec),
                "ok": plan.ok,
                "errors": plan.errors,
                "warnings": plan.warnings,
                "checks": plan.checks,
                "wiring": plan.wiring,
                "modules": plan.modules,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0 if plan.ok else 1
    print(f"需求：{args.requirement}")
    print(f"器件：{spec.device.name}    SYSCLK {spec.sysclk / 1e6:g} MHz")
    for point in points:
        print(f"  · {point}")
    if notes:
        print("\n未覆盖/提醒：")
        for item in notes:
            print(f"  - {item}")
    if not plan.ok:
        print("\n校验失败：")
        for item in plan.errors:
            print(f"  - {item}")
        return 1
    print("\n计算核对：")
    for item in plan.checks:
        print(f"  · {item}")
    if plan.warnings:
        print("\n提醒：")
        for item in plan.warnings:
            print(f"  - {item}")
    print("\n接线：")
    for item in plan.wiring:
        print(f"  · {item}")
    print("\n" + "─" * 78)
    print(plan.main_c)
    return 0


def cmd_develop(args) -> int:
    from .build.pipeline import develop

    delivery = develop(
        args.requirement,
        workspace=args.workspace,
        project=args.project,
        device=args.device,
        build=not args.no_build,
        max_fix=args.max_fix,
        clean=args.clean,
        sysclk=args.sysclk,
    )
    print(delivery.summary())
    return 0 if delivery.ok else 2


def cmd_agent(args) -> int:
    from .assistant.agent import DevAssistant

    assistant = DevAssistant(
        workspace=args.workspace,
        project=args.project,
        max_steps=args.max_steps,
        auto_build=not args.no_build,
    )
    result = assistant.run(args.requirement)
    print(result.summary())
    return 0 if result.ok else 2


def cmd_build_project(args) -> int:
    from .build.keil import KeilBuilder
    from .gen.project import find_project

    if args.project:
        project = Path(args.project)
    elif args.workspace:
        project = find_project(args.workspace)
    else:
        raise SystemExit("请用 --project 指定 .uvprojx，或用 --workspace 指定工程目录")
    if project is None:
        raise SystemExit(f"{args.workspace} 下没有找到 .uvprojx")
    result = KeilBuilder().build(project, clean=args.clean, log_path=args.log)
    print(f"工程：{project}")
    print(result.summary())
    if not result.ok and args.show_log and result.log_path:
        print("\n--- 日志末尾 ---")
        print("\n".join(result.log_text.splitlines()[-60:]))
    return 0 if result.ok else 2


def cmd_project(args) -> int:
    from .gen.project import KeilProject, find_project

    if args.project:
        project = Path(args.project)
    elif args.workspace:
        project = find_project(args.workspace)
    else:
        raise SystemExit("请用 --project 指定 .uvprojx，或用 --workspace 指定工程目录")
    if project is None:
        raise SystemExit(f"{args.workspace} 下没有找到 .uvprojx")
    obj = KeilProject(project)
    print(obj.describe())
    root = obj.dir.parent
    from .gen.cubemx import find_ioc, parse_ioc

    ioc = find_ioc(root)
    if ioc is not None:
        print()
        print(parse_ioc(ioc).describe())
    if args.json:
        print(json.dumps(obj.summary(), ensure_ascii=False, indent=2))
    return 0


def cmd_mcp(args) -> int:
    from . import mcp_keil_server

    argv = ["--transport", args.transport, "--port", str(args.port)]
    if args.workspace:
        argv.extend(["--workspace", args.workspace])
    return mcp_keil_server.main(argv)


def cmd_graph(args) -> int:
    """器件知识图谱（Neo4j）：建图 / 结构化查询 / 与 mcu.py 互验。"""
    from .graphs.graph import GraphStore

    try:
        store = GraphStore()
    except RuntimeError as exc:
        print(f"图谱不可用：{exc}")
        return 2

    try:
        if not store.ping():
            print(f"连不上 Neo4j（{store.uri}）：先启动容器，见 neo4j/README.md")
            return 2

        if args.build:
            result = store.build(reset=args.reset)
            print("建图完成")
            print(result.summary())
            issues = store.verify()
            print("校验：图与 mcu.py 一致 ✓" if not issues else "校验不一致：\n" + "\n".join(f"- {item}" for item in issues))
            return 0 if not issues else 1

        if args.reset:
            store.reset()
            print("已清空图谱节点")
            return 0

        if args.stats:
            print(store.describe("stats"))
            return 0

        if args.verify:
            print(store.describe("verify"))
            return 0

        if args.cypher:
            print(store.describe("cypher", args.cypher))
            return 0

        if args.pin:
            print(store.describe("pin", args.pin))
        elif args.timer:
            print(store.describe("timer", args.timer))
        elif args.usart:
            print(store.describe("usart", args.usart))
        elif args.device:
            print(store.describe("device", args.device))
        elif args.pwm_pins:
            print(store.describe("pwm_pins"))
        else:
            print(store.describe("stats"))
        return 0
    finally:
        store.close()


def cmd_graph_docs(args) -> int:
    """文档知识图谱（GraphRAG）：建图 / 符号查询 / 邻域扩展。"""
    from .graphs.graph_docs import DocGraphStore

    try:
        store = DocGraphStore()
    except RuntimeError as exc:
        print(f"图谱不可用：{exc}")
        return 2

    try:
        if not store.ping():
            print(f"连不上 Neo4j（{store.uri}）：先启动容器，见 neo4j/README.md")
            return 2

        if args.build:
            result = store.build(reset=args.reset)
            print("建图完成")
            print(result.summary())
            return 0

        if args.reset:
            store.reset()
            print("已清空文档图节点")
            return 0

        if args.stats:
            print(store.describe("stats"))
        elif args.symbol:
            print(store.describe("symbol", args.symbol))
        elif args.chunk:
            print(store.describe("chunk", args.chunk))
        elif args.expand:
            print(store.describe("expand", args.expand))
        elif args.calls:
            print(store.describe("calls", args.calls))
        elif args.callers:
            print(store.describe("callers", args.callers))
        elif args.peripheral:
            print(store.describe("peripheral", args.peripheral))
        elif args.cypher:
            print(store.describe("cypher", args.cypher))
        else:
            print(store.describe("stats"))
        return 0
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 GBK，输出里的符号会直接报 UnicodeEncodeError
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(prog="stm32_rag", description="STM32F10xxx 参考手册 RAG")
    parser.add_argument("--collection", default=None, help="Chroma 集合名（默认按来源取）")
    sub = parser.add_subparsers(dest="command", required=True)
    sources = ["rm", "ds", "um", "hal", "examples"]

    p_inspect = sub.add_parser("inspect", help="体检语料并预览切块")
    p_inspect.add_argument("--source", choices=sources, default="rm", help="语料来源")
    p_inspect.add_argument("--pages", help="印刷页码范围，如 160-200")
    p_inspect.add_argument("--no-figures", action="store_true")
    p_inspect.add_argument("--show", type=int, default=0, help="打印前 N 个块")
    p_inspect.set_defaults(func=cmd_inspect)

    p_build = sub.add_parser("build", help="建库")
    p_build.add_argument("--source", choices=sources, default="rm", help="语料来源")
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
    p_stats.add_argument("--source", default="rm", help="内置来源或 docs:<slug>")
    p_stats.set_defaults(func=cmd_stats)

    p_ingest = sub.add_parser("ingest", help="PDF -> MinerU -> Markdown -> 建库（全自动，可多份）")
    p_ingest.add_argument("pdf", nargs="*", help="待处理的 PDF，可多个：a.pdf b.pdf")
    p_ingest.add_argument("--md", help="已有 Markdown，跳过 MinerU")
    p_ingest.add_argument("--tier", default="basic", choices=["flash", "basic", "standard", "advanced"])
    p_ingest.add_argument("--pages", help="只解析指定页，如 1-10")
    p_ingest.add_argument("--force", action="store_true", help="已有 Markdown 也重新解析")
    p_ingest.add_argument("--rebuild", action="store_true", help="重建该文档的集合")
    p_ingest.add_argument("--workers", type=int, default=8)
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="提问")
    p_ask.add_argument("question")
    p_ask.add_argument("--sources", default="auto", help="auto 或逗号列表：rm,ds,um,hal,examples")
    p_ask.add_argument("-k", type=int, default=None, help="最终片段数")
    p_ask.add_argument("--vector-k", type=int, default=None)
    p_ask.add_argument("--no-rerank", action="store_true")
    p_ask.add_argument("--no-expand", action="store_true")
    p_ask.add_argument("--with-figures", action="store_true", help="命中插图时把原图随消息发给视觉模型")
    p_ask.add_argument("--graph-expand", action="store_true", help="开启文档图谱跨语料扩展（需先 graph-docs --build）")
    p_ask.add_argument("--show-context", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="锚点评测")
    p_eval.add_argument("--cases", help="评测集路径")
    p_eval.add_argument("--sources", default="auto", help="auto 或逗号列表")
    p_eval.add_argument("-k", type=int, default=None)
    p_eval.add_argument("--vector-k", type=int, default=None)
    p_eval.add_argument("--no-rerank", action="store_true")
    p_eval.add_argument("--no-expand", action="store_true")
    p_eval.add_argument("--check", action="store_true", help="先校验锚点是否在语料中")
    p_eval.add_argument("--graph-expand", action="store_true", help="开启文档图谱跨语料扩展（A/B 用）")
    p_eval.add_argument("--json", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_eg = sub.add_parser("eval-graph", help="跨语料 A/B：图谱扩展能不能把期望来源拉进 top-k")
    p_eg.add_argument("--cases", default=None, help="函数类评测集（默认 eval_cases_functions.json）")
    p_eg.add_argument("-k", type=int, default=6)
    p_eg.add_argument("--json", action="store_true")
    p_eg.set_defaults(func=cmd_eval_graph)

    p_ae = sub.add_parser("agent-eval", help="Agent 模式评测：成功率 / 自修复 / 回退 / 耗时 / token")
    p_ae.add_argument("--cases", default=None, help="用例集（默认 eval_cases_agent.json）")
    p_ae.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    p_ae.add_argument("--only", default="", help="只跑指定 id，逗号分隔，如 a03,a21")
    p_ae.add_argument("--max-steps", type=int, default=40, help="单个用例的 Agent 递归上限")
    p_ae.add_argument("--case-timeout", type=float, default=300.0, help="单条用例超时秒数")
    p_ae.add_argument("--no-build", action="store_true", help="只生成不编译（很快，但成功判定不可信）")
    p_ae.add_argument("--keep", action="store_true", help="保留上一轮评测目录")
    p_ae.add_argument("--out", default=None, help="把明细 JSON 写到文件")
    p_ae.add_argument("--json", action="store_true", help="同时打印 JSON")
    p_ae.set_defaults(func=cmd_agent_eval)

    # ---------------- 开发助手 ----------------
    p_scaffold = sub.add_parser("scaffold", help="生成可编译的 Keil 工程骨架")
    p_scaffold.add_argument("--dest", required=True, help="生成目录")
    p_scaffold.add_argument("--device", default=None, help="器件型号，默认 STM32F103RB")
    p_scaffold.add_argument("--name", default="stm32_app", help="工程名")
    p_scaffold.add_argument("--modules", default="tim,uart", help="需要启用的 HAL 模块，逗号分隔")
    p_scaffold.set_defaults(func=cmd_scaffold)

    p_codegen = sub.add_parser("codegen", help="只解析需求并打印生成的代码（不落盘）")
    p_codegen.add_argument("requirement")
    p_codegen.add_argument("--device", default=None)
    p_codegen.add_argument("--json", action="store_true")
    p_codegen.set_defaults(func=cmd_codegen)

    p_develop = sub.add_parser("develop", help="确定性开发流水线（生成 + 编译 + 自修复 + 报告）")
    p_develop.add_argument("requirement")
    p_develop.add_argument("--workspace", default=None, help="交付目录（默认 workspaces/<需求>）")
    p_develop.add_argument("--project", default=None, help="复用已有 .uvprojx")
    p_develop.add_argument("--device", default=None)
    p_develop.add_argument("--sysclk", type=int, default=72_000_000)
    p_develop.add_argument("--max-fix", type=int, default=3, help="自动修复-重编次数上限")
    p_develop.add_argument("--clean", action="store_true", help="先全量重编")
    p_develop.add_argument("--no-build", action="store_true")
    p_develop.set_defaults(func=cmd_develop)

    p_agent = sub.add_parser("agent", help="LLM 编排的开发助手（失败自动回退到 develop）")
    p_agent.add_argument("requirement")
    p_agent.add_argument("--workspace", default=None)
    p_agent.add_argument("--project", default=None)
    p_agent.add_argument("--max-steps", type=int, default=40, help="LangGraph 递归步数上限")
    p_agent.add_argument("--no-build", action="store_true")
    p_agent.set_defaults(func=cmd_agent)

    p_build = sub.add_parser("compile", aliases=["keil"], help="用 UV4 编译已有工程")
    p_build.add_argument("--workspace", default=None, help="在哪个目录下找 .uvprojx")
    p_build.add_argument("--project", default=None, help="直接指定 .uvprojx")
    p_build.add_argument("--clean", action="store_true", help="先删中间产物")
    p_build.add_argument("--log", default=None, help="日志路径")
    p_build.add_argument("--show-log", action="store_true", help="失败时打印日志末尾")
    p_build.set_defaults(func=cmd_build_project)

    p_project = sub.add_parser("project", help="查看 Keil 工程结构")
    p_project.add_argument("--workspace", default=None)
    p_project.add_argument("--project", default=None)
    p_project.add_argument("--json", action="store_true")
    p_project.set_defaults(func=cmd_project)

    p_mcp = sub.add_parser("mcp", help="启动 Keil MCP Server")
    p_mcp.add_argument("--workspace", default=None)
    p_mcp.add_argument("--transport", default="stdio", choices=["stdio", "sse"])
    p_mcp.add_argument("--port", type=int, default=8765)
    p_mcp.set_defaults(func=cmd_mcp)

    p_graph = sub.add_parser("graph", help="器件知识图谱（Neo4j）：建图 / 查询 / 校验")
    p_graph.add_argument("--build", action="store_true", help="建图（幂等，可重复执行）")
    p_graph.add_argument("--reset", action="store_true", help="先清空图谱节点（可配 --build）")
    p_graph.add_argument("--stats", action="store_true", help="看节点/关系数量")
    p_graph.add_argument("--verify", action="store_true", help="与 mcu.py 静态表互验")
    p_graph.add_argument("--pin", default=None, help="查引脚能力，如 PA8")
    p_graph.add_argument("--timer", default=None, help="查定时器，如 TIM3")
    p_graph.add_argument("--usart", default=None, help="查串口，如 USART1")
    p_graph.add_argument("--device", default=None, help="查器件，如 STM32F103C8")
    p_graph.add_argument("--pwm-pins", action="store_true", help="列出所有能出 PWM 的引脚")
    p_graph.add_argument("--cypher", default=None, help="只读 Cypher 查询")
    p_graph.set_defaults(func=cmd_graph)

    p_gdocs = sub.add_parser("graph-docs", help="文档知识图谱（GraphRAG）：建图 / 符号 / 邻域扩展")
    p_gdocs.add_argument("--build", action="store_true", help="从 Chroma metadata 建图（幂等）")
    p_gdocs.add_argument("--reset", action="store_true", help="先清空文档图节点（可配 --build）")
    p_gdocs.add_argument("--stats", action="store_true", help="看节点/关系数量")
    p_gdocs.add_argument("--symbol", default=None, help="查符号出现在哪些块，如 HAL_GPIO_Init")
    p_gdocs.add_argument("--chunk", default=None, help="查块详情 + 相邻块（传 Chroma chunk id）")
    p_gdocs.add_argument("--expand", default=None, help="给一个 chunk id，看能扩出哪些跨语料邻块")
    p_gdocs.add_argument("--calls", default=None, help="调用链（2 跳），如 HAL_GPIO_Init")
    p_gdocs.add_argument("--callers", default=None, help="反向：谁调用了它")
    p_gdocs.add_argument("--peripheral", default=None, help="查外设：哪些块提到它 + 器件图信息，如 TIM1")
    p_gdocs.add_argument("--cypher", default=None, help="只读 Cypher 查询")
    p_gdocs.set_defaults(func=cmd_graph_docs)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
