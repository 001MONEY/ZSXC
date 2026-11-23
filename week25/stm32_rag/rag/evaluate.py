"""锚点式评测：看证据有没有进 top-k，而不是看模型说得像不像。

每题给若干「锚点短语」（寄存器名、地址、关键值）。检索结果里出现全部锚点算命中，
同时记录 MRR —— 第一个命中片段排在第几位。这样同一套题可以横向比向量、混合、重排。
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

from ..core.config import PACKAGE_DIR, RetrieveConfig
from .retrieve import HybridRetriever

CASES_PATH = PACKAGE_DIR / "eval_cases.json"
FUNCTION_CASES_PATH = PACKAGE_DIR / "eval_cases_functions.json"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").lower()


def load_cases(path: str | Path = CASES_PATH) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    for case in cases:
        if not case.get("q") or not case.get("anchors"):
            raise ValueError(f"题目缺少 q/anchors：{case}")
    return cases


def check_anchors(cases: list[dict], documents: list[str]) -> list[tuple[str, str]]:
    """检查锚点是否真的在语料里，防止题目写错。返回缺失的 (问题, 锚点)。"""
    corpus = _normalize("\n".join(documents))
    missing: list[tuple[str, str]] = []
    for case in cases:
        for anchor in case["anchors"]:
            if _normalize(anchor) not in corpus:
                missing.append((case["q"], anchor))
    return missing


def run_eval(
    retriever: HybridRetriever,
    cases: list[dict],
    k: int | None = None,
    verbose: bool = True,
) -> dict:
    # 评测必须和问答看同一份证据：不在这里二次截断，-k 只改检索配置
    cfg = replace(retriever.cfg, final_k=k) if k else retriever.cfg
    rows: list[dict] = []
    for case in cases:
        hits = retriever.search(case["q"], cfg=cfg)
        anchors = [_normalize(a) for a in case["anchors"]]

        joined = _normalize("\n".join(h.text for h in hits))
        hit = all(a in joined for a in anchors)

        partial = 0.0
        reciprocal_rank = 0.0
        for rank, h in enumerate(hits, start=1):
            text = _normalize(h.text)
            covered = sum(1 for a in anchors if a in text)
            partial = max(partial, covered / len(anchors))
            if reciprocal_rank == 0.0 and any(a in text for a in anchors):
                reciprocal_rank = 1.0 / rank

        rows.append(
            {
                "q": case["q"],
                "hit": hit,
                "coverage": round(partial, 2),
                "mrr": round(reciprocal_rank, 3),
                "top": hits[0].label if hits else "",
            }
        )
        if verbose:
            mark = "OK" if hit else ("~ " if partial > 0 else "X ")
            print(f"{mark} 覆盖{partial:.2f} RR{reciprocal_rank:.2f} | {case['q']}")
            if not hit and hits:
                print(f"    top1: {hits[0].label}")

    n = len(rows)
    summary = {
        "cases": n,
        "hit_rate": round(sum(1 for r in rows if r["hit"]) / n, 3) if n else 0.0,
        "avg_coverage": round(sum(r["coverage"] for r in rows) / n, 3) if n else 0.0,
        "mrr": round(sum(r["mrr"] for r in rows) / n, 3) if n else 0.0,
        "rows": rows,
    }
    if verbose:
        print(
            f"\n汇总：{summary['cases']} 题｜全中 {summary['hit_rate']:.1%}｜"
            f"平均覆盖 {summary['avg_coverage']:.1%}｜MRR {summary['mrr']:.3f}"
        )
    return summary


# ---------------------------------------------------------------- 跨语料 A/B

def load_function_cases(path: str | Path | None = None) -> list[dict]:
    """函数类评测题：`expect` 是期望出现在 top-k 里的来源（如 um）。"""
    data = json.loads(Path(path or FUNCTION_CASES_PATH).read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    for case in cases:
        if not case.get("q") or not case.get("expect"):
            raise ValueError(f"题目缺少 q/expect：{case}")
    return cases


def run_cross_source_eval(make_retriever, cases: list[dict], graph_expand: bool, k: int = 6) -> dict:
    """看「期望来源」有没有进 top-k；用来量化图谱扩展的收益。

    关闭重排：这里测的是召回，不是模型打分。
    """
    cfg = RetrieveConfig(graph_expand=graph_expand, use_rerank=False, final_k=k)
    retriever = make_retriever(cfg)
    rows: list[dict] = []
    for case in cases:
        hits = retriever.search(case["q"], cfg=cfg)
        sources = [str(h.metadata.get("source_key") or "") for h in hits]
        expected = {str(item) for item in case["expect"]}
        rows.append(
            {
                "q": case["q"],
                "hit": bool(expected & set(sources)),
                "sources": sources,
                "added": sum(1 for h in hits if h.metadata.get("graph_expanded")),
            }
        )
    n = len(rows)
    return {
        "cases": n,
        "rate": round(sum(1 for r in rows if r["hit"]) / n, 3) if n else 0.0,
        "avg_added": round(sum(r["added"] for r in rows) / n, 2) if n else 0.0,
        "rows": rows,
    }
