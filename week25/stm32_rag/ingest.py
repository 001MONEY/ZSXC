"""PDF -> MinerU -> Markdown -> 清洗/切块 -> 向量库，一条命令跑完。

用法：
    python -m stm32_rag ingest 手册.pdf              # 自动起 MinerU 服务、解析、建库
    python -m stm32_rag ingest 手册.pdf --md x.md    # 已有 Markdown，跳过 MinerU
    python -m stm32_rag ingest 手册.pdf --rebuild    # 重建该文档集合

产物：
    mineru_out/<stem>.md                 # MinerU 的 Markdown（可复用）
    chroma_db/stm32_doc_<slug>_...       # 文档专属集合
    chroma_db/custom_sources.json        # 注册表：answers 路由据此把新文档纳入检索
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import time
from pathlib import Path

from .config import (
    CUSTOM_SOURCES_FILE,
    MINERU_EXE,
    MINERU_MD_DIR,
    MINERU_TIER,
    get_embeddings,
)
from .index import Stm32Index
from .loader_md import markdown_documents, summarize

_NOT_RUNNING_MARKERS = ("未在运行", "not running", "stopped", "no running")
_JOB_RE = re.compile(r"(?:ID|id)[:：]\s*(\d+)")


def _run(exe: Path, args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(exe), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def server_status(exe: Path = MINERU_EXE) -> str:
    result = _run(exe, ["server", "status"], timeout=60)
    return (result.stdout or "") + (result.stderr or "")


def ensure_server(exe: Path = MINERU_EXE, wait_seconds: int = 240) -> None:
    """MinerU 的 parse 依赖常驻服务；没起就拉起并等到健康。"""
    if not Path(exe).is_file():
        raise FileNotFoundError(
            f"找不到 MinerU 可执行文件：{exe}（可用环境变量 MINERU_EXE 指定，安装方法见 MINERU.md）"
        )
    text = server_status(exe)
    if not any(marker in text.lower() for marker in _NOT_RUNNING_MARKERS):
        return

    print("MinerU 服务未运行，正在启动（首次需加载模型，约 1~3 分钟）…")
    subprocess.Popen(
        [str(exe), "server", "start"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        time.sleep(5)
        text = server_status(exe)
        if not any(marker in text.lower() for marker in _NOT_RUNNING_MARKERS):
            print("MinerU 服务已就绪。")
            return
    raise TimeoutError(f"MinerU 服务在 {wait_seconds}s 内未就绪：{text.strip()[:200]}")


def mineru_parse(
    pdf: Path,
    out_md: Path,
    tier: str = MINERU_TIER,
    wait: int = 900,
    pages: str | None = None,
) -> Path:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    print(f"MinerU 解析中：{pdf.name}（tier={tier}{', pages=' + pages if pages else ''}）…")
    started = time.time()
    args = ["parse", str(pdf), "-o", str(out_md), "--tier", tier, "--wait", str(wait)]
    if pages:
        args.extend(["--pages", pages])
    result = _run(MINERU_EXE, args, timeout=wait + 300)
    if result.returncode != 0 or not out_md.is_file():
        tail = ((result.stdout or "") + (result.stderr or "")).strip()[-600:]
        raise RuntimeError(f"MinerU 解析失败（exit={result.returncode}）：{tail}")
    print(f"解析完成：{out_md}（{time.time() - started:.0f}s）")
    return out_md


def slugify(name: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()
    return slug or "doc"


def load_registry() -> dict:
    if not CUSTOM_SOURCES_FILE.is_file():
        return {}
    try:
        return json.loads(CUSTOM_SOURCES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def register_source(name: str, collection: str, doc: str, pdf: str, md: str) -> None:
    registry = load_registry()
    registry[name] = {
        "collection": collection,
        "doc": doc,
        "pdf": pdf,
        "md": md,
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    CUSTOM_SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_SOURCES_FILE.write_text(json.dumps(registry, ensure_ascii=False, indent=1), encoding="utf-8")


def ingest(
    pdf: str | Path | None = None,
    md: str | Path | None = None,
    collection: str | None = None,
    tier: str = MINERU_TIER,
    rebuild: bool = False,
    workers: int = 8,
    pages: str | None = None,
    force: bool = False,
) -> dict:
    """PDF（或现成 Markdown）-> 切块 -> 建库 -> 注册到自定义来源。"""
    pdf_path = Path(pdf) if pdf else None
    md_path = Path(md) if md else None

    if md_path is None:
        if pdf_path is None:
            raise ValueError("至少要给 pdf 或 md 其一。")
        if not pdf_path.is_file():
            raise FileNotFoundError(f"找不到 PDF：{pdf_path}")
        ensure_server()
        md_path = MINERU_MD_DIR / f"{pdf_path.stem}.md"
        if force or not md_path.is_file():
            job = mineru_enqueue(pdf_path, md_path, tier=tier, pages=pages)
            print(f"MinerU 入队：{pdf_path.name}（任务 {job}），等待解析…")
            started = time.time()
            if job:
                wait_for_job(job)
                # --no-wait 只入队不落盘；任务 done 后再跑一次 --wait 命中缓存，秒回并写出 Markdown
                mineru_parse(pdf_path, md_path, tier=tier, pages=pages)
            else:
                mineru_parse(pdf_path, md_path, tier=tier, pages=pages)
            print(f"解析完成：{md_path}（{time.time() - started:.0f}s）")
        else:
            print(f"复用已有 Markdown：{md_path}")
    elif not md_path.is_file():
        raise FileNotFoundError(f"找不到 Markdown：{md_path}")

    return _build_from_markdown(pdf_path, md_path, collection, rebuild, workers)


def _build_from_markdown(
    pdf_path: Path | None,
    md_path: Path,
    collection: str | None = None,
    rebuild: bool = False,
    workers: int = 8,
) -> dict:
    doc_name = pdf_path.stem if pdf_path else md_path.stem
    slug = slugify(doc_name)
    collection = collection or f"stm32_doc_{slug}_text_embedding_v4"
    source_name = f"docs:{slug}"

    documents = markdown_documents(md_path, doc=doc_name)
    if not documents:
        raise RuntimeError(f"{md_path.name} 没有切出任何块。")

    index = Stm32Index(collection=collection, embeddings=get_embeddings())
    result = index.add_documents(documents, rebuild=rebuild, workers=workers)
    register_source(
        source_name,
        collection,
        doc_name,
        str(pdf_path) if pdf_path else "",
        str(md_path),
    )
    result.update(
        {
            "source": source_name,
            "collection": collection,
            "doc": doc_name,
            "markdown": str(md_path),
            "summary": summarize(documents),
        }
    )
    return result


def mineru_enqueue(
    pdf: Path,
    out_md: Path,
    tier: str = MINERU_TIER,
    pages: str | None = None,
) -> str | None:
    """把解析任务入队（--no-wait），立即返回任务 ID；服务端后台继续跑。"""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    args = ["parse", str(pdf), "-o", str(out_md), "--tier", tier, "--no-wait"]
    if pages:
        args.extend(["--pages", pages])
    result = _run(MINERU_EXE, args, timeout=180)
    if result.returncode != 0:
        tail = ((result.stdout or "") + (result.stderr or "")).strip()[-400:]
        raise RuntimeError(f"MinerU 入队失败（exit={result.returncode}）：{tail}")
    text = (result.stdout or "") + (result.stderr or "")
    match = _JOB_RE.search(text)
    return match.group(1) if match else None


def wait_for_markdown(out_md: Path, timeout: int = 1800, poll: int = 5) -> Path:
    """轮询等待 Markdown 落盘（用作物化后的 sanity check）。"""
    started = time.time()
    while time.time() - started < timeout:
        if out_md.is_file() and out_md.stat().st_size > 0:
            return out_md
        time.sleep(poll)
    raise TimeoutError(f"MinerU 未在 {timeout}s 内产出 {out_md}")


def mineru_job_state(job_id: str) -> str:
    result = _run(MINERU_EXE, ["show", "parse", str(job_id)], timeout=60)
    return (result.stdout or "") + (result.stderr or "")


def wait_for_job(job_id: str, timeout: int = 1800, poll: int = 5) -> None:
    """轮询解析任务，直到 done（服务端单机队列，任务是串行跑的）。"""
    started = time.time()
    last = ""
    while time.time() - started < timeout:
        last = mineru_job_state(job_id).lower()
        if "done" in last or "success" in last:
            return
        if "fail" in last or "error" in last or "cancel" in last:
            raise RuntimeError(f"MinerU 任务 {job_id} 失败：{last.strip()[:300]}")
        time.sleep(poll)
    raise TimeoutError(f"MinerU 任务 {job_id} 在 {timeout}s 内未完成")


def ingest_batch(
    pdfs: list[str | Path],
    tier: str = MINERU_TIER,
    pages: str | None = None,
    rebuild: bool = False,
    workers: int = 8,
    force: bool = False,
) -> list[dict]:
    """多份 PDF：全部先入队，MinerU 后台逐本解析；完成一本就切块入库。"""
    ensure_server()
    jobs: list[tuple[Path, Path, str | None]] = []
    for raw in pdfs:
        pdf = Path(raw)
        if not pdf.is_file():
            raise FileNotFoundError(f"找不到 PDF：{pdf}")
        md_path = MINERU_MD_DIR / f"{pdf.stem}.md"
        if md_path.is_file() and not force:
            print(f"复用已有 Markdown：{pdf.name}")
            jobs.append((pdf, md_path, None))
            continue
        job = mineru_enqueue(pdf, md_path, tier=tier, pages=pages)
        print(f"入队：{pdf.name}（任务 {job}）")
        jobs.append((pdf, md_path, job))

    results: list[dict] = []
    for index, (pdf, md_path, job) in enumerate(jobs, start=1):
        if job is None and md_path.is_file():
            print(f"[{index}/{len(jobs)}] 跳过解析：{pdf.name}")
        else:
            print(f"[{index}/{len(jobs)}] 等待 {pdf.name} 解析…")
            started = time.time()
            if job:
                wait_for_job(job)
            mineru_parse(pdf, md_path, tier=tier, pages=pages)
            wait_for_markdown(md_path)
            print(f"  解析完成（{time.time() - started:.0f}s），开始切块入库…")
        results.append(_build_from_markdown(pdf, md_path, rebuild=rebuild, workers=workers))
    return results


async def aingest_batch(**kwargs) -> list[dict]:
    """异步封装：内部仍走线程，但不会阻塞调用方的事件循环。"""
    return await asyncio.to_thread(ingest_batch, **kwargs)
