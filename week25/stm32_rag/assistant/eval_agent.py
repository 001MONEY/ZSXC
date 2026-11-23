"""Agent 模式评测：同一批自由说法需求，分别跑确定性 `develop` 与 LLM `agent`。

统计口径：
- **匹配（match）**：工程编译通过，且 main.c / 组件文件符合用例里的 `expect`
  （引脚、PWM 定时器与通道数、频率、占空比、串口实例/波特率、组件文件）；
- **一次成功率**：match 且 `fix_rounds == 0`、未回退 develop；
- **自修复轮数**：agent 轨迹里调用 `auto_fix_build` 的次数；
- **回退比例**：Agent 创建/执行抛异常时 `DevAssistant` 自动退回 `develop`；
- **耗时与 token**：整次运行墙钟时间，token 来自 LangChain 回调。

用法：
    python -m stm32_rag agent-eval                  # 全量 26 条
    python -m stm32_rag agent-eval --limit 3         # 先试 3 条
    python -m stm32_rag agent-eval --only a03,a21    # 指定用例
    python -m stm32_rag agent-eval --no-build        # 只生成不编译（快，但 match 无意义）
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core.config import PACKAGE_DIR, WORKSPACE_ROOT

DEFAULT_CASES = PACKAGE_DIR / "eval_cases_agent.json"
COMPONENT_FILES = {
    "keys": "key.c",
    "buzzer": "buzzer.c",
    "adc": "adc_knob.c",
    "oled": "oled_ssd1306.c",
    "serial_cmd": "serial_cmd.c",
    "sched": "app_sched.c",
}


@dataclass
class CaseResult:
    id: str
    requirement: str
    # develop 基线
    d_ok: bool = False
    d_match: bool = False
    d_elapsed: float = 0.0
    d_missing: list[str] = field(default_factory=list)
    # agent
    a_ok: bool = False
    a_match: bool = False
    a_elapsed: float = 0.0
    a_missing: list[str] = field(default_factory=list)
    fix_rounds: int = 0
    fallback: bool = False
    tokens: dict = field(default_factory=dict)
    error: str = ""
    answer: str = ""


def load_cases(path: str | Path | None = None) -> list[dict]:
    file = Path(path) if path else DEFAULT_CASES
    return json.loads(Path(file).read_text(encoding="utf-8"))


def _tim_block(main: str, timer: str) -> str:
    match = re.search(rf"static void MX_{timer}_Init\(void\)\n\{{(.*?)\n\}}", main, re.S)
    return match.group(1) if match else ""


def _timing(block: str) -> tuple[int, int] | None:
    psc = re.search(r"Init\.Prescaler = (\d+);", block)
    arr = re.search(r"Init\.Period = (\d+);", block)
    if not psc or not arr:
        return None
    return int(psc.group(1)), int(arr.group(1))


def _duty(block: str) -> float | None:
    timing = _timing(block)
    pulse = re.search(r"sConfigOC\.Pulse = (\d+);", block)
    if timing is None or pulse is None:
        return None
    return int(pulse.group(1)) / (timing[1] + 1)


def match_workspace(workspace: Path, expect: dict) -> tuple[bool, list[str]]:
    """按用例 expect 检查生成结果；返回 (是否匹配, 缺失项)。"""
    workspace = Path(workspace)
    main_path = workspace / "Core" / "Src" / "main.c"
    main = main_path.read_text(encoding="utf-8", errors="replace") if main_path.is_file() else ""
    missing: list[str] = []

    for pin in expect.get("pins", []):
        match = re.fullmatch(r"P([A-E])(\d{1,2})", pin.upper())
        if match is None:
            continue
        port, number = match.group(1), int(match.group(2))
        if f"GPIO_PIN_{number}" not in main:
            missing.append(f"引脚 {pin} 未出现")

    for key in expect.get("components", []):
        name = COMPONENT_FILES.get(key)
        if name and not (workspace / "Core" / "Src" / name).is_file():
            missing.append(f"缺少组件 {key}")

    for group in expect.get("pwm_list", []):
        block = _tim_block(main, group["timer"])
        if not block:
            missing.append(f"{group['timer']} 未初始化")
            continue
        channels = len(re.findall(r"ConfigChannel\([^;]*TIM_CHANNEL_\d", block))
        if channels < int(group.get("channels", 1)):
            missing.append(f"{group['timer']} 只有 {channels} 个通道")
        target = group.get("freq_hz")
        timing = _timing(block)
        if target and timing:
            freq = 72_000_000 / ((timing[0] + 1) * (timing[1] + 1))
            if abs(freq - target) / target > 0.02:
                missing.append(f"{group['timer']} 频率 {freq:.0f} Hz ≠ {target}")

    uart = expect.get("uart")
    if uart:
        instance = uart["instance"]
        if f"MX_{instance}_UART_Init" not in main:
            missing.append(f"{instance} 未初始化")
        baud = uart.get("baud")
        if baud and f"BaudRate = {baud};" not in main:
            missing.append(f"{instance} 波特率不是 {baud}")

    if "duty" in expect and expect["duty"] is not None:
        blocks = re.findall(r"static void MX_TIM\d+_Init\(void\)\n\{(.*?)\n\}", main, re.S)
        ratios = [value for value in (_duty(block) for block in blocks) if value is not None]
        if not ratios or abs(ratios[0] - float(expect["duty"])) > 0.02:
            missing.append(f"占空比不符（{ratios[0] if ratios else '无'}）")

    return (not missing), missing


def _refused(text: str, keys: list[str]) -> bool:
    return all(key in text for key in keys) and ("未支持" in text or "不支持" in text)


def run_case(case: dict, root: Path, build: bool, max_steps: int) -> CaseResult:
    from ..build.pipeline import develop
    from .agent import DevAssistant

    expect = case.get("expect") or {}
    result = CaseResult(id=case["id"], requirement=case["requirement"])

    d_ws = root / "develop" / case["id"]
    started = time.monotonic()
    delivery = develop(case["requirement"], workspace=d_ws, build=build, max_fix=0)
    result.d_elapsed = round(time.monotonic() - started, 1)
    d_match, d_missing = match_workspace(d_ws, expect)
    if expect.get("refuse"):
        notes = " ".join(list(delivery.problems or []) + list(delivery.steps or []))
        d_match = _refused(notes, expect["refuse"])
        d_missing = [] if d_match else [f"未提示不支持 {expect['refuse']}"]
    result.d_ok = bool(delivery.ok)
    result.d_match = bool(delivery.ok and d_match)
    result.d_missing = d_missing

    a_ws = root / "agent" / case["id"]
    assistant = DevAssistant(workspace=a_ws, max_steps=max_steps, auto_build=build)
    started = time.monotonic()
    agent_result = assistant.run(case["requirement"])
    result.a_elapsed = agent_result.elapsed_s or round(time.monotonic() - started, 1)
    if expect.get("refuse"):
        a_match = _refused(agent_result.answer, expect["refuse"])
        a_missing = [] if a_match else [f"未提示不支持 {expect['refuse']}"]
    else:
        a_match, a_missing = match_workspace(a_ws, expect)
    result.a_ok = bool(agent_result.ok)
    result.a_match = bool(agent_result.ok and a_match)
    result.a_missing = a_missing
    result.fix_rounds = agent_result.fix_rounds
    result.fallback = agent_result.fallback
    result.tokens = agent_result.tokens or {}
    result.error = agent_result.error
    result.answer = agent_result.answer[:160]
    return result


def _summary(results: list[CaseResult]) -> dict:
    total = len(results)
    agent_used = [item for item in results if not item.fallback]
    return {
        "cases": total,
        "develop_ok": sum(1 for item in results if item.d_ok),
        "develop_match": sum(1 for item in results if item.d_match),
        "develop_avg_s": round(sum(item.d_elapsed for item in results) / total, 1) if total else 0,
        "agent_ok": sum(1 for item in results if item.a_ok),
        "agent_match": sum(1 for item in results if item.a_match),
        "agent_first_pass": sum(
            1 for item in results if item.a_match and item.fix_rounds == 0 and not item.fallback
        ),
        "fallback": sum(1 for item in results if item.fallback),
        "fix_total": sum(item.fix_rounds for item in agent_used),
        "fix_avg": round(sum(item.fix_rounds for item in agent_used) / len(agent_used), 2) if agent_used else 0,
        "agent_avg_s": round(sum(item.a_elapsed for item in results) / total, 1) if total else 0,
        "tokens_total": sum(int(item.tokens.get("total") or 0) for item in results),
        "tokens_avg": round(sum(int(item.tokens.get("total") or 0) for item in results) / total) if total else 0,
        "calls_avg": round(sum(int(item.tokens.get("calls") or 0) for item in results) / total, 1) if total else 0,
    }


def _with_timeout(case: dict, root: Path, build: bool, max_steps: int, timeout: float) -> CaseResult:
    """用守护线程跑单条用例：网络挂起时超时放行，不拖死整批。"""
    box: dict = {}

    def target() -> None:
        try:
            box["result"] = run_case(case, root, build, max_steps)
        except Exception as exc:  # noqa: BLE001
            box["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return CaseResult(id=case["id"], requirement=case["requirement"], error=f"超时（>{timeout:.0f}s）")
    if "result" in box:
        return box["result"]
    return CaseResult(id=case["id"], requirement=case["requirement"], error=box.get("error", "未知错误"))


def run_agent_eval(
    cases: str | Path | list[dict] | None = None,
    limit: int | None = None,
    only: list[str] | None = None,
    build: bool = True,
    max_steps: int = 40,
    root: str | Path | None = None,
    keep_previous: bool = False,
    case_timeout: float = 300.0,
    verbose: bool = True,
) -> dict:
    items = cases if isinstance(cases, list) else load_cases(cases)
    if only:
        wanted = {item.strip() for item in only if item.strip()}
        items = [case for case in items if case["id"] in wanted]
    if limit:
        items = items[:limit]

    root = Path(root) if root else WORKSPACE_ROOT / "agent_eval"
    if root.exists() and not keep_previous:
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    results: list[CaseResult] = []
    for index, case in enumerate(items, start=1):
        if verbose:
            print(f"[{index}/{len(items)}] {case['id']} {case['requirement']}", flush=True)
        try:
            item = _with_timeout(case, root, build, max_steps, case_timeout)
        except Exception as exc:  # 单条失败不能中断整批评测
            item = CaseResult(id=case["id"], requirement=case["requirement"], error=f"{type(exc).__name__}: {exc}")
        results.append(item)
        if verbose:
            mark = "OK " if item.a_match else "MISS"
            print(
                f"    develop={'match' if item.d_match else 'miss'} "
                f"| agent={mark} fix={item.fix_rounds} fb={item.fallback} "
                f"{item.a_elapsed:.0f}s tok={item.tokens.get('total', 0)}",
                flush=True,
            )

    return {
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "summary": _summary(results),
        "cases": [asdict(item) for item in results],
    }


def format_report(report: dict) -> str:
    """把评测结果排成 Markdown 表格。"""
    s = report["summary"]
    lines = [
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 用例数 | {s['cases']} |",
        f"| develop 编译通过 | {s['develop_ok']}/{s['cases']} |",
        f"| develop 需求匹配 | {s['develop_match']}/{s['cases']} |",
        f"| agent 编译通过 | {s['agent_ok']}/{s['cases']} |",
        f"| agent 需求匹配 | {s['agent_match']}/{s['cases']} |",
        f"| agent 一次成功（无修复、无回退） | {s['agent_first_pass']}/{s['cases']} |",
        f"| 回退 develop | {s['fallback']}/{s['cases']} |",
        f"| 自修复总轮数 / 平均 | {s['fix_total']} / {s['fix_avg']} |",
        f"| 平均耗时（agent） | {s['agent_avg_s']} s |",
        f"| token 总量 / 平均 | {s['tokens_total']} / {s['tokens_avg']} |",
        f"| 平均 LLM 调用次数 | {s['calls_avg']} |",
        "",
        "| 用例 | develop | agent | 修复轮 | 回退 | 耗时 | token |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["cases"]:
        lines.append(
            f"| {item['id']} | {'✓' if item['d_match'] else '✗'} | {'✓' if item['a_match'] else '✗'}"
            f" | {item['fix_rounds']} | {'是' if item['fallback'] else ''}"
            f" | {item['a_elapsed']:.0f}s | {item['tokens'].get('total', 0)} |"
        )
    return "\n".join(lines)
