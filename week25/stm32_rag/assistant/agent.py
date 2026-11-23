"""Agent / Orchestrator：需求分析 → 任务规划 → 工具选择 → 执行控制 → 交付。

对应架构图的中间层。和 `pipeline.py` 的分工：

- `pipeline.py` 是确定性的“手”，会在没有大模型时独立跑完整条链路；
- 本模块是“脑”，用 LangChain `create_agent` 把自然语言需求拆成工具调用序列，
  并且**始终通过工具**去检索、生成、编译，避免模型直接手写 C 代码。

模型只决定“做什么”，工具保证“做得对”：引脚复用、分频、编译都由代码判断。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..core.config import WORKSPACE_ROOT, get_llm
from ..build.keil import BuildResult
from .tools import WorkspaceSession


class _TokenCounter:
    """统计一次 agent 运行的 token（LangChain 回调，兼容 OpenAI/百炼两种返回）。"""

    def __init__(self) -> None:
        from langchain_core.callbacks import BaseCallbackHandler

        counter = self

        class _Handler(BaseCallbackHandler):
            def on_llm_end(self, response, **kwargs):  # noqa: ANN001
                usage = {}
                output = getattr(response, "llm_output", None) or {}
                usage = output.get("token_usage") or output.get("usage") or {}
                if not usage:
                    for batch in getattr(response, "generations", []) or []:
                        for gen in batch:
                            meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
                            if meta:
                                usage = meta
                                break
                        if usage:
                            break
                if usage:
                    counter.calls += 1
                    counter.input += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                    counter.output += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)

        self._handler = _Handler()
        self.calls = 0
        self.input = 0
        self.output = 0

    @property
    def handler(self):
        return self._handler

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "input": self.input,
            "output": self.output,
            "total": self.input + self.output,
        }

SYSTEM_PROMPT = """你是 STM32 AI 开发助手，目标是把中文需求变成**可编译**的 STM32F103 工程。

固定工作流（按顺序执行，不要跳步）：

1. 需求分析：识别外设（GPIO / TIM PWM / 定时器中断 / USART / 按键 / 蜂鸣器 / ADC / OLED）、
   引脚、频率、占空比、波特率、是否需要 DMA、主循环动作。
2. 知识核对：至少调用一次 stm32_knowledge_search，确认 HAL 函数名、引脚复用、
   时钟约束。检索不到就说明“知识库未命中”，不要编造。
   涉及**引脚能不能复用、DMA 用哪个通道、外设挂在哪条总线、中断向量名**时，
   优先用 stm32_graph_query（器件图谱，确定事实）而不是查文档，
   例如 stm32_graph_query("pin", "PA8")、stm32_graph_query("usart", "USART1")。
   想知道**某个函数/宏到底在哪讲、它调用了谁、哪个章节讲这个外设**时，
   用 stm32_doc_graph，例如 stm32_doc_graph("calls", "HAL_UART_Init")、
   stm32_doc_graph("peripheral", "TIM1")。
3. 规格化：把需求写成 generate_application 的 JSON 规格，**不要自己手写 main.c**。
   需求里出现下面这些词，**必须**写对应字段，否则等于没实现：
   | 需求关键词 | 规格字段 |
   | --- | --- |
   | 按键 / 消抖 | `"keys": {"pins": ["PB12","PB13"], "debounce_ms": 20}` |
   | 蜂鸣器 / 音调 | `"buzzer": {"pin": "PB0"}`（自动占用一个 PWM 通道） |
   | ADC / 电位器 / 旋钮 | `"adc": {"pin": "PA0", "samples": 8}` |
   | OLED / 显示屏 | `"oled": {"instance": "I2C1"}`（自动配 PB6/PB7） |
   | 串口命令行 / 接收命令 | `"uart": [...]` + `"serial_cmd": {"instance": "USART1"}` |
   | 1 ms 调度 / 任务框架 | `"sched": true` |
   | 多个引脚同频 PWM | `"pwm"` 数组写多条（同一定时器共享频率） |
4. 生成：调用 generate_application。它会返回“计算核对”（PSC/ARR/CCR、BRR 误差、
   引脚复用）。若报“规格校验未通过”，按提示修正规格后重试。
5. 编译：调用 build_project。失败时先 auto_fix_build 补齐模块/宏/路径；
   仍失败再 read_source 找到出错行、用 write_source 修改后重编译。
6. 交付：用中文写交付说明：编译状态、关键参数、接线、验证方法、下一步建议。

硬性规则：
- 只能通过工具改动工作区文件；不要臆造未经验证的 HAL 函数与寄存器。
- **检索最多 2 次**；同一个问题不要反复查，查到关键参数就进入生成。
- `generate_application` 报校验错误时，按提示改规格重试，最多 3 次；不要绕开校验手写 main.c。
- `build_project` 通过后**立即结束**：不要再 read_source / 重复检索 / 重复编译，直接用中文给出交付说明。
- 编译失败时最多 `auto_fix_build` 2 轮，仍失败就说明原因并结束。
- 生成器目前支持 GPIO、TIM PWM（同一定时器多通道，共频）、定时器更新中断、
  USART（发送轮询/DMA，接收中断 + 命令行）、ADC+DMA 滑动平均、I2C SSD1306 OLED，
  以及按键/蜂鸣器/1 ms 调度组件。SPI / CAN / 编码器 / 输入捕获未支持，
  需求里出现这些时明确说明，不要假装生成。
- 同一定时器的多个通道必须同频（共享 PSC/ARR），不一致时先告诉用户改用两个定时器。
- 频率/波特率必须来自需求；需求没给的用默认值（1 kHz、占空比 50%、115200）
  并在交付说明里点明。
- 最终答复不超过 400 字，包含：编译状态、关键参数、接线、下一步。"""

MAX_REPORT_CHARS = 1600


@dataclass
class AgentResult:
    ok: bool
    requirement: str
    answer: str = ""
    workspace: Path | None = None
    project: Path | None = None
    build: BuildResult | None = None
    actions: list[str] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)
    report_path: Path | None = None
    error: str = ""
    spec: object | None = None
    fallback: bool = False
    fix_rounds: int = 0
    elapsed_s: float = 0.0
    tokens: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"需求：{self.requirement}"]
        if self.error:
            lines.append(f"错误：{self.error}")
        if self.build is not None:
            lines.append(self.build.summary())
        if self.answer:
            lines.append("--- 交付说明 ---")
            lines.append(self.answer)
        if self.project:
            lines.append(f"工程：{self.project}")
        if self.report_path:
            lines.append(f"报告：{self.report_path}")
        return "\n".join(lines)


class DevAssistant:
    """编排器。"""

    def __init__(
        self,
        workspace: str | Path | None = None,
        project: str | Path | None = None,
        llm=None,
        max_steps: int = 40,
        auto_build: bool = True,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve() if workspace else None
        self.project = Path(project).expanduser().resolve() if project else None
        self._llm = llm
        self.max_steps = max_steps
        self.auto_build = auto_build

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    # ------------------------------------------------------------ 主流程
    def run(self, requirement: str) -> AgentResult:
        started = time.monotonic()
        workspace = self.workspace or _default_workspace(requirement)
        workspace.mkdir(parents=True, exist_ok=True)
        session = WorkspaceSession(workspace=workspace, project_path=self.project)

        try:
            agent = _create_agent(self.llm, session)
        except Exception as exc:  # 模型配置缺失：退回确定性流水线
            return self._fallback(
                requirement, workspace, f"无法创建 Agent：{type(exc).__name__}: {exc}", started
            )

        counter = _TokenCounter()
        try:
            result = agent.invoke(
                {"messages": [("user", _user_prompt(requirement, workspace))]},
                config={"recursion_limit": self.max_steps, "callbacks": [counter.handler]},
            )
        except Exception as exc:
            return self._fallback(
                requirement, workspace, f"Agent 执行失败：{type(exc).__name__}: {exc}", started
            )

        messages = _messages_of(result)
        answer, transcript = _extract(messages)

        if session.build is None and self.auto_build:
            session.build_report()

        report = _write_agent_report(
            requirement=requirement,
            workspace=workspace,
            session=session,
            answer=answer,
            transcript=transcript,
        )
        ok = bool(session.build and session.build.ok)
        fix_rounds = sum(1 for line in transcript if line.startswith("← auto_fix_build"))
        return AgentResult(
            ok=ok,
            requirement=requirement,
            answer=answer,
            workspace=workspace,
            project=session.project.path if session.project else None,
            build=session.build,
            actions=session.actions,
            transcript=transcript,
            report_path=report,
            spec=session.spec,
            fix_rounds=fix_rounds,
            elapsed_s=round(time.monotonic() - started, 1),
            tokens=counter.summary(),
        )

    def _fallback(self, requirement: str, workspace: Path, reason: str, started: float = 0.0) -> AgentResult:
        from ..build.pipeline import develop

        delivery = develop(requirement, workspace=workspace, project=self.project)
        return AgentResult(
            ok=delivery.ok,
            requirement=requirement,
            answer=f"（Agent 不可用，已用确定性流水线完成）{reason}",
            workspace=workspace,
            project=delivery.project,
            build=delivery.build,
            actions=delivery.steps,
            report_path=delivery.report_path,
            error=reason,
            spec=delivery.spec,
            fallback=True,
            elapsed_s=round(time.monotonic() - started, 1) if started else 0.0,
        )


# ---------------------------------------------------------------- 内部

def _create_agent(llm, session: WorkspaceSession):
    from langchain.agents import create_agent

    return create_agent(model=llm, tools=session.tools(), system_prompt=SYSTEM_PROMPT)


def _user_prompt(requirement: str, workspace: Path) -> str:
    return (
        f"需求：{requirement}\n"
        f"工作区：{workspace}\n"
        "请按固定工作流完成：先检索知识库核对参数，再生成代码，然后编译并按需修复，最后给出交付说明。"
    )


def _default_workspace(requirement: str) -> Path:
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", requirement).strip("_")[:32] or "task"
    return WORKSPACE_ROOT / slug


def _messages_of(result) -> list:
    if isinstance(result, dict):
        return list(result.get("messages") or [])
    return list(getattr(result, "messages", []) or [])


def _extract(messages: list) -> tuple[str, list[str]]:
    """从消息序列里取出最终答复与工具调用轨迹。"""
    answer = ""
    transcript: list[str] = []
    for message in messages:
        kind = getattr(message, "type", "")
        content = getattr(message, "content", "")
        if kind == "ai":
            calls = getattr(message, "tool_calls", None) or []
            for call in calls:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
                args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
                transcript.append(f"→ 调用 {name}({_short(args)})")
            text = _content_text(content)
            if text.strip():
                answer = text.strip()
        elif kind == "tool":
            name = getattr(message, "name", "tool")
            text = _content_text(content)
            transcript.append(f"← {name}：{_short(text, 300)}")
    return answer, transcript


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content or "")


def _short(value, limit: int = 220) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _write_agent_report(
    requirement: str,
    workspace: Path,
    session: WorkspaceSession,
    answer: str,
    transcript: list[str],
) -> Path:
    path = workspace / "AGENT.md"
    lines = [
        f"# {requirement}",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## 交付说明",
        "",
        answer or "（模型没有给出文字答复）",
        "",
    ]
    if session.spec is not None:
        from ..gen.codegen import spec_summary

        import json

        lines.extend(["## 规格", "", "```json", json.dumps(spec_summary(session.spec), ensure_ascii=False, indent=2), "```", ""])
    if session.build is not None:
        lines.extend(["## 编译", "", session.build.summary(), ""])
    if session.actions:
        lines.extend(["## 动作记录", ""] + [f"- {item}" for item in session.actions] + [""])
    if transcript:
        lines.extend(["## 工具轨迹", "", "```text"] + [item[: max(MAX_REPORT_CHARS, 300)] for item in transcript] + ["```", ""])
    lines.extend(
        [
            "## 文件",
            "",
            "```text",
            _tree(workspace),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _tree(root: Path, limit: int = 60) -> str:
    lines: list[str] = []
    for item in sorted(root.rglob("*")):
        if item.is_dir():
            continue
        if any(part in {"Objects", "Listings", "__pycache__"} for part in item.parts):
            continue
        rel = item.relative_to(root)
        lines.append(f"{rel}  ({item.stat().st_size} B)")
        if len(lines) >= limit:
            lines.append("…")
            break
    return "\n".join(lines)
