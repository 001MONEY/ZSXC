"""开发流水线：需求 → 规划 → 代码 → 编译 → 自修复 → 交付。

这条流水线是**确定性**的（不依赖大模型），好处是：

- 同样的需求每次得到同样的工程，编译结果可复现；
- LLM（`agent.py`）负责理解需求和查资料，真正落盘与编译都走这里，
  模型写错的时候不会把工程写坏；
- 每一步都留痕（`CHANGES.md` / `REPORT.md`），交付物里能看到改了什么、为什么。

自修复只处理“工程配置类”错误（缺模块、缺宏、缺头文件、缺 system 文件）——
这些是自动生成工程最容易犯的错；真正的语法/逻辑错误仍然要靠人或 Agent 改代码。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import DEFAULT_DEVICE, DEFAULT_HSE, DEFAULT_SYSCLK, WORKSPACE_ROOT
from .codegen import (
    AppSpec,
    CodePlan,
    GpioSpec,
    LoopSpec,
    PwmSpec,
    TimerSpec,
    UartSpec,
    plan_application,
    spec_summary,
)
from .mcu import USART_PINS, Pin, resolve_device, timer_for_pin
from .project import DEFAULT_GROUP, KeilProject, find_project
from .scaffold import HAL_MODULE_SOURCES, enable_hal_conf, ensure_hal_modules, scaffold_project
from .keil import BuildResult, KeilBuilder, QuickFix, suggest_fixes

# ---------------------------------------------------------------- 需求解析

_DEVICE_RE = re.compile(r"STM32F10\d[A-Z]{1,2}\d[A-Z0-9]{0,3}", re.IGNORECASE)
_FREQ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(MHz|兆赫|kHz|千赫|Hz|赫兹|赫)", re.IGNORECASE)
_DUTY_RE = re.compile(r"(?:占空比|duty)\s*(?:[:：=]|为|是)?\s*(\d+(?:\.\d+)?)\s*%?", re.IGNORECASE)
_BAUD_RE = re.compile(r"(\d{3,6})\s*(?:bps|波特)?", re.IGNORECASE)
_INTERVAL_RE = re.compile(r"每\s*(\d+(?:\.\d+)?)\s*(ms|毫秒|s|秒|us|微秒)", re.IGNORECASE)
_HELLO_RE = re.compile(r"(?:打印|发送|输出)(?:字符串)?\s*[\"“]([^\"”]+)[\"”]")

_PWM_RE = re.compile(r"PWM|脉宽|占空比|pwm", re.IGNORECASE)
_UART_RE = re.compile(r"USART\s*([1-5])|UART\s*([1-5])|串口|uart|usart", re.IGNORECASE)
_TIMER_RE = re.compile(r"TIM\s*([1-8])|定时器|timer", re.IGNORECASE)
_IRQ_RE = re.compile(r"中断|IRQ|interrupt", re.IGNORECASE)
_GPIO_RE = re.compile(r"输出|输入|翻转|闪|点亮|LED|按键|GPIO", re.IGNORECASE)
_DMA_RE = re.compile(r"\bDMA\b|直接存储|直接内存", re.IGNORECASE)
_PRINTF_RE = re.compile(r"printf|打印|Hello|发送字符串", re.IGNORECASE)

_PIN_RE = re.compile(r"\bP([A-E])(\d{1,2})\b", re.IGNORECASE)


def _freq_to_hz(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit in ("mhz", "兆赫"):
        return value * 1e6
    if unit in ("khz", "千赫"):
        return value * 1e3
    return value


def _fmt_hz(value: float) -> str:
    if value >= 1e6:
        return f"{value / 1e6:g} MHz"
    if value >= 1e3:
        return f"{value / 1e3:g} kHz"
    return f"{value:g} Hz"


def _interval_to_ms(value: float, unit: str) -> int:
    unit = unit.lower()
    if unit in ("s", "秒"):
        return int(value * 1000)
    if unit in ("us", "微秒"):
        return max(1, int(value / 1000))
    return int(value)


def parse_requirement(
    text: str,
    device: str | None = None,
    sysclk: int = DEFAULT_SYSCLK,
    hse: int = DEFAULT_HSE,
) -> tuple[AppSpec, list[str], list[str]]:
    """把自然语言需求拆成结构化规格。

    返回 (规格, 未支持的补充说明, 解析出的要点)。规则解析不追求覆盖所有说法，
    拿不准的地方宁可在 `notes` 里说清楚，也不猜着生成。
    """
    chip_match = _DEVICE_RE.search(text)
    dev = resolve_device(device or (chip_match.group(0) if chip_match else DEFAULT_DEVICE))
    notes: list[str] = []
    points: list[str] = []

    # 频率：取第一个带单位的数值；波特率单独识别
    freq_hz = 1000.0
    freq_match = _FREQ_RE.search(text)
    if freq_match:
        freq_hz = _freq_to_hz(float(freq_match.group(1)), freq_match.group(2))
        points.append(f"目标频率 {_fmt_hz(freq_hz)}")

    duty = 0.5
    duty_match = _DUTY_RE.search(text)
    if duty_match:
        duty = max(0.0, min(1.0, float(duty_match.group(1)) / 100))
        points.append(f"占空比 {duty * 100:g}%")

    pins = [Pin(m.group(1).upper(), int(m.group(2))) for m in _PIN_RE.finditer(text)]
    pins = list(dict.fromkeys(pins))
    if pins:
        points.append("引脚 " + "、".join(pin.name for pin in pins))

    uart_instance = ""
    uart_match = _UART_RE.search(text)
    if uart_match:
        number = uart_match.group(1) or uart_match.group(2)
        uart_instance = f"USART{number}" if number else "USART1"

    timer_instance = ""
    timer_match = _TIMER_RE.search(text)
    if timer_match:
        timer_instance = f"TIM{timer_match.group(1)}" if timer_match.group(1) else ""

    is_pwm = bool(_PWM_RE.search(text))
    is_irq = bool(_IRQ_RE.search(text))
    has_dma = bool(_DMA_RE.search(text))
    wants_print = bool(_PRINTF_RE.search(text))

    spec = AppSpec(device=dev, sysclk=sysclk, hse=hse, title=_slugify(text)[:40] or "app")

    # ---- 串口
    if uart_instance:
        if uart_instance not in USART_PINS:
            notes.append(f"{uart_instance} 不在支持列表里，按 USART1 处理")
            uart_instance = "USART1"
        baud = 115200
        baud_match = _BAUD_RE.search(text)
        if baud_match:
            candidate = int(baud_match.group(1))
            if 1200 <= candidate <= 4_000_000:
                baud = candidate
                points.append(f"波特率 {baud}")
        spec.uarts.append(
            UartSpec(
                instance=uart_instance,
                baud=baud,
                dma=has_dma,
                printf=True,
            )
        )
        used = {USART_PINS[uart_instance]["TX"][0], USART_PINS[uart_instance]["RX"][0]}
        pins = [pin for pin in pins if pin not in used]

    # ---- PWM / 定时中断
    if is_pwm or (freq_match and pins and not is_irq):
        target_pin = pins[0] if pins else Pin("A", 8)
        if timer_instance:
            from .mcu import channel_of_pin

            channel = channel_of_pin(target_pin, timer_instance)
            if channel is None:
                # 用户点名了定时器但引脚不支持：保留原样，让 validate() 报明确错误
                spec.pwm.append(PwmSpec(pin=target_pin, timer=timer_instance, channel="CH1", freq_hz=freq_hz, duty=duty))
            else:
                spec.pwm.append(PwmSpec(pin=target_pin, timer=timer_instance, channel=channel, freq_hz=freq_hz, duty=duty))
                points.append(f"{target_pin.name} → {timer_instance}_{channel}")
        else:
            options = timer_for_pin(target_pin, dev)
            if options is None:
                spec.pwm.append(PwmSpec(pin=target_pin, timer="TIM1", channel="CH1", freq_hz=freq_hz, duty=duty))
                notes.append(f"{target_pin.name} 在这颗封装里没有定时器复用")
            else:
                timer, channel = options
                spec.pwm.append(PwmSpec(pin=target_pin, timer=timer, channel=channel, freq_hz=freq_hz, duty=duty))
                points.append(f"{target_pin.name} 自动匹配 {timer}_{channel}")
        pins = [pin for pin in pins if pin not in [spec.pwm[-1].pin]]

    # ---- 定时器更新中断
    if is_irq and (timer_match or not is_pwm):
        timer = timer_instance or "TIM2"
        toggle = pins[0] if pins else None
        spec.timers.append(TimerSpec(timer=timer, freq_hz=freq_hz if freq_match else 1000.0, toggle_pin=toggle))
        if toggle:
            pins = [pin for pin in pins if pin != toggle]
        points.append(f"{timer} 更新中断 {spec.timers[-1].freq_hz / 1e3:g} kHz")

    # ---- GPIO
    for pin in pins:
        if any(item.pin == pin for item in spec.pwm):
            continue
        mode = "output_pp"
        if re.search(r"输入|按键", text):
            mode = "input_pullup"
        initial = "" if mode.startswith(("input", "analog", "it_")) else "low"
        spec.gpios.append(GpioSpec(pin=pin, mode=mode, initial=initial, label="用户配置"))
        points.append(f"{pin.name} 配置为 {mode}")

    # ---- 主循环
    interval = 500
    interval_match = _INTERVAL_RE.search(text)
    if interval_match:
        interval = _interval_to_ms(float(interval_match.group(1)), interval_match.group(2))
        points.append(f"周期 {interval} ms")
    elif "每秒" in text or "每秒钟" in text:
        interval = 1000
        points.append("周期 1000 ms（每秒）")
    elif "每毫秒" in text:
        interval = 1

    toggle_pin = next((g.pin for g in spec.gpios if "翻转" in text or "闪" in text or "LED" in text.upper()), None)
    if toggle_pin:
        spec.loop = LoopSpec(kind="toggle", pin=toggle_pin, interval_ms=interval)
    elif wants_print and spec.uarts:
        message = (_HELLO_RE.search(text).group(1) if _HELLO_RE.search(text) else "Hello from STM32")
        spec.loop = LoopSpec(kind="print", interval_ms=max(interval, 10), message=message)
        points.append(f"主循环每 {spec.loop.interval_ms} ms 打印 {message!r}")

    # ---- 不支持的诉求，明确说出来
    for keyword, hint in (
        ("ADC", "ADC 采样"),
        ("I2C", "I2C 通信"),
        ("SPI", "SPI 通信"),
        ("CAN", "CAN 通信"),
        ("USB", "USB 通信"),
        ("编码器", "编码器接口"),
        ("输入捕获", "输入捕获"),
    ):
        if re.search(rf"\b{keyword}\b", text, re.IGNORECASE):
            notes.append(f"当前生成器还不支持{hint}（{keyword}），可以用工程分析/编译工具手工接入")

    return spec, notes, points


# ---------------------------------------------------------------- 修复

def apply_quick_fix(workspace: Path, project: KeilProject, fix: QuickFix) -> str:
    """执行一条修复动作，返回人类可读的结果说明。"""
    workspace = Path(workspace)
    if fix.action == "add_define":
        added = project.add_define(fix.value)
        return f"加宏 {fix.value}" if added else f"宏 {fix.value} 已存在"
    if fix.action == "add_include":
        added = project.add_include_path(fix.value)
        return f"加头文件路径 {fix.value}" if added else f"头文件路径 {fix.value} 已存在"
    if fix.action == "add_file":
        rel = fix.value.replace("..\\", "").replace("\\", "/")
        if project.add_file(rel, group=DEFAULT_GROUP):
            return f"加源文件 {rel}"
        return f"源文件 {rel} 已在工程里"
    if fix.action == "add_module":
        module = fix.value
        created = ensure_hal_modules(workspace, [module])
        enable_hal_conf(workspace / "Core" / "Inc" / "stm32f1xx_hal_conf.h", [module])
        added = []
        for source in HAL_MODULE_SOURCES.get(module, ()):
            if project.add_file(f"Drivers/STM32F1xx_HAL_Driver/Src/{source}", group="Drivers/STM32F1xx_HAL_Driver"):
                added.append(source)
        detail = "、".join(added) or "（文件已在工程里）"
        return f"补 {module.upper()} 模块：{detail}" + ("（新拷贝）" if created else "")
    return f"需要人工处理：{fix.reason}"


def _sync_project_config(workspace: Path, project: KeilProject, modules: list[str]) -> list[str]:
    """把生成代码需要的模块/宏/头文件路径同步进工程。"""
    actions: list[str] = []
    base_modules = {"gpio", "rcc", "cortex", "hal", "dma", "flash"}
    need = sorted(set(modules) | base_modules)
    created = ensure_hal_modules(workspace, need)
    conf_enabled = enable_hal_conf(workspace / "Core" / "Inc" / "stm32f1xx_hal_conf.h", need)
    if created:
        actions.append(f"拷贝 HAL 源文件 {len(created)} 个：{'、'.join(created)}")
    if conf_enabled:
        actions.append(f"打开 HAL 模块开关：{'、'.join(conf_enabled)}")

    hal_added = []
    for source in ("stm32f1xx_hal.c", "stm32f1xx_hal_cortex.c", "stm32f1xx_hal_dma.c",
                   "stm32f1xx_hal_flash.c", "stm32f1xx_hal_flash_ex.c", "stm32f1xx_hal_gpio.c",
                   "stm32f1xx_hal_rcc.c", "stm32f1xx_hal_rcc_ex.c"):
        if project.add_file(f"Drivers/STM32F1xx_HAL_Driver/Src/{source}", group="Drivers/STM32F1xx_HAL_Driver"):
            hal_added.append(source)
    for module in need:
        for source in HAL_MODULE_SOURCES.get(module, ()):
            if project.add_file(f"Drivers/STM32F1xx_HAL_Driver/Src/{source}", group="Drivers/STM32F1xx_HAL_Driver"):
                hal_added.append(source)
    if hal_added:
        actions.append(f"工程新增 HAL 源文件 {len(hal_added)} 个")

    defines = project.add_define("USE_HAL_DRIVER", _device_define(project))
    if defines:
        actions.append(f"工程新增宏：{'、'.join(defines)}")

    includes = project.add_include_path(
        r"..\Core\Inc",
        r"..\Drivers\CMSIS\Include",
        r"..\Drivers\CMSIS\Device\ST\STM32F1xx\Include",
        r"..\Drivers\STM32F1xx_HAL_Driver\Inc",
    )
    if includes:
        actions.append(f"工程新增头文件路径：{'、'.join(includes)}")
    return actions


def _device_define(project: KeilProject) -> str:
    return resolve_device(project.device).define


# ---------------------------------------------------------------- 交付

@dataclass
class Delivery:
    ok: bool
    requirement: str
    workspace: Path
    project: Path | None = None
    spec: AppSpec | None = None
    plan: CodePlan | None = None
    build: BuildResult | None = None
    steps: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    report_path: Path | None = None
    changelog_path: Path | None = None
    error: str = ""

    def summary(self) -> str:
        lines = [f"需求：{self.requirement}"]
        if self.error:
            lines.append(f"失败：{self.error}")
        if self.problems:
            lines.append("需要人工确认：" + "；".join(self.problems))
        if self.build is not None:
            lines.append(self.build.summary())
        if self.project:
            lines.append(f"工程：{self.project}")
        if self.report_path:
            lines.append(f"报告：{self.report_path}")
        if self.changelog_path:
            lines.append(f"修改记录：{self.changelog_path}")
        return "\n".join(lines)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", text).strip("_")
    return slug[:32] or "app"


def develop(
    requirement: str,
    workspace: str | Path | None = None,
    project: str | Path | None = None,
    device: str | None = None,
    sysclk: int = DEFAULT_SYSCLK,
    hse: int = DEFAULT_HSE,
    build: bool = True,
    max_fix: int = 3,
    clean: bool = False,
    title: str | None = None,
) -> Delivery:
    """跑完整条流水线。"""
    # CubeMX 工程：没显式指定器件时，用 .ioc 里的真实型号
    if device is None and workspace is not None:
        from .cubemx import find_ioc, parse_ioc

        probe = find_ioc(workspace)
        if probe is not None:
            info = parse_ioc(probe)
            device = info.mcu or None

    spec, problems, points = parse_requirement(requirement, device=device, sysclk=sysclk, hse=hse)
    spec.title = title or spec.title

    if workspace is None:
        workspace = WORKSPACE_ROOT / _slugify(title or requirement)
    workspace = Path(workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    steps: list[str] = [f"解析需求：{'；'.join(points) or '无明确外设'}"]
    plan = plan_application(spec)
    if not plan.ok:
        delivery = Delivery(
            ok=False,
            requirement=requirement,
            workspace=workspace,
            spec=spec,
            plan=plan,
            steps=steps,
            problems=problems,
            error="代码生成校验未通过：" + "；".join(plan.errors),
        )
        delivery.report_path = _write_report(delivery, points)
        return delivery

    # ---- 工程：有就用，没有就生成
    existing = Path(project).expanduser() if project else find_project(workspace)
    if existing and Path(existing).is_file():
        project_obj = KeilProject(existing)
        steps.append(f"复用现有工程：{project_obj.path}")
    else:
        name = _project_name(spec)
        scaffold = scaffold_project(
            workspace,
            device=spec.device,
            name=name,
            modules=tuple(sorted(set(plan.modules) | {"tim", "uart"})),
        )
        project_obj = KeilProject(scaffold["project"])
        steps.append(f"生成 Keil 工程：{project_obj.path}（器件 {spec.device.name}）")

    # ---- 写代码（CubeMX 工程走增量插入，不覆盖生成代码）
    from .cubemx import apply_to_project, find_ioc

    main_c = workspace / "Core" / "Src" / "main.c"
    cubemx_mode = bool(
        find_ioc(workspace)
        and existing
        and main_c.is_file()
        and "USER CODE BEGIN" in main_c.read_text(encoding="utf-8", errors="replace")
    )

    changed: list[str] = []
    if cubemx_mode:
        patch = apply_to_project(workspace, spec, project_obj)
        steps.extend(patch.applied)
        steps.extend(f"跳过：{item}" for item in patch.skipped)
        steps.extend(f"提醒：{item}" for item in patch.warnings)
        problems.extend(patch.errors)
        if not patch.ok:
            delivery = Delivery(
                ok=False,
                requirement=requirement,
                workspace=workspace,
                project=project_obj.path,
                spec=spec,
                plan=plan,
                steps=steps,
                problems=problems,
                error="CubeMX 工程增量修改失败：" + "；".join(patch.errors),
            )
            delivery.report_path = _write_report(delivery, points)
            return delivery
        changed = patch.changed_files
        spec = patch.spec or spec
        plan = CodePlan(
            ok=True,
            warnings=plan.warnings,
            wiring=plan.wiring,
            checks=list(dict.fromkeys(plan.checks + patch.checks)),
            modules=patch.modules or plan.modules,
            files={},
            main_c="",
        )
        steps.append(f"修改文件：{', '.join(changed) or '无（需求已满足）'}")
    else:
        for rel, content in plan.files.items():
            path = workspace / rel
            before = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
            if before != content.strip():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                changed.append(rel)
        steps.append(f"写入源码：{', '.join(changed) if changed else '内容无变化'}")

    # ---- 同步工程配置
    sync_actions = _sync_project_config(workspace, project_obj, plan.modules)
    for rel in ("Core/Src/main.c", "Core/Src/stm32f1xx_it.c", "Core/Src/system_stm32f1xx.c"):
        if project_obj.add_file(rel, group=DEFAULT_GROUP if "main" in rel or "it" in rel else "CMSIS"):
            sync_actions.append(f"工程新增源码 {rel}")
    startup = workspace / "Core" / "Startup" / spec.device.startup
    if startup.is_file() and project_obj.add_file(f"Core/Startup/{startup.name}", group="Startup"):
        sync_actions.append(f"工程新增启动文件 {startup.name}")
    project_obj.save()
    steps.extend(sync_actions or ["工程配置已是最新"])

    delivery = Delivery(
        ok=True,
        requirement=requirement,
        workspace=workspace,
        project=project_obj.path,
        spec=spec,
        plan=plan,
        steps=steps,
        problems=problems,
    )

    # ---- 编译 + 自修复
    if build:
        builder = KeilBuilder()
        fixes_applied: list[str] = []
        for attempt in range(max_fix + 1):
            result = builder.build(project_obj.path, clean=clean and attempt == 0)
            delivery.build = result
            steps.append(f"第 {attempt + 1} 次编译：{'通过' if result.ok else '失败'}")
            if result.ok or attempt == max_fix:
                break
            fixes = suggest_fixes(result.errors)
            if not fixes:
                steps.append("没有可自动应用的工程配置修复，停在这里交给人工/Agent")
                break
            for fix in fixes:
                if fix.action == "manual":
                    steps.append(f"跳过人工项：{fix.reason}")
                    continue
                detail = apply_quick_fix(workspace, project_obj, fix)
                steps.append(f"自动修复：{detail}（{fix.reason}）")
                fixes_applied.append(detail)
            project_obj.save()
        if delivery.build is not None and delivery.build.ok:
            delivery.ok = True
        else:
            delivery.ok = False
        if fixes_applied:
            delivery.steps.append("修复汇总：" + "；".join(fixes_applied))

    # ---- 交付文档
    delivery.changelog_path = _append_changelog(delivery, changed, points)
    delivery.report_path = _write_report(delivery, points)
    return delivery


def _project_name(spec: AppSpec) -> str:
    """给 Keil 工程起一个纯 ASCII 名字（工程路径带中文在部分工具链下不稳）。"""
    parts = [spec.device.name.lower()]
    for item in spec.pwm:
        parts.append(f"pwm_{item.pin.name.lower()}")
    for item in spec.uarts:
        parts.append(item.instance.lower())
    for item in spec.timers:
        parts.append(f"{item.timer.lower()}_irq")
    if spec.gpios and not spec.pwm:
        parts.append("gpio_" + "_".join(g.pin.name.lower() for g in spec.gpios[:2]))
    return "-".join(parts)[:60]


def _append_changelog(delivery: Delivery, changed: list[str], points: list[str]) -> Path:
    path = delivery.workspace / "CHANGES.md"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"\n## {stamp} · {delivery.requirement}\n"]
    lines.append(f"- 器件：{delivery.spec.device.name}，SYSCLK {delivery.spec.sysclk / 1e6:g} MHz")
    if points:
        lines.append("- 解析要点：" + "；".join(points))
    lines.append("- 变更文件：" + ("、".join(changed) if changed else "无"))
    if delivery.build is not None:
        state = "通过" if delivery.build.ok else "失败"
        lines.append(f"- 编译：{state}（{delivery.build.program_size or '无体积信息'}）")
        for issue in delivery.build.errors[:5]:
            lines.append(f"  - {issue.text()}")
    lines.append("")
    header = "# 修改记录\n\n每条需求一次追加，便于回溯“谁在什么时候改了什么、编译结果如何”。\n"
    if not path.is_file():
        path.write_text(header, encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def _write_report(delivery: Delivery, points: list[str]) -> Path:
    path = delivery.workspace / "REPORT.md"
    spec = delivery.spec
    plan = delivery.plan
    lines = [f"# {delivery.requirement}", "", f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    if delivery.error:
        lines.extend(["## 失败原因", "", delivery.error, ""])
    if spec is not None:
        lines.extend(
            [
                "## 规格",
                "",
                f"- 器件：**{spec.device.name}**（{spec.device.flash_kb} KB Flash / {spec.device.ram_kb} KB RAM）",
                f"- 系统时钟：{spec.sysclk / 1e6:g} MHz（HSE {spec.hse / 1e6:g} MHz）",
            ]
        )
        for line in points:
            lines.append(f"- {line}")
        lines.append("")
        summary = spec_summary(spec)
        lines.extend(["```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""])
    if plan is not None:
        lines.extend(["## 生成依据（代码生成时算过的账）", ""])
        lines.extend(f"- {item}" for item in plan.checks)
        if plan.warnings:
            lines.extend(["", "### 提醒", ""])
            lines.extend(f"- ⚠️ {item}" for item in plan.warnings)
        if plan.wiring:
            lines.extend(["", "## 接线 / 验证", ""])
            lines.extend(f"- {item}" for item in plan.wiring)
        lines.extend(["", "## 生成文件", ""])
        lines.extend(f"- `{name}`" for name in plan.files)
        lines.append("")
    if delivery.problems:
        lines.extend(["## 未覆盖的诉求", ""])
        lines.extend(f"- {item}" for item in delivery.problems)
        lines.append("")
    if delivery.build is not None:
        build = delivery.build
        lines.extend(
            [
                "## 编译结果",
                "",
                f"- 状态：{'✅ 通过' if build.ok else '❌ 失败'}（退出码 {build.exit_code}）",
                f"- Program Size：{build.program_size or '—'}",
                f"- 耗时：{build.elapsed or f'{build.seconds}s'}",
                f"- 日志：`{build.log_path}`",
            ]
        )
        if build.errors:
            lines.append("")
            lines.append("### 错误")
            lines.extend(f"- `{issue.text()}`" for issue in build.errors[:10])
        if build.warnings:
            lines.append("")
            lines.append("### 警告")
            lines.extend(f"- `{issue.text()}`" for issue in build.warnings[:10])
        lines.append("")
    if delivery.steps:
        lines.extend(["## 执行步骤", ""])
        lines.extend(f"{i}. {item}" for i, item in enumerate(delivery.steps, start=1))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
