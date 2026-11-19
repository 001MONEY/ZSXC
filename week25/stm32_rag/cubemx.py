"""CubeMX 工程识别与增量修改（不覆盖 CubeMX 生成的代码）。

为什么需要单独一层：`generate_application` 是「整文件生成」，适合从零建工程；
但 CubeMX 生成的 `main.c / stm32f1xx_hal_msp.c / stm32f1xx_it.c` 里有：

- `/* USER CODE BEGIN xxx */ ... /* USER CODE END xxx */` 用户代码区；
- 已经生成好的 `MX_GPIO_Init` / `SystemClock_Config` / `HAL_MspInit`；
- 按外设拆分的 `tim.c` / `usart.c` / `gpio.c`。

直接覆盖会丢掉这些内容，还会出现同名函数重复定义。这里的做法是：

1. 读 `.ioc`：拿到器件、已配置外设、引脚信号、波特率；
2. 与需求比对：已配置的直接跳过（改波特率就定点替换），未配置的才生成；
3. 生成时用「内联 MSP」模式（时钟/引脚写进 init 函数），不定义 `HAL_xxx_MspInit`；
4. 只往 USER CODE 段里插入，并用 `/* stm32_rag:xxx:BEGIN/END */` 包起来，
   重复执行会替换而不是叠加。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from .codegen import (
    AppSpec,
    merge_specs,
    plan_application,
    render_gpio_init,
    spec_from_payload,
    spec_to_dict,
)
from .mcu import Pin, channel_of_pin
from .project import KeilProject

STATE_DIR = ".stm32rag"
STATE_FILE = "spec.json"


def load_state(root: str | Path) -> AppSpec | None:
    """读回上一轮累积的规格（多轮需求叠加用）。"""
    path = Path(root).expanduser().resolve() / STATE_DIR / STATE_FILE
    if not path.is_file():
        return None
    try:
        return spec_from_payload(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def save_state(root: str | Path, spec: AppSpec) -> Path:
    path = Path(root).expanduser().resolve() / STATE_DIR / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec_to_dict(spec), ensure_ascii=False, indent=2), encoding="utf-8")
    return path

# ---------------------------------------------------------------- .ioc

@dataclass
class CubeMXInfo:
    path: Path
    mcu: str = ""
    package: str = ""
    version: str = ""
    peripherals: list[str] = field(default_factory=list)
    signals: dict[str, str] = field(default_factory=dict)  # 引脚键（可能带 -TAMPER-RTC 后缀）-> 信号
    bauds: dict[str, str] = field(default_factory=dict)  # USART1 -> "115200"
    sysclk: str = ""

    def pin_signal(self, pin: Pin) -> str | None:
        for key, value in self.signals.items():
            if key.split("-")[0].strip().upper() == pin.name:
                return value
        return None

    def has_peripheral(self, name: str) -> bool:
        target = name.upper()
        return any(item.upper() == target for item in self.peripherals)

    def configured_uarts(self) -> list[str]:
        found: list[str] = []
        for value in self.signals.values():
            match = re.match(r"(USART\d|UART\d)_", value or "")
            if match:
                found.append(match.group(1))
        return sorted(set(found))

    def describe(self) -> str:
        pins = [f"{key}={value}" for key, value in sorted(self.signals.items()) if value]
        lines = [
            f"CubeMX  : {self.path}",
            f"器件    : {self.mcu}（{self.package}）" if self.mcu else "器件    : 未知",
            f"外设    : {', '.join(self.peripherals) or '（无）'}",
        ]
        if self.bauds:
            lines.append("波特率  : " + "，".join(f"{k}={v}" for k, v in sorted(self.bauds.items())))
        if self.sysclk:
            lines.append(f"SYSCLK  : {self.sysclk}")
        lines.append("引脚    : " + (", ".join(pins[:16]) + ("…" if len(pins) > 16 else "") if pins else "（无）"))
        return "\n".join(lines)


def find_ioc(root: str | Path) -> Path | None:
    root = Path(root).expanduser()
    if root.is_file() and root.suffix.lower() == ".ioc":
        return root
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("*.ioc"))
    if not candidates:
        candidates = sorted(root.rglob("*.ioc"))
    return candidates[0] if candidates else None


def parse_ioc(path: str | Path) -> CubeMXInfo:
    path = Path(path).expanduser().resolve()
    text = path.read_text(encoding="utf-8", errors="replace")
    info = CubeMXInfo(path=path)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if re.fullmatch(r"Mcu\.IP\d+", key):
            info.peripherals.append(value)
        elif key.endswith(".Signal"):
            info.signals[key[: -len(".Signal")]] = value
        elif key.endswith(".BaudRate"):
            info.bauds[key[: -len(".BaudRate")]] = value
        elif key in ("Mcu.CPN", "Mcu.UserName") and value:
            info.mcu = info.mcu or value
        elif key == "Mcu.Name" and value:
            info.mcu = info.mcu or value
        elif key == "Mcu.Package":
            info.package = value
        elif key in ("File.Version", "MxCube.Version"):
            info.version = info.version or value
        elif key in ("RCC.SYSCLKFreq_VALUE", "RCC.SYSCLKFreq_Value", "RCC.HCLKFreq_Value"):
            info.sysclk = value
    return info


# ---------------------------------------------------------------- 注入锚点

_MARK = "stm32_rag"


class AnchorMissing(RuntimeError):
    """CubeMX 文件里找不到 USER CODE 锚点（常见于手工删掉了标记）。"""


def _block(name: str) -> tuple[str, str]:
    return f"/* {_MARK}:{name}:BEGIN */", f"/* {_MARK}:{name}:END */"


def _strip(text: str, name: str) -> str:
    begin, end = _block(name)
    return re.sub(
        rf"[ \t]*{re.escape(begin)}.*?{re.escape(end)}\n?",
        "",
        text,
        flags=re.DOTALL,
    )


def insert_section(text: str, anchor: str, code: str, name: str) -> str:
    """把代码插到 `/* USER CODE BEGIN <anchor> */` 之后；重复执行会先删旧块。"""
    code = (code or "").strip("\n")
    if not code:
        return _strip(text, name)
    text = _strip(text, name)
    pattern = re.compile(rf"/\*\s*USER CODE BEGIN {re.escape(anchor)}\s*\*/")
    match = pattern.search(text)
    if not match:
        raise AnchorMissing(f"找不到锚点 USER CODE BEGIN {anchor}")
    begin, end = _block(name)
    block = f"\n{begin}\n{code}\n{end}"
    return text[: match.end()] + block + text[match.end() :]


def _has_definition(text: str, function: str) -> bool:
    return re.search(rf"\bvoid\s+{re.escape(function)}\s*\(", text) is not None


# ---------------------------------------------------------------- 定点修改

def set_uart_baud(root: str | Path, instance: str, baud: int) -> list[str]:
    """把工程里 `huartN.Init.BaudRate = X;` 与 .ioc 一起改成新波特率。"""
    root = Path(root).expanduser().resolve()
    number = instance.upper().replace("USART", "").replace("UART", "")
    pattern = re.compile(rf"(\bhuart{number}\.Init\.BaudRate\s*=\s*)(\d+)")
    changed: list[str] = []
    for path in sorted((root / "Core" / "Src").rglob("*.c")):
        text = path.read_text(encoding="utf-8", errors="replace")
        new_text, count = pattern.subn(lambda m: f"{m.group(1)}{baud}", text)
        if count:
            path.write_text(new_text, encoding="utf-8")
            changed.append(f"{path.relative_to(root)}（{count} 处）")
    ioc = find_ioc(root)
    if ioc is not None:
        text = ioc.read_text(encoding="utf-8", errors="replace")
        new_text, count = re.subn(
            rf"(?m)^{instance}\.BaudRate=\d+$", f"{instance}.BaudRate={baud}", text
        )
        if count:
            ioc.write_text(new_text, encoding="utf-8")
            changed.append(f"{ioc.name}（.ioc 同步）")
    if not changed:
        raise AnchorMissing(f"工程里找不到 {instance} 的波特率赋值语句")
    return changed


# ---------------------------------------------------------------- 增量应用

@dataclass
class PatchResult:
    ok: bool
    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    modules: list[str] = field(default_factory=list)
    spec: AppSpec | None = None


def apply_to_project(
    root: str | Path,
    spec: AppSpec,
    project: KeilProject | None = None,
) -> PatchResult:
    """在 CubeMX 工程上增量应用一份规格。"""
    root = Path(root).expanduser().resolve()
    ioc = find_ioc(root)
    if ioc is None:
        return PatchResult(ok=False, errors=[f"{root} 下没有 .ioc，不是 CubeMX 工程"])
    info = parse_ioc(ioc)
    main_c = root / "Core" / "Src" / "main.c"
    if not main_c.is_file():
        return PatchResult(ok=False, errors=[f"找不到 {main_c}"])
    main_text = main_c.read_text(encoding="utf-8", errors="replace")
    if "USER CODE BEGIN" not in main_text:
        return PatchResult(
            ok=False,
            errors=[
                "main.c 里没有 USER CODE 锚点（可能被手工删掉了）。"
                "这种情况请先用 read_source 看清代码再局部修改，或重新用 CubeMX 生成。"
            ],
        )

    result = PatchResult(ok=True)
    # 多轮需求叠加：把新规格合并到上一轮记录的规格上（同键覆盖，其余保留）
    merged = merge_specs(load_state(root), spec)
    result.spec = merged
    filtered = replace(
        merged,
        pwm=list(merged.pwm),
        uarts=list(merged.uarts),
        timers=list(merged.timers),
        gpios=list(merged.gpios),
    )

    # ---- 串口：已配置的改波特率，未配置的才生成
    for uart in list(filtered.uarts):
        if info.has_peripheral(uart.instance) or uart.instance in info.configured_uarts():
            current = info.bauds.get(uart.instance)
            if current is not None and int(current) == uart.baud:
                result.skipped.append(f"{uart.instance} 已由 CubeMX 配置为 {uart.baud}，未改动")
            else:
                try:
                    changed = set_uart_baud(root, uart.instance, uart.baud)
                    result.applied.append(f"{uart.instance} 波特率 → {uart.baud}：{'、'.join(changed)}")
                    result.changed_files.extend(changed)
                except AnchorMissing as exc:
                    result.errors.append(str(exc))
                    result.ok = False
            filtered.uarts.remove(uart)

    # ---- PWM：同引脚同信号视为已配置；不同信号直接报冲突
    for pwm in list(filtered.pwm):
        signal = info.pin_signal(pwm.pin)
        if not signal:
            continue
        timer = pwm.timer or ""
        channel = pwm.channel or (channel_of_pin(pwm.pin, timer) if timer else None)
        expected = f"{timer}_{channel}" if timer and channel else ""
        if expected and signal.upper() == expected.upper():
            result.skipped.append(f"{pwm.pin.name} 已由 CubeMX 配置为 {signal}，未改动")
        else:
            result.errors.append(
                f"{pwm.pin.name} 在 .ioc 里已占用（{signal}），不能再生成 {expected or 'PWM'}；"
                "请换引脚或在 CubeMX 里修改"
            )
            result.ok = False
        filtered.pwm.remove(pwm)

    # ---- 定时器中断：已作为外设配置的定时器不能重复初始化
    for timer in list(filtered.timers):
        if info.has_peripheral(timer.timer):
            result.skipped.append(f"{timer.timer} 已是 CubeMX 外设，未重复生成初始化（中断可在 CubeMX 里使能）")
            filtered.timers.remove(timer)

    # ---- GPIO：已配置的引脚不重复初始化
    for gpio in list(filtered.gpios):
        signal = info.pin_signal(gpio.pin)
        if signal:
            result.skipped.append(f"{gpio.pin.name} 已由 CubeMX 配置为 {signal}，未重复初始化")
            filtered.gpios.remove(gpio)

    if not result.ok:
        return result

    # ---- 生成并插入
    plan = plan_application(filtered, inline_msp=True)
    result.checks = plan.checks
    result.modules = plan.modules
    if not plan.ok:
        result.ok = False
        result.errors.extend(plan.errors)
        return result

    sections = plan.sections
    has_new = any(item.strip() for key, item in sections.items() if key != "printf")
    if not has_new:
        save_state(root, merged)
        return result

    # main.c：每个块都重写一次（内容为空时等于删除上一轮的旧块，避免留下失效函数）
    text = main_text
    text = insert_section(text, "Includes", "#include <stdio.h>" if sections.get("printf") else "", "includes")
    text = insert_section(text, "PV", sections["handles"], "handles")
    text = insert_section(text, "PFP", sections["prototypes"], "prototypes")

    gpio_block = ""
    gpio_call = ""
    if filtered.gpios:
        gpio_block = render_gpio_init(filtered, filtered.gpios, name="stm32rag_gpio_init")
        gpio_call = "  stm32rag_gpio_init();"

    combined = "\n".join(item for item in (gpio_call, sections["init_calls"], sections["start"]) if item)
    text = insert_section(text, "2", combined, "init_calls")

    loop_anchor = "WHILE" if re.search(r"/\*\s*USER CODE BEGIN WHILE\s*\*/", text) else "3"
    text = insert_section(text, loop_anchor, sections["loop"], "loop")

    tail_parts = [
        sections["init_funcs"],
        sections["nvic"],
        gpio_block,
        sections["callbacks"],
        sections["retarget"],
    ]
    tail = "\n\n".join(item for item in tail_parts if item.strip())
    text = insert_section(text, "4", tail, "functions")

    main_c.write_text(text, encoding="utf-8")
    result.changed_files.append("Core/Src/main.c")

    # stm32f1xx_it.c：只加中断函数（同样先清旧块）
    it_c = root / "Core" / "Src" / "stm32f1xx_it.c"
    if it_c.is_file():
        it_text = it_c.read_text(encoding="utf-8", errors="replace")
        externs = "\n".join(
            "extern " + line
            for line in sections["handles"].splitlines()
            if line.strip().endswith(";")
        )
        code = "\n".join(item for item in (externs, sections["it_handlers"]) if item)
        try:
            it_text = insert_section(it_text, "1", code, "irq")
            it_c.write_text(it_text, encoding="utf-8")
            if code:
                result.changed_files.append("Core/Src/stm32f1xx_it.c")
        except AnchorMissing:
            if code:
                result.warnings.append("stm32f1xx_it.c 没有 USER CODE BEGIN 1，中断函数未插入")

    result.applied.insert(0, "按 USER CODE 段插入生成代码（"+ "、".join(
        key for key in ("handles", "prototypes", "init_calls", "loop", "functions") if sections.get(key)
    ) + "）")
    save_state(root, merged)
    return result
