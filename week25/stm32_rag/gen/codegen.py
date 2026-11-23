"""外设代码生成：把「配置 PA8 输出 1 MHz PWM」这类需求变成可编译的 HAL 代码。

分工：

- `mcu.py` 负责静态事实（引脚复用、DMA 映射、分频计算）；
- 本模块负责把事实变成 `main.c / main.h / stm32f1xx_it.c`，
  并且**先校验后生成**：引脚复用不对、频率分不出来、波特率误差过大时直接报错，
  不会先写出一份编译不过的代码再去猜。

生成的代码沿用 CubeMX 的命名习惯（`MX_xxx_Init` / `HAL_xxx_MspInit`），
这样和 UM1850 / HAL 源码里的函数名能一一对上，检索到的资料可以直接验证。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .mcu import (
    ADVANCED_TIMERS,
    I2C_PINS,
    USART_PINS,
    Device,
    Pin,
    Timing,
    adc_channel_of_pin,
    device_i2cs,
    dma_channel,
    dma_irq,
    gpio_clock_ok,
    pwm_timing,
    resolve_device,
    timer_clock,
    timer_for_pin,
    uart_brr,
)

# ---------------------------------------------------------------- 规格

@dataclass
class PwmSpec:
    """一个 PWM 输出通道。"""

    pin: Pin
    timer: str = ""  # 留空则按引脚自动选
    channel: str = ""  # "CH1"
    freq_hz: float = 1000.0
    duty: float = 0.5


@dataclass
class GpioSpec:
    """普通 GPIO 配置（输出/输入/外部中断）。"""

    pin: Pin
    mode: str = "output_pp"  # output_pp/output_od/input/input_pullup/analog/it_rising/it_falling
    initial: str = ""  # "high" / "low" / ""
    label: str = ""


@dataclass
class UartSpec:
    """串口配置。"""

    instance: str = "USART1"
    baud: int = 115200
    tx: Pin | None = None
    rx: Pin | None = None
    dma: bool = False
    printf: bool = True


@dataclass
class TimerSpec:
    """定时器更新中断（可顺带翻转一个引脚）。"""

    timer: str = "TIM2"
    freq_hz: float = 1000.0
    toggle_pin: Pin | None = None


@dataclass
class LoopSpec:
    """主循环里的周期性动作。"""

    kind: str = "none"  # none / toggle / print
    pin: Pin | None = None
    interval_ms: int = 500
    message: str = ""


@dataclass
class KeySpec:
    """按键组件：低电平有效，横跨扫描 + 20 ms 消抖。"""

    pins: list[Pin] = field(default_factory=list)
    active_low: bool = True
    debounce_ms: int = 20


@dataclass
class BuzzerSpec:
    """无源蜂鸣器：占用一个 PWM 通道，运行时改频率。"""

    pin: Pin = field(default_factory=lambda: Pin("B", 0))
    timer: str = ""
    channel: str = ""


@dataclass
class SerialCmdSpec:
    """串口命令行：中断接收 + 环形缓冲 + 行解析。"""

    instance: str = "USART1"


@dataclass
class AdcSpec:
    """ADC 旋钮：DMA 循环采样 + 滑动平均。"""

    pin: Pin = field(default_factory=lambda: Pin("A", 0))
    samples: int = 8
    label: str = "旋钮"


@dataclass
class OledSpec:
    """SSD1306 0.96\" OLED（硬件 I2C）。"""

    instance: str = "I2C1"
    address: int = 0x78
    width: int = 128
    height: int = 64


@dataclass
class AppSpec:
    """一次生成请求的全部输入。"""

    device: Device
    sysclk: int = 72_000_000
    hse: int = 8_000_000
    pwm: list[PwmSpec] = field(default_factory=list)
    gpios: list[GpioSpec] = field(default_factory=list)
    uarts: list[UartSpec] = field(default_factory=list)
    timers: list[TimerSpec] = field(default_factory=list)
    loop: LoopSpec | None = None
    keys: KeySpec | None = None
    buzzer: BuzzerSpec | None = None
    serial_cmd: SerialCmdSpec | None = None
    adc: AdcSpec | None = None
    oled: OledSpec | None = None
    sched: bool = False
    title: str = "app"

    def all_pins(self) -> list[Pin]:
        pins: list[Pin] = [spec.pin for spec in self.pwm] + [spec.pin for spec in self.gpios]
        for uart in self.uarts:
            pins.extend(p for p in (uart.tx, uart.rx) if p)
        for timer in self.timers:
            if timer.toggle_pin:
                pins.append(timer.toggle_pin)
        if self.keys:
            pins.extend(self.keys.pins)
        if self.buzzer:
            pins.append(self.buzzer.pin)
        if self.adc:
            pins.append(self.adc.pin)
        return list(dict.fromkeys(pins))


@dataclass
class CodePlan:
    """生成结果：可以落盘的文件 + 依据 + 接线说明。"""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    wiring: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    modules: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    main_c: str = ""
    sections: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------- 校验

def validate(spec: AppSpec) -> tuple[list[str], list[str], list[str]]:
    """返回 (错误, 警告, 依据说明)。所有硬约束都必须在生成前通过。"""
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []

    for pin in spec.all_pins():
        if not gpio_clock_ok(pin, spec.device):
            errors.append(f"{pin.name}: {spec.device.name} 是 {len(spec.device.ports)} 个端口的封装，没有 GPIO{pin.port}")

    for pwm in spec.pwm:
        picked = timer_for_pin(pwm.pin, spec.device, prefer=pwm.timer or None)
        timer = pwm.timer or (picked[0] if picked else "")
        channel = pwm.channel or (picked[1] if picked else "")
        if not timer or not channel:
            options = [f"{t}_{c}" for t, c in _pin_options(pwm.pin, spec.device)]
            hint = f"；可用：{', '.join(options)}" if options else "；这颗封装里该引脚没有定时器复用"
            errors.append(f"{pwm.pin.name} 不能输出 {pwm.timer or 'PWM'}{hint}")
            continue
        valid = [f"{t}_{c}" for t, c in _pin_options(pwm.pin, spec.device)]
        if f"{timer}_{channel}" not in valid:
            errors.append(
                f"{pwm.pin.name} 的定时器复用是 {', '.join(valid) or '无'}，不能配成 {timer}_{channel}"
            )
            continue
        tim_clk, source = timer_clock(timer, spec.sysclk)
        try:
            timing = pwm_timing(tim_clk, pwm.freq_hz, pwm.duty)
        except ValueError as exc:
            errors.append(f"{timer}_{channel} {pwm.freq_hz:g} Hz：{exc}")
            continue
        checks.append(f"{pwm.pin.name} → {timer}_{channel}；{source}；{timing.describe(pwm.freq_hz)}")
        error_pct = abs(timing.actual_hz - pwm.freq_hz) / pwm.freq_hz * 100
        if error_pct > 0.5:
            warnings.append(
                f"{timer}_{channel} 实际 {timing.actual_hz / 1000:g} kHz，与目标 {pwm.freq_hz / 1000:g} kHz "
                f"相差 {error_pct:.2f}%（PSC/ARR 只有 16 位，无法精确整除）"
            )

    for uart in spec.uarts:
        if uart.instance not in USART_PINS:
            errors.append(f"不支持的串口实例：{uart.instance}（可选 {', '.join(USART_PINS)}）")
            continue
        expected_tx = USART_PINS[uart.instance]["TX"][0]
        expected_rx = USART_PINS[uart.instance]["RX"][0]
        if uart.tx and uart.tx != expected_tx:
            errors.append(f"{uart.instance}_TX 默认引脚是 {expected_tx.name}，不是 {uart.tx.name}（用重映射需要额外改 AFIO）")
        if uart.rx and uart.rx != expected_rx:
            errors.append(f"{uart.instance}_RX 默认引脚是 {expected_rx.name}，不是 {uart.rx.name}（用重映射需要额外改 AFIO）")
        pclk = spec.sysclk // (1 if uart.instance == "USART1" else 2)
        brr, actual, error = uart_brr(pclk, uart.baud)
        checks.append(
            f"{uart.instance}：PCLK={pclk / 1e6:g} MHz，BRR=0x{brr:X}，实际 {actual:.1f} bps（误差 {error:+.2f}%）"
        )
        if abs(error) > 3.0:
            warnings.append(f"{uart.instance} {uart.baud} bps 误差 {error:+.2f}%，超过 3%，通信可能不稳")
        if uart.dma:
            channel = dma_channel(uart.instance, "TX")
            if channel:
                checks.append(f"{uart.instance}_TX 的 DMA 固定映射：{channel}（RM0008 表 14-4）")
            else:
                errors.append(f"{uart.instance}_TX 没有 DMA 请求映射")

    for timer in spec.timers:
        if timer.timer in ADVANCED_TIMERS:
            pass  # TIM1 更新中断走 TIM1_UP_IRQn，已在生成器里处理
        tim_clk, source = timer_clock(timer.timer, spec.sysclk)
        try:
            timing = pwm_timing(tim_clk, timer.freq_hz, 0.5)
        except ValueError as exc:
            errors.append(f"{timer.timer} 定时 {timer.freq_hz:g} Hz：{exc}")
            continue
        checks.append(f"{timer.timer} 更新中断 {timing.describe(timer.freq_hz)}；{source}")
        if timer.toggle_pin and not gpio_clock_ok(timer.toggle_pin, spec.device):
            errors.append(f"{timer.toggle_pin.name}: 该封装没有 GPIO{timer.toggle_pin.port}")

    if spec.loop and spec.loop.kind == "toggle":
        if not spec.loop.pin:
            errors.append("主循环翻转动作缺少引脚")
        elif not any(item.pin == spec.loop.pin for item in spec.gpios):
            warnings.append(f"主循环要翻转 {spec.loop.pin.name}，但没有为它生成 GPIO 输出配置，已自动补上")

    # ---- 同一定时器多通道：频率必须一致，通道不能重叠
    pwm_groups: dict[str, list[tuple[str, Pin, float]]] = {}
    for pwm in spec.pwm:
        picked = timer_for_pin(pwm.pin, spec.device, prefer=pwm.timer or None)
        if picked is None:
            continue
        timer, channel = (pwm.timer, pwm.channel) if pwm.timer and pwm.channel else picked
        pwm_groups.setdefault(timer, []).append((channel, pwm.pin, pwm.freq_hz))
    for timer, items in pwm_groups.items():
        freqs = sorted({round(item[2], 3) for item in items})
        if len(freqs) > 1:
            text = "、".join(_fmt_hz(freq) for freq in freqs)
            errors.append(f"{timer} 被要求输出 {text}：同一定时器只能有一个频率（共享 PSC/ARR）")
        seen: dict[str, str] = {}
        for channel, pin, _ in items:
            if channel in seen:
                errors.append(f"{timer}_{channel} 被 {seen[channel]} 和 {pin.name} 同时占用")
            seen[channel] = pin.name
        if len(items) > 1:
            checks.append(
                f"{timer} 多通道：{', '.join(f'{ch}={pin.name}' for ch, pin, _ in items)}"
                f"（PSC/ARR 共用，占空比各自独立）"
            )

    # ---- 应用层组件
    if spec.keys is not None:
        if not spec.keys.pins:
            errors.append("按键组件没有给引脚")
        if len(spec.keys.pins) > 8:
            errors.append("按键组件最多支持 8 个键")
        else:
            checks.append(
                "按键："
                + "、".join(pin.name for pin in spec.keys.pins)
                + f"（{'低' if spec.keys.active_low else '高'}电平有效，{spec.keys.debounce_ms} ms 消抖）"
            )
    if spec.buzzer is not None:
        picked = timer_for_pin(spec.buzzer.pin, spec.device, prefer=spec.buzzer.timer or None)
        if picked is None:
            errors.append(f"蜂鸣器 {spec.buzzer.pin.name} 没有定时器复用，无法输出音调")
        else:
            timer, channel = (
                (spec.buzzer.timer, spec.buzzer.channel)
                if spec.buzzer.timer and spec.buzzer.channel
                else picked
            )
            checks.append(f"蜂鸣器 {spec.buzzer.pin.name} → {timer}_{channel}（预分频固定到 1 MHz，ARR 按音名算）")
    if spec.serial_cmd is not None:
        if not any(uart.instance == spec.serial_cmd.instance for uart in spec.uarts):
            errors.append(f"串口命令行需要先配置 {spec.serial_cmd.instance}（在 uart 里加一条）")
        else:
            checks.append(f"串口命令行：{spec.serial_cmd.instance} 中断接收 + 256 字节环形缓冲 + 行解析")
    if spec.adc is not None:
        channel = adc_channel_of_pin(spec.adc.pin)
        if channel is None:
            errors.append(f"{spec.adc.pin.name} 不是 ADC1 的输入脚（可用 PA0~PA7 / PB0 / PB1）")
        elif not 2 <= spec.adc.samples <= 64:
            errors.append("滑动平均样本数要在 2~64 之间")
        else:
            checks.append(
                f"ADC1_IN{channel} ← {spec.adc.pin.name}；DMA1_Channel1 循环采样 {spec.adc.samples} 点取平均"
            )
    if spec.oled is not None:
        if spec.oled.instance not in I2C_PINS:
            errors.append(f"不支持的 I2C 实例：{spec.oled.instance}")
        elif spec.oled.instance not in device_i2cs(spec.device):
            errors.append(f"{spec.device.name} 没有引出 {spec.oled.instance} 的默认引脚")
        else:
            scl = I2C_PINS[spec.oled.instance]["SCL"].name
            sda = I2C_PINS[spec.oled.instance]["SDA"].name
            checks.append(
                f"OLED：{spec.oled.instance} SCL={scl} / SDA={sda}，地址 0x{spec.oled.address:02X}，"
                f"{spec.oled.width}x{spec.oled.height}"
            )

    # ---- 引脚一物多主：先提醒，硬冲突（同一 PWM 通道）上面已经报错
    usage: dict[str, str] = {}

    def _claim(pin: Pin, owner: str) -> None:
        if pin.name in usage and usage[pin.name] != owner:
            warnings.append(f"{pin.name} 同时被「{usage[pin.name]}」和「{owner}」使用，请确认接线")
        usage.setdefault(pin.name, owner)

    for pwm in spec.pwm:
        _claim(pwm.pin, "PWM")
    for gpio in spec.gpios:
        _claim(gpio.pin, "GPIO")
    for uart in spec.uarts:
        for pin in (uart.tx, uart.rx):
            if pin:
                _claim(pin, uart.instance)
    for timer in spec.timers:
        if timer.toggle_pin:
            _claim(timer.toggle_pin, f"{timer.timer} 中断")
    if spec.keys is not None:
        for pin in spec.keys.pins:
            _claim(pin, "按键")
    if spec.buzzer is not None:
        _claim(spec.buzzer.pin, "蜂鸣器")
    if spec.adc is not None:
        _claim(spec.adc.pin, "ADC")
    if spec.oled is not None and spec.oled.instance in I2C_PINS:
        for pin in I2C_PINS[spec.oled.instance].values():
            _claim(pin, "OLED I2C")

    if spec.pwm and not spec.device.ports:
        errors.append("器件信息不完整")
    return errors, warnings, checks


def _pin_options(pin: Pin, device: Device) -> list[tuple[str, str]]:
    from .mcu import pin_timer_options

    return pin_timer_options(pin, device)


# ---------------------------------------------------------------- 时间格式

def _fmt_hz(value: float) -> str:
    if value >= 1e6:
        return f"{value / 1e6:g} MHz"
    if value >= 1e3:
        return f"{value / 1e3:g} kHz"
    return f"{value:g} Hz"


def _clock_body(sysclk: int, hse: int) -> tuple[str, list[str], int]:
    """返回 SystemClock_Config 函数体、说明、最终 sysclk。"""
    if hse <= 0:
        hse = 8_000_000
    mul = int(round(sysclk / hse))
    notes: list[str] = []
    if not (2 <= mul <= 16) or abs(mul * hse - sysclk) > 1:
        notes.append(f"SYSCLK {sysclk / 1e6:g} MHz 不是 HSE {hse / 1e6:g} MHz 的整数倍，改用 72 MHz")
        sysclk, mul = 72_000_000, 9
    latency = 2 if sysclk > 48e6 else (1 if sysclk > 24e6 else 0)
    notes.append(f"SYSCLK {sysclk / 1e6:g} MHz = HSE {hse / 1e6:g} MHz × {mul}，FLASH_LATENCY_{latency}")
    body = f"""/* SYSCLK {sysclk} Hz：HSE {hse} Hz × {mul} */
static void SystemClock_Config(void)
{{
  RCC_OscInitTypeDef RCC_OscInitStruct = {{0}};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {{0}};

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL{mul};
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {{
    Error_Handler();
  }}

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                              | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_{latency}) != HAL_OK)
  {{
    Error_Handler();
  }}
}}"""
    return body, notes, sysclk


# ---------------------------------------------------------------- 生成器

class _Builder:
    def __init__(self, spec: AppSpec, inline_msp: bool = False) -> None:
        self.spec = spec
        # CubeMX 工程里 HAL_xxx_MspInit 通常已被 CubeMX 写在 stm32f1xx_hal_msp.c，
        # 再定义一次会重复符号；增量模式下把时钟/引脚配置内联进 init 函数。
        self.inline_msp = inline_msp
        self.handle_decls: list[str] = []
        self.init_protos: list[str] = ["static void MX_GPIO_Init(void);"]
        self.init_funcs: list[str] = []
        self.main_init_calls: list[str] = []
        self.start_calls: list[str] = []
        self.msp_tim: list[str] = []
        self.msp_uart: list[str] = []
        self.nvic: list[str] = []
        self.it_handlers: list[str] = []
        self.callbacks: list[str] = []
        self.loop_lines: list[str] = []
        self.notes: list[str] = []
        self.modules: set[str] = {"gpio"}
        self.printf_handle: str | None = None
        self.comp_includes: list[str] = []
        self.msp_adc: list[str] = []
        self.msp_i2c: list[str] = []
        # timer -> {channels:[(CHx, pulse)], pins:[Pin], irq:bool, timing:Timing, mode}
        self.tim_cfg: dict[str, dict] = {}

    # ---------------------------------------------- 定时器 / PWM
    def _tim_entry(self, timer: str) -> dict:
        entry = self.tim_cfg.get(timer)
        if entry is None:
            entry = {"channels": [], "pins": [], "irq": False, "timing": None, "mode": "base"}
            self.tim_cfg[timer] = entry
            self.modules.add("tim")
            handle = f"htim{timer[3:]}"
            if handle not in "".join(self.handle_decls):
                self.handle_decls.append(f"TIM_HandleTypeDef {handle};")
            self._ensure_timer_init(timer)
        return entry

    def _ensure_timer_init(self, timer: str) -> None:
        proto = f"static void MX_{timer}_Init(void);"
        if proto not in self.init_protos:
            self.init_protos.append(proto)
            self.main_init_calls.append(f"  MX_{timer}_Init();")

    def add_pwm(self, pwm: PwmSpec) -> None:
        picked = timer_for_pin(pwm.pin, self.spec.device, prefer=pwm.timer or None)
        timer, channel = (pwm.timer, pwm.channel) if pwm.timer and pwm.channel else picked  # type: ignore[misc]
        tim_clk, source = timer_clock(timer, self.spec.sysclk)
        timing = pwm_timing(tim_clk, pwm.freq_hz, pwm.duty)
        entry = self._tim_entry(timer)
        entry["timing"] = timing
        entry["mode"] = "pwm" if entry["mode"] == "base" else entry["mode"]
        entry["channels"].append((channel, timing.pulse))
        if pwm.pin not in entry["pins"]:
            entry["pins"].append(pwm.pin)
        handle = f"htim{timer[3:]}"
        self.start_calls.append(
            f"  /* {pwm.pin.name} → {timer}_{channel}：{_fmt_hz(pwm.freq_hz)}，占空比 {pwm.duty * 100:g}% */\n"
            f"  if (HAL_TIM_PWM_Start(&{handle}, TIM_CHANNEL_{channel[2:]}) != HAL_OK)\n"
            "  {\n    Error_Handler();\n  }"
        )
        self.notes.append(f"{pwm.pin.name} → {timer}_{channel}；{source}；{timing.describe(pwm.freq_hz)}")

    def add_timer(self, timer_spec: TimerSpec) -> None:
        timer = timer_spec.timer
        entry = self._tim_entry(timer)
        handle = f"htim{timer[3:]}"
        tim_clk, source = timer_clock(timer, self.spec.sysclk)
        timing = pwm_timing(tim_clk, timer_spec.freq_hz, 0.5)
        if entry["timing"] is None:
            entry["timing"] = timing
        entry["irq"] = True
        self.start_calls.append(
            f"  /* {timer} 更新中断：{_fmt_hz(timer_spec.freq_hz)} */\n"
            f"  if (HAL_TIM_Base_Start_IT(&{handle}) != HAL_OK)\n"
            "  {\n    Error_Handler();\n  }"
        )
        handler = f"{timer}_UP_IRQHandler" if timer in ADVANCED_TIMERS else f"{timer}_IRQHandler"
        self.it_handlers.append(
            f"/* {timer} 更新中断 */\nvoid {handler}(void)\n{{\n  HAL_TIM_IRQHandler(&{handle});\n}}"
        )
        if timer_spec.toggle_pin:
            self.callbacks.append(
                "void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)\n"
                "{\n"
                f"  if (htim->Instance == {timer})\n"
                "  {\n"
                f"    HAL_GPIO_TogglePin({timer_spec.toggle_pin.gpio}, {timer_spec.toggle_pin.pin_macro});\n"
                "  }\n"
                "}"
            )
        self.notes.append(f"{timer} 更新中断：{source}；{timing.describe(timer_spec.freq_hz)}")

    def add_buzzer(self, buzzer: BuzzerSpec) -> None:
        """蜂鸣器只占一个 PWM 通道，频率运行时改，所以初始化用 1 kHz 占位。"""
        picked = timer_for_pin(buzzer.pin, self.spec.device, prefer=buzzer.timer or None)
        timer, channel = (buzzer.timer, buzzer.channel) if buzzer.timer and buzzer.channel else picked  # type: ignore[misc]
        tim_clk, source = timer_clock(timer, self.spec.sysclk)
        timing = pwm_timing(tim_clk, 1000.0, 0.5)
        entry = self._tim_entry(timer)
        if entry["timing"] is None:
            entry["timing"] = timing
        entry["mode"] = "pwm" if entry["mode"] == "base" else entry["mode"]
        entry["channels"].append((channel, timing.pulse))
        if buzzer.pin not in entry["pins"]:
            entry["pins"].append(buzzer.pin)
        self.notes.append(f"蜂鸣器 {buzzer.pin.name} → {timer}_{channel}；{source}；播放时改 ARR")

    def finalize(self) -> None:
        """所有外设登记完再统一发初始化函数：同一 TIM 的多个通道会合进一个 MX_TIMx_Init。"""
        for timer, entry in self.tim_cfg.items():
            self.init_funcs.append(self._tim_init_func(timer, entry))
            irq = f"{timer}_UP_IRQn" if timer in ADVANCED_TIMERS else f"{timer}_IRQn"
            if entry["irq"]:
                if self.inline_msp:
                    self.nvic.append(f"  HAL_NVIC_SetPriority({irq}, 1, 0);\n  HAL_NVIC_EnableIRQ({irq});")
                    if "static void MX_NVIC_Init(void);" not in self.init_protos:
                        self.init_protos.append("static void MX_NVIC_Init(void);")
                        self.main_init_calls.append("  MX_NVIC_Init();")
                else:
                    self._ensure_tim_msp(timer, entry["pins"], irq=irq)
            elif not self.inline_msp:
                self._ensure_tim_msp(timer, entry["pins"])

    def _tim_init_func(self, timer: str, entry: dict) -> str:
        handle = f"htim{timer[3:]}"
        advanced = timer in ADVANCED_TIMERS
        timing: Timing = entry["timing"]
        channels: list[tuple[str, int]] = entry["channels"]
        pins: list[Pin] = entry["pins"]
        lines = [
            f"static void MX_{timer}_Init(void)",
            "{",
            "  TIM_ClockConfigTypeDef sClockSourceConfig = {0};",
        ]
        if channels:
            lines.append("  TIM_OC_InitTypeDef sConfigOC = {0};")
        if advanced:
            lines.append("  TIM_MasterConfigTypeDef sMasterConfig = {0};")
        if self.inline_msp and pins:
            lines.append("  GPIO_InitTypeDef GPIO_InitStruct = {0};")
        lines.append("")
        if self.inline_msp:
            lines.extend(line[2:] for line in self._tim_msp_body(timer, pins, declare=False))
            lines.append("")
        lines.extend(
            [
                f"  {handle}.Instance = {timer};",
                f"  {handle}.Init.Prescaler = {timing.psc};",
                "  " + f"{handle}.Init.CounterMode = TIM_COUNTERMODE_UP;",
                f"  {handle}.Init.Period = {timing.arr};",
                f"  {handle}.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;",
            ]
        )
        if advanced:
            lines.append(f"  {handle}.Init.RepetitionCounter = 0;")
        lines.extend(
            [
                f"  {handle}.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;",
                f"  if (HAL_TIM_Base_Init(&{handle}) != HAL_OK)",
                "  {",
                "    Error_Handler();",
                "  }",
                "",
                "  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;",
                f"  if (HAL_TIM_ConfigClockSource(&{handle}, &sClockSourceConfig) != HAL_OK)",
                "  {",
                "    Error_Handler();",
                "  }",
            ]
        )
        if channels:
            lines.append(f"  if (HAL_TIM_PWM_Init(&{handle}) != HAL_OK)")
            lines.append("  {")
            lines.append("    Error_Handler();")
            lines.append("  }")
        if advanced:
            lines.extend(
                [
                    "",
                    "  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;",
                    "  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;",
                    f"  if (HAL_TIMEx_MasterConfigSynchronization(&{handle}, &sMasterConfig) != HAL_OK)",
                    "  {",
                    "    Error_Handler();",
                    "  }",
                ]
            )
        for channel, pulse in channels:
            lines.extend(
                [
                    "",
                    "  sConfigOC.OCMode = TIM_OCMODE_PWM1;",
                    f"  sConfigOC.Pulse = {pulse};",
                    "  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;",
                ]
            )
            if advanced:
                lines.append("  sConfigOC.OCNPolarity = TIM_OCNPOLARITY_HIGH;")
                lines.append("  sConfigOC.OCIdleState = TIM_OCIDLESTATE_RESET;")
                lines.append("  sConfigOC.OCNIdleState = TIM_OCNIDLESTATE_RESET;")
            lines.append("  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;")
            lines.append(
                f"  if (HAL_TIM_PWM_ConfigChannel(&{handle}, &sConfigOC, TIM_CHANNEL_{channel[2:]}) != HAL_OK)"
            )
            lines.append("  {")
            lines.append("    Error_Handler();")
            lines.append("  }")
        lines.append("}")
        return "\n".join(lines)

    def _ensure_tim_msp(self, timer: str, pins: list[Pin], irq: str | None = None) -> None:
        if any(f"htim->Instance == {timer}" in block for block in self.msp_tim):
            return
        body = self._tim_msp_body(timer, pins, irq)
        self.msp_tim.append("\n".join([f"  if (htim->Instance == {timer})", "  {"] + body + ["  }"]))

    def _tim_msp_body(
        self,
        timer: str,
        pins: list[Pin],
        irq: str | None = None,
        declare: bool = True,
    ) -> list[str]:
        """时钟使能 + 引脚复用 + NVIC 的语句行（四条空格缩进）。"""
        lines = [f"    __HAL_RCC_{timer}_CLK_ENABLE();"]
        if pins:
            for port in dict.fromkeys(pin.port for pin in pins):
                lines.append(f"    __HAL_RCC_GPIO{port}_CLK_ENABLE();")
            if declare:
                lines.append("    GPIO_InitTypeDef GPIO_InitStruct = {0};")
            lines.append("")
            lines.append("    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;")
            lines.append("    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;")
            for port in dict.fromkeys(pin.port for pin in pins):
                port_pins = [p for p in pins if p.port == port]
                lines.append("    GPIO_InitStruct.Pin = " + " | ".join(p.pin_macro for p in port_pins) + ";")
                lines.append(f"    HAL_GPIO_Init(GPIO{port}, &GPIO_InitStruct);")
        if irq:
            lines.append(f"    HAL_NVIC_SetPriority({irq}, 1, 0);")
            lines.append(f"    HAL_NVIC_EnableIRQ({irq});")
        return lines

    # ---------------------------------------------- UART
    def add_uart(self, uart: UartSpec) -> None:
        self.modules.add("uart")
        handle = f"huart{uart.instance[5:]}"
        self.handle_decls.append(f"UART_HandleTypeDef {handle};")
        self.init_protos.append(f"static void MX_{uart.instance}_UART_Init(void);")
        self.main_init_calls.append(f"  MX_{uart.instance}_UART_Init();")
        tx = uart.tx or USART_PINS[uart.instance]["TX"][0]
        rx = uart.rx or USART_PINS[uart.instance]["RX"][0]

        decls = ["  GPIO_InitTypeDef GPIO_InitStruct = {0};"] if self.inline_msp else []
        preamble: list[str] = []
        if self.inline_msp:
            preamble = [
                line[2:]
                for line in self._uart_msp_body(uart, tx, rx, declare=False, link_expr=f"&{handle}")
            ]
        body = [
            f"static void MX_{uart.instance}_UART_Init(void)",
            "{",
            *decls,
            "",
            *preamble,
            *( [""] if preamble else [] ),
            f"  {handle}.Instance = {uart.instance};",
            f"  {handle}.Init.BaudRate = {uart.baud};",
            f"  {handle}.Init.WordLength = UART_WORDLENGTH_8B;",
            f"  {handle}.Init.StopBits = UART_STOPBITS_1;",
            f"  {handle}.Init.Parity = UART_PARITY_NONE;",
            f"  {handle}.Init.Mode = UART_MODE_TX_RX;",
            f"  {handle}.Init.HwFlowCtl = UART_HWCONTROL_NONE;",
            f"  {handle}.Init.OverSampling = UART_OVERSAMPLING_16;",
            f"  if (HAL_UART_Init(&{handle}) != HAL_OK)",
            "  {",
            "    Error_Handler();",
            "  }",
            "}",
        ]
        self.init_funcs.append("\n".join(body))
        if uart.printf:
            self.printf_handle = handle
        if not self.inline_msp:
            self._add_uart_msp(uart, tx, rx)
        if uart.dma:
            channel = dma_channel(uart.instance, "TX")
            if channel:
                name = f"hdma_{uart.instance.lower()}_tx"
                self.handle_decls.append(f"DMA_HandleTypeDef {name};")
                self.main_init_calls.append("  MX_NVIC_Init();")
                if "static void MX_NVIC_Init(void);" not in self.init_protos:
                    self.init_protos.append("static void MX_NVIC_Init(void);")
                self.nvic.append(
                    f"  /* {uart.instance}_TX → {channel} */\n"
                    f"  __HAL_RCC_DMA1_CLK_ENABLE();\n"
                    f"  HAL_NVIC_SetPriority({dma_irq(channel)}, 1, 0);\n"
                    f"  HAL_NVIC_EnableIRQ({dma_irq(channel)});"
                )
                self.it_handlers.append(
                    f"void {dma_irq(channel)[:-4]}Handler(void)\n{{\n  HAL_DMA_IRQHandler(&{name});\n}}"
                )
                self.modules.add("dma")
                self.notes.append(f"{uart.instance}_TX 走 {channel}（RM0008 表 14-4 的固定映射）")

    def _add_uart_msp(self, uart: UartSpec, tx: Pin, rx: Pin) -> None:
        if any(f"huart->Instance == {uart.instance}" in block for block in self.msp_uart):
            return
        body = self._uart_msp_body(uart, tx, rx)
        self.msp_uart.append("\n".join([f"  if (huart->Instance == {uart.instance})", "  {"] + body + ["  }"]))

    def _uart_msp_body(
        self,
        uart: UartSpec,
        tx: Pin,
        rx: Pin,
        declare: bool = True,
        link_expr: str = "huart",
    ) -> list[str]:
        """串口时钟 + 引脚复用 + DMA 的语句行（四条空格缩进）。"""
        lines = [f"    __HAL_RCC_{uart.instance}_CLK_ENABLE();"]
        for port in dict.fromkeys(pin.port for pin in (tx, rx)):
            lines.append(f"    __HAL_RCC_GPIO{port}_CLK_ENABLE();")
        if declare:
            lines.append("    GPIO_InitTypeDef GPIO_InitStruct = {0};")
        lines.append("")
        for port in dict.fromkeys(pin.port for pin in (tx, rx)):
            pins = [p for p in (tx, rx) if p.port == port]
            lines.append("    GPIO_InitStruct.Pin = " + " | ".join(p.pin_macro for p in pins) + ";")
            lines.append("    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;")
            lines.append("    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;")
            lines.append(f"    HAL_GPIO_Init(GPIO{port}, &GPIO_InitStruct);")
        if uart.dma:
            channel = dma_channel(uart.instance, "TX")
            if channel:
                name = f"hdma_{uart.instance.lower()}_tx"
                lines.extend(
                    [
                        "",
                        f"    {name}.Instance = {channel};",
                        f"    {name}.Init.Direction = DMA_MEMORY_TO_PERIPH;",
                        f"    {name}.Init.PeriphInc = DMA_PINC_DISABLE;",
                        f"    {name}.Init.MemInc = DMA_MINC_ENABLE;",
                        f"    {name}.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;",
                        f"    {name}.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;",
                        f"    {name}.Init.Mode = DMA_NORMAL;",
                        f"    {name}.Init.Priority = DMA_PRIORITY_LOW;",
                        f"    if (HAL_DMA_Init(&{name}) != HAL_OK)",
                        "    {",
                        "      Error_Handler();",
                        "    }",
                        f"    __HAL_LINKDMA({link_expr}, hdmatx, {name});",
                    ]
                )
        return lines

    # ---------------------------------------------- ADC / I2C
    def add_adc(self, adc: AdcSpec) -> None:
        channel = adc_channel_of_pin(adc.pin)
        self.modules.update({"adc", "dma"})
        self.handle_decls.append("ADC_HandleTypeDef hadc1;")
        self.handle_decls.append("DMA_HandleTypeDef hdma_adc1;")
        self.init_protos.append("static void MX_ADC1_Init(void);")
        self.main_init_calls.append("  MX_ADC1_Init();")
        lines = [
            "static void MX_ADC1_Init(void)",
            "{",
            "  ADC_ChannelConfTypeDef sConfig = {0};",
        ]
        if self.inline_msp:
            lines.append("  GPIO_InitTypeDef GPIO_InitStruct = {0};")
        lines.append("")
        if self.inline_msp:
            lines.extend(line[2:] for line in self._adc_msp_body(adc, channel, declare=False, link="&hadc1"))
            lines.append("")
        lines.extend(
            [
                "  hadc1.Instance = ADC1;",
                "  hadc1.Init.ScanConvMode = ADC_SCAN_DISABLE;",
                "  hadc1.Init.ContinuousConvMode = ENABLE;",
                "  hadc1.Init.DiscontinuousConvMode = DISABLE;",
                "  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;",
                "  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;",
                "  hadc1.Init.NbrOfConversion = 1;",
                "  if (HAL_ADC_Init(&hadc1) != HAL_OK)",
                "  {",
                "    Error_Handler();",
                "  }",
                "",
                f"  sConfig.Channel = ADC_CHANNEL_{channel};",
                "  sConfig.Rank = ADC_REGULAR_RANK_1;",
                "  sConfig.SamplingTime = ADC_SAMPLETIME_239CYCLES_5;",
                "  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)",
                "  {",
                "    Error_Handler();",
                "  }",
                "}",
            ]
        )
        self.init_funcs.append("\n".join(lines))
        if not self.inline_msp:
            body = self._adc_msp_body(adc, channel)
            self.msp_adc.append("\n".join(["  if (adcHandle->Instance == ADC1)", "  {"] + body + ["  }"]))
        self.notes.append(f"ADC1_IN{channel} ← {adc.pin.name}；DMA1_Channel1 循环搬运")

    def _adc_msp_body(self, adc: AdcSpec, channel: int, declare: bool = True, link: str = "adcHandle") -> list[str]:
        lines = [
            "    __HAL_RCC_ADC1_CLK_ENABLE();",
            f"    __HAL_RCC_GPIO{adc.pin.port}_CLK_ENABLE();",
        ]
        if declare:
            lines.append("    GPIO_InitTypeDef GPIO_InitStruct = {0};")
        lines.extend(
            [
                "",
                f"    GPIO_InitStruct.Pin = {adc.pin.pin_macro};",
                "    GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;",
                "    GPIO_InitStruct.Pull = GPIO_NOPULL;",
                f"    HAL_GPIO_Init({adc.pin.gpio}, &GPIO_InitStruct);",
                "",
                "    __HAL_RCC_DMA1_CLK_ENABLE();",
                "    hdma_adc1.Instance = DMA1_Channel1;",
                "    hdma_adc1.Init.Direction = DMA_PERIPH_TO_MEMORY;",
                "    hdma_adc1.Init.PeriphInc = DMA_PINC_DISABLE;",
                "    hdma_adc1.Init.MemInc = DMA_MINC_ENABLE;",
                "    hdma_adc1.Init.PeriphDataAlignment = DMA_PDATAALIGN_HALFWORD;",
                "    hdma_adc1.Init.MemDataAlignment = DMA_MDATAALIGN_HALFWORD;",
                "    hdma_adc1.Init.Mode = DMA_CIRCULAR;",
                "    hdma_adc1.Init.Priority = DMA_PRIORITY_LOW;",
                "    if (HAL_DMA_Init(&hdma_adc1) != HAL_OK)",
                "    {",
                "      Error_Handler();",
                "    }",
                f"    __HAL_LINKDMA({link}, DMA_Handle, hdma_adc1);",
            ]
        )
        return lines

    def add_i2c(self, oled: OledSpec) -> None:
        instance = oled.instance
        pins = I2C_PINS[instance]
        handle = f"hi2c{instance[3:]}"
        self.modules.add("i2c")
        self.handle_decls.append(f"I2C_HandleTypeDef {handle};")
        self.init_protos.append(f"static void MX_{instance}_Init(void);")
        self.main_init_calls.append(f"  MX_{instance}_Init();")
        lines = [
            f"static void MX_{instance}_Init(void)",
            "{",
        ]
        if self.inline_msp:
            lines.append("  GPIO_InitTypeDef GPIO_InitStruct = {0};")
        lines.append("")
        if self.inline_msp:
            lines.extend(line[2:] for line in self._i2c_msp_body(oled, declare=False))
            lines.append("")
        lines.extend(
            [
                f"  {handle}.Instance = {instance};",
                f"  {handle}.Init.ClockSpeed = {400000 if oled.address else 100000};",
                f"  {handle}.Init.DutyCycle = I2C_DUTYCYCLE_2;",
                f"  {handle}.Init.OwnAddress1 = 0;",
                f"  {handle}.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;",
                f"  {handle}.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;",
                f"  {handle}.Init.OwnAddress2 = 0;",
                f"  {handle}.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;",
                f"  {handle}.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;",
                f"  if (HAL_I2C_Init(&{handle}) != HAL_OK)",
                "  {",
                "    Error_Handler();",
                "  }",
                "}",
            ]
        )
        self.init_funcs.append("\n".join(lines))
        if not self.inline_msp:
            body = self._i2c_msp_body(oled)
            self.msp_i2c.append("\n".join([f"  if (hi2c->Instance == {instance})", "  {"] + body + ["  }"]))
        self.notes.append(
            f"{instance}：SCL={pins['SCL'].name}、SDA={pins['SDA'].name}，400 kHz（OLED 地址 0x{oled.address:02X}）"
        )

    def _i2c_msp_body(self, oled: OledSpec, declare: bool = True) -> list[str]:
        instance = oled.instance
        scl = I2C_PINS[instance]["SCL"]
        sda = I2C_PINS[instance]["SDA"]
        ports = list(dict.fromkeys([scl.port, sda.port]))
        lines = [f"    __HAL_RCC_{instance}_CLK_ENABLE();"]
        for port in ports:
            lines.append(f"    __HAL_RCC_GPIO{port}_CLK_ENABLE();")
        if declare:
            lines.append("    GPIO_InitTypeDef GPIO_InitStruct = {0};")
        lines.append("")
        for port in ports:
            group = [pin for pin in (scl, sda) if pin.port == port]
            lines.append("    GPIO_InitStruct.Pin = " + " | ".join(pin.pin_macro for pin in group) + ";")
            lines.append("    GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;")
            lines.append("    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;")
            lines.append(f"    HAL_GPIO_Init(GPIO{port}, &GPIO_InitStruct);")
        return lines

    # ---------------------------------------------- GPIO / EXTI
    def add_gpio(self, gpio: GpioSpec) -> None:
        if gpio.mode in ("it_rising", "it_falling"):
            self.modules.add("exti")
            irq = _exti_irq(gpio.pin)
            self.nvic.append(f"  HAL_NVIC_SetPriority({irq}, 2, 0);\n  HAL_NVIC_EnableIRQ({irq});")
            if "static void MX_NVIC_Init(void);" not in self.init_protos:
                self.init_protos.append("static void MX_NVIC_Init(void);")
                self.main_init_calls.append("  MX_NVIC_Init();")
            self.it_handlers.append(
                f"void {irq[:-4]}Handler(void)\n{{\n  HAL_GPIO_EXTI_IRQHandler({gpio.pin.pin_macro});\n}}"
            )
            self.callbacks.append(
                "void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)\n"
                "{\n"
                f"  if (GPIO_Pin == {gpio.pin.pin_macro})\n"
                "  {\n    /* TODO: 在这里处理外部事件 */\n  }\n}"
            )

    # ---------------------------------------------- 主循环
    def add_loop(self, loop: LoopSpec) -> None:
        if loop.kind == "toggle" and loop.pin:
            self.loop_lines.append(f"    HAL_GPIO_TogglePin({loop.pin.gpio}, {loop.pin.pin_macro});")
            self.loop_lines.append(f"    HAL_Delay({loop.interval_ms});")
        elif loop.kind == "print" and loop.message and self.printf_handle:
            message = loop.message.replace('"', '\\"')
            uart = next((item for item in self.spec.uarts if f"huart{item.instance[5:]}" == self.printf_handle), None)
            if uart is not None and uart.dma:
                self.loop_lines.append(f'    static const uint8_t msg[] = "{message}\\r\\n";')
                self.loop_lines.append(
                    f"    HAL_UART_Transmit_DMA(&{self.printf_handle}, (uint8_t *)msg, sizeof(msg) - 1);"
                )
            else:
                self.loop_lines.append(f'    printf("{message}\\r\\n");')
            self.loop_lines.append(f"    HAL_Delay({loop.interval_ms});")

    # ---------------------------------------------- 分段输出（CubeMX 增量修改用）
    def _msp_tim_func(self) -> str:
        if not self.msp_tim:
            return ""
        return "void HAL_TIM_Base_MspInit(TIM_HandleTypeDef *htim)\n{\n" + "\n\n".join(self.msp_tim) + "\n}"

    def _msp_uart_func(self) -> str:
        if not self.msp_uart:
            return ""
        return "void HAL_UART_MspInit(UART_HandleTypeDef *huart)\n{\n" + "\n\n".join(self.msp_uart) + "\n}"

    def _msp_adc_func(self) -> str:
        if not self.msp_adc:
            return ""
        return "void HAL_ADC_MspInit(ADC_HandleTypeDef *adcHandle)\n{\n" + "\n\n".join(self.msp_adc) + "\n}"

    def _msp_i2c_func(self) -> str:
        if not self.msp_i2c:
            return ""
        return "void HAL_I2C_MspInit(I2C_HandleTypeDef *hi2c)\n{\n" + "\n\n".join(self.msp_i2c) + "\n}"

    def _nvic_func(self) -> str:
        if not self.nvic:
            return ""
        return "static void MX_NVIC_Init(void)\n{\n" + "\n".join(self.nvic) + "\n}"

    def _retarget_block(self) -> str:
        if not self.printf_handle:
            return ""
        return (
            "/* printf 重定向：AC6 + microlib 下重写 fputc 即可把字符送到串口 */\n"
            "int fputc(int ch, FILE *f)\n{\n"
            "  (void)f;\n"
            f"  HAL_UART_Transmit(&{self.printf_handle}, (uint8_t *)&ch, 1, HAL_MAX_DELAY);\n"
            "  return ch;\n"
            "}"
        )

    def sections(self) -> dict[str, str]:
        """把生成内容拆成可插入 CubeMX USER CODE 的几块。"""
        prototypes = [p for p in dict.fromkeys(self.init_protos) if p != "static void MX_GPIO_Init(void);"]
        return {
            "handles": "\n".join(dict.fromkeys(self.handle_decls)),
            "prototypes": "\n".join(prototypes),
            "init_calls": "\n".join(dict.fromkeys(self.main_init_calls)),
            "init_funcs": "\n\n".join(self.init_funcs),
            "msp": "\n\n".join(
                item
                for item in (
                    self._msp_tim_func(),
                    self._msp_uart_func(),
                    self._msp_adc_func(),
                    self._msp_i2c_func(),
                )
                if item
            ),
            "nvic": self._nvic_func(),
            "start": "\n\n".join(self.start_calls),
            "loop": "\n".join(self.loop_lines),
            "callbacks": "\n\n".join(self.callbacks),
            "it_handlers": "\n\n".join(self.it_handlers),
            "retarget": self._retarget_block(),
            "printf": "1" if self.printf_handle else "",
        }

    # ---------------------------------------------- 输出
    def render(self) -> dict[str, str]:
        spec = self.spec
        gpio_init = self._gpio_init()
        parts = [
            item
            for item in (
                self._msp_tim_func(),
                self._msp_uart_func(),
                self._msp_adc_func(),
                self._msp_i2c_func(),
            )
            if item
        ]
        msp = ("\n\n".join(parts) + "\n\n" if parts else "") + (
            "void HAL_MspInit(void)\n{\n"
            "  __HAL_RCC_AFIO_CLK_ENABLE();\n"
            "  __HAL_RCC_PWR_CLK_ENABLE();\n"
            "}\n"
        )

        nvic_func = self._nvic_func() + "\n" if self.nvic else ""

        main_init_calls = "\n".join(dict.fromkeys(self.main_init_calls))
        start_calls = "\n\n".join(self.start_calls) or "  /* 没有需要启动的外设 */"
        loop = "\n".join(self.loop_lines) or "    /* 主循环暂无周期性动作 */"
        callbacks = "\n\n".join(self.callbacks)
        retarget = ("\n\n" + self._retarget_block()) if self.printf_handle else ""

        main_c = f"""/* main.c —— 由 stm32_rag 代码生成器生成（{spec.title}）
 *
 * 生成依据：RM0008（寄存器/复用）· UM1850（HAL API）· HAL 源码（参数语义）
 * 需要手工补充的逻辑写在 USER CODE 段内。
 */
#include "main.h"
{chr(10).join('#include "' + inc + '"' for inc in self.comp_includes)}
{chr(10).join(line for line in ['#include <stdio.h>'] if self.printf_handle)}

/* 句柄 */
{chr(10).join(dict.fromkeys(self.handle_decls))}

/* 私有函数声明 */
static void SystemClock_Config(void);
{chr(10).join(self.init_protos)}

int main(void)
{{
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
{main_init_calls}

{start_calls}

  while (1)
  {{
{loop}
  }}
}}

/* ------------------------------ 外设初始化 ------------------------------ */
{gpio_init}

{chr(10).join(self.init_funcs)}

{nvic_func}
/* ------------------------------ MSP：时钟与引脚 ------------------------------ */
{msp}
/* ------------------------------ 时钟树 ------------------------------ */
{_clock_body(spec.sysclk, spec.hse)[0]}

/* ------------------------------ 错误处理 ------------------------------ */
void Error_Handler(void)
{{
  __disable_irq();
  while (1)
  {{
  }}
}}
{chr(10) + callbacks if callbacks else ''}{retarget}
"""

        main_h = (
            "/* main.h —— 由 stm32_rag 代码生成器生成 */\n"
            "#ifndef __MAIN_H\n"
            "#define __MAIN_H\n\n"
            "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n"
            '#include "stm32f1xx_hal.h"\n'
            + ("#include <stdio.h>\n" if self.printf_handle else "")
            + "\n"
            + "\n".join(f"extern {decl}" for decl in dict.fromkeys(self.handle_decls))
            + "\n\nvoid Error_Handler(void);\n\n"
            "#ifdef __cplusplus\n}\n#endif\n\n#endif /* __MAIN_H */\n"
        )

        it_c = (
            "/* stm32f1xx_it.c —— 由 stm32_rag 代码生成器生成 */\n"
            '#include "main.h"\n'
            '#include "stm32f1xx_it.h"\n\n'
            "void NMI_Handler(void)\n{\n}\n\n"
            "void HardFault_Handler(void)\n{\n  while (1)\n  {\n  }\n}\n\n"
            "void MemManage_Handler(void)\n{\n  while (1)\n  {\n  }\n}\n\n"
            "void BusFault_Handler(void)\n{\n  while (1)\n  {\n  }\n}\n\n"
            "void UsageFault_Handler(void)\n{\n  while (1)\n  {\n  }\n}\n\n"
            "void SVC_Handler(void)\n{\n}\n\n"
            "void DebugMon_Handler(void)\n{\n}\n\n"
            "void PendSV_Handler(void)\n{\n}\n\n"
            "void SysTick_Handler(void)\n{\n  HAL_IncTick();\n}\n"
            + ("\n" + "\n\n".join(self.it_handlers) + "\n" if self.it_handlers else "")
        )

        it_h = (
            "/* stm32f1xx_it.h —— 由 stm32_rag 代码生成器生成 */\n"
            "#ifndef __STM32F1xx_IT_H\n#define __STM32F1xx_IT_H\n\n"
            "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n"
            "void NMI_Handler(void);\nvoid HardFault_Handler(void);\nvoid MemManage_Handler(void);\n"
            "void BusFault_Handler(void);\nvoid UsageFault_Handler(void);\nvoid SVC_Handler(void);\n"
            "void DebugMon_Handler(void);\nvoid PendSV_Handler(void);\nvoid SysTick_Handler(void);\n\n"
            "#ifdef __cplusplus\n}\n#endif\n\n#endif /* __STM32F1xx_IT_H */\n"
        )

        return {
            "Core/Inc/main.h": main_h,
            "Core/Src/main.c": main_c,
            "Core/Inc/stm32f1xx_it.h": it_h,
            "Core/Src/stm32f1xx_it.c": it_c,
        }

    def _gpio_init(self, gpios: list[GpioSpec] | None = None, add_loop_pin: bool = True, name: str = "MX_GPIO_Init") -> str:
        lines = [f"static void {name}(void)", "{", "  GPIO_InitTypeDef GPIO_InitStruct = {0};"]
        specs = list(self.spec.gpios if gpios is None else gpios)
        # 主循环要翻转的引脚如果没显式配置，补一个推挽输出
        loop_pin = (
            self.spec.loop.pin
            if add_loop_pin and self.spec.loop and self.spec.loop.kind == "toggle"
            else None
        )
        if loop_pin and not any(item.pin == loop_pin for item in specs):
            specs.append(GpioSpec(pin=loop_pin, mode="output_pp", initial="high", label="主循环翻转"))
        if not specs:
            lines.append("  /* 没有独立 GPIO 需求 */")
        else:
            for port in dict.fromkeys(item.pin.port for item in specs):
                lines.append(f"  __HAL_RCC_GPIO{port}_CLK_ENABLE();")
            for item in specs:
                mode = {
                    "output_pp": "GPIO_MODE_OUTPUT_PP",
                    "output_od": "GPIO_MODE_OUTPUT_OD",
                    "input": "GPIO_MODE_INPUT",
                    "input_pullup": "GPIO_MODE_INPUT",
                    "analog": "GPIO_MODE_ANALOG",
                    "it_rising": "GPIO_MODE_IT_RISING",
                    "it_falling": "GPIO_MODE_IT_FALLING",
                }.get(item.mode, "GPIO_MODE_OUTPUT_PP")
                pull = "GPIO_PULLUP" if item.mode == "input_pullup" else "GPIO_NOPULL"
                speed = "GPIO_SPEED_FREQ_HIGH" if item.mode.startswith("it_") else "GPIO_SPEED_FREQ_LOW"
                lines.extend(
                    [
                        "",
                        f"  /* {item.pin.name}{(' · ' + item.label) if item.label else ''} */",
                        f"  GPIO_InitStruct.Pin = {item.pin.pin_macro};",
                        f"  GPIO_InitStruct.Mode = {mode};",
                        f"  GPIO_InitStruct.Pull = {pull};",
                        f"  GPIO_InitStruct.Speed = {speed};",
                        f"  HAL_GPIO_Init({item.pin.gpio}, &GPIO_InitStruct);",
                    ]
                )
                if item.initial in ("high", "low"):
                    state = "GPIO_PIN_SET" if item.initial == "high" else "GPIO_PIN_RESET"
                    lines.append(f"  HAL_GPIO_WritePin({item.pin.gpio}, {item.pin.pin_macro}, {state});")
        lines.append("}")
        return "\n".join(lines)


def _exti_irq(pin: Pin) -> str:
    if pin.number >= 10:
        return "EXTI15_10_IRQn"
    if pin.number >= 5:
        return "EXTI9_5_IRQn"
    return "EXTI4_IRQn"


# ---------------------------------------------------------------- 对外入口

def plan_application(spec: AppSpec, inline_msp: bool = False) -> CodePlan:
    """校验 -> 生成 4 个文件；校验不过时不产生任何代码。

    inline_msp=True 用于修改已有 CubeMX 工程：不生成 HAL_xxx_MspInit，
    改把时钟/引脚配置写进各自的 init 函数，避免与 CubeMX 的 msp.c 重复定义。
    """
    errors, warnings, checks = validate(spec)
    if errors:
        return CodePlan(ok=False, errors=errors, warnings=warnings, checks=checks)
    if inline_msp and (spec.keys or spec.buzzer or spec.serial_cmd or spec.adc or spec.oled or spec.sched):
        return CodePlan(
            ok=False,
            errors=["CubeMX 增量模式暂不支持应用层组件（按键/蜂鸣器/串口命令行/ADC/OLED/调度器）"],
            warnings=warnings,
            checks=checks,
        )

    builder = _Builder(spec, inline_msp=inline_msp)
    for pwm in spec.pwm:
        builder.add_pwm(pwm)
    for timer in spec.timers:
        builder.add_timer(timer)
    for uart in spec.uarts:
        builder.add_uart(uart)
    for gpio in spec.gpios:
        builder.add_gpio(gpio)
    if spec.loop:
        builder.add_loop(spec.loop)

    # ---- 应用层组件：先补外设初始化，再插组件初始化，顺序即 main() 里的执行顺序
    if spec.buzzer is not None:
        builder.add_buzzer(spec.buzzer)
        builder.comp_includes.append("buzzer.h")
        builder.main_init_calls.append("  Buzzer_Init();")
        builder.loop_lines.append("    Buzzer_Task();")
    if spec.adc is not None:
        builder.add_adc(spec.adc)
        builder.comp_includes.append("adc_knob.h")
        builder.main_init_calls.append("  Adc_Init();")
        builder.loop_lines.append("    Adc_Task();")
    if spec.oled is not None:
        builder.add_i2c(spec.oled)
        builder.comp_includes.append("oled_ssd1306.h")
        builder.main_init_calls.append("  OLED_Init();")
        builder.loop_lines.append("    /* OLED 每 100 ms 刷一帧，避免 I2C 占用主循环 */")
        builder.loop_lines.append("    {")
        builder.loop_lines.append("      static uint32_t oled_tick;")
        builder.loop_lines.append("      if ((HAL_GetTick() - oled_tick) >= 100U)")
        builder.loop_lines.append("      {")
        builder.loop_lines.append("        oled_tick = HAL_GetTick();")
        builder.loop_lines.append('        OLED_ShowString(0, 0, "STM32 ready");')
        if spec.adc is not None:
            builder.loop_lines.append('        OLED_Printf(0, 16, "ADC: %3u%%", Adc_GetPercent());')
            builder.loop_lines.append("        OLED_ProgressBar(0, 40, 100, Adc_GetPercent());")
        if spec.keys is not None:
            builder.loop_lines.append('        OLED_Printf(0, 32, "KEY: %02X", Key_Scan());')
        if spec.serial_cmd is not None:
            builder.loop_lines.append('        OLED_Printf(0, 48, "CMD: %s", "serial");')
        builder.loop_lines.append("      }")
        builder.loop_lines.append("    }")
    if spec.serial_cmd is not None:
        instance = spec.serial_cmd.instance
        irq = f"{instance}_IRQn"
        handle = f"huart{instance[5:]}"
        # NVIC 必须先于 SerialCmd_Init（里面会开 HAL_UART_Receive_IT）
        builder.nvic.append(
            f"  /* {instance} 接收中断 */\n"
            f"  HAL_NVIC_SetPriority({irq}, 0, 0);\n  HAL_NVIC_EnableIRQ({irq});"
        )
        if "static void MX_NVIC_Init(void);" not in builder.init_protos:
            builder.init_protos.append("static void MX_NVIC_Init(void);")
            builder.main_init_calls.append("  MX_NVIC_Init();")
        builder.comp_includes.append("serial_cmd.h")
        builder.main_init_calls.append("  SerialCmd_Init();")
        builder.loop_lines.append("    SerialCmd_Task();")
        builder.it_handlers.append(
            f"/* {instance} 接收中断（命令行） */\n"
            f"void {irq[:-4]}Handler(void)\n{{\n  HAL_UART_IRQHandler(&{handle});\n}}"
        )
    if spec.keys is not None:
        builder.comp_includes.append("key.h")
        builder.main_init_calls.append("  Key_Init();")
        builder.loop_lines.append("    (void)Key_Scan();")
        if spec.buzzer is not None:
            # 按键 + 蜂鸣器：任何键按下响一声 C5，演示组件协作
            builder.loop_lines.append("    {")
            builder.loop_lines.append("      static uint8_t last_keys;")
            builder.loop_lines.append("      uint8_t now_keys = Key_Scan();")
            builder.loop_lines.append("      if (now_keys != last_keys)")
            builder.loop_lines.append("      {")
            builder.loop_lines.append("        last_keys = now_keys;")
            builder.loop_lines.append("        if (now_keys != 0U)")
            builder.loop_lines.append("        {")
            builder.loop_lines.append("          Buzzer_Tone(NOTE_C5, 60);")
            builder.loop_lines.append("        }")
            builder.loop_lines.append("      }")
            builder.loop_lines.append("    }")
    if spec.sched:
        builder.comp_includes.append("app_sched.h")
        builder.main_init_calls.append("  Sched_Init();")
        builder.loop_lines.append("    Sched_Run();")

    builder.finalize()

    files = builder.render()

    # ---- 应用层组件的 .c/.h 直接落盘
    from .components import adc_files, buzzer_files, key_files, oled_files, sched_files, serial_cmd_files

    extra: list = []
    if spec.sched:
        extra.extend(sched_files())
    if spec.keys is not None:
        extra.extend(
            key_files(
                [(pin.gpio, pin.pin_macro) for pin in spec.keys.pins],
                debounce_ms=spec.keys.debounce_ms,
                active_low=spec.keys.active_low,
            )
        )
    if spec.buzzer is not None:
        picked = timer_for_pin(spec.buzzer.pin, spec.device, prefer=spec.buzzer.timer or None)
        timer, channel = (
            (spec.buzzer.timer, spec.buzzer.channel)
            if spec.buzzer.timer and spec.buzzer.channel
            else picked  # type: ignore[misc]
        )
        tim_clk, _ = timer_clock(timer, spec.sysclk)
        extra.extend(buzzer_files(timer, tim_clk, channel))
    if spec.serial_cmd is not None:
        uart = next((item for item in spec.uarts if item.instance == spec.serial_cmd.instance), None)
        baud = uart.baud if uart else 115200
        cmd_files, _, _ = serial_cmd_files(spec.serial_cmd.instance, baud)
        extra.extend(cmd_files)
    if spec.adc is not None:
        channel = adc_channel_of_pin(spec.adc.pin)
        if channel is not None:
            extra.extend(adc_files("ADC1", channel, spec.adc.samples, spec.adc.label))
    if spec.oled is not None:
        extra.extend(
            oled_files(spec.oled.instance, spec.oled.address, spec.oled.width, spec.oled.height)
        )
    for item in extra:
        files[item.path] = item.content
    modules = sorted(builder.modules)
    # validate() 与 builder 都会记录依据，去重后展示更干净
    merged_checks = list(dict.fromkeys(checks + builder.notes))
    plan = CodePlan(
        ok=True,
        errors=[],
        warnings=warnings,
        wiring=_wiring(spec),
        checks=merged_checks,
        modules=modules,
        files=files,
        main_c=files["Core/Src/main.c"],
        sections=builder.sections(),
    )
    return plan


def _wiring(spec: AppSpec) -> list[str]:
    lines: list[str] = []
    for pwm in spec.pwm:
        timer = pwm.timer or (timer_for_pin(pwm.pin, spec.device) or ("?", "?"))[0]
        channel = pwm.channel or (timer_for_pin(pwm.pin, spec.device) or ("?", "?"))[1]
        lines.append(f"{pwm.pin.name} → {timer}_{channel}（PWM {_fmt_hz(pwm.freq_hz)}）→ 接示波器或 LED")
    for uart in spec.uarts:
        tx = uart.tx or USART_PINS[uart.instance]["TX"][0]
        rx = uart.rx or USART_PINS[uart.instance]["RX"][0]
        lines.append(f"{uart.instance}：{tx.name}=TX、{rx.name}=RX（{uart.baud} 8N1）→ 接 USB-TTL")
    for gpio in spec.gpios:
        lines.append(f"{gpio.pin.name}：{gpio.mode}" + (f"，初始 {gpio.initial}" if gpio.initial else ""))
    for timer in spec.timers:
        if timer.toggle_pin:
            lines.append(f"{timer.toggle_pin.name}：由 {timer.timer} 更新中断翻转（{_fmt_hz(timer.freq_hz)}）")
    if spec.keys is not None:
        lines.append(
            "按键："
            + "、".join(pin.name for pin in spec.keys.pins)
            + " → 另一端接 GND（内部上拉，低电平有效）"
        )
    if spec.buzzer is not None:
        lines.append(f"蜂鸣器 {spec.buzzer.pin.name}：PWM 输出 → 三极管基极，无源蜂鸣器两端接蜂鸣器")
    if spec.adc is not None:
        lines.append(f"电位器：两端接 3.3V / GND，中间抽头 → {spec.adc.pin.name}")
    if spec.oled is not None and spec.oled.instance in I2C_PINS:
        i2c = I2C_PINS[spec.oled.instance]
        lines.append(
            f"OLED：VCC 3.3V、GND，SCL={i2c['SCL'].name}、SDA={i2c['SDA'].name}（模块自带 4.7k 上拉）"
        )
    if spec.loop and spec.loop.kind == "toggle" and spec.loop.pin:
        lines.append(f"{spec.loop.pin.name}：主循环每 {spec.loop.interval_ms} ms 翻转一次")
    return lines


def render_gpio_init(spec: AppSpec, gpios: list[GpioSpec], name: str = "MX_GPIO_Init") -> str:
    """单独渲染一个 GPIO 初始化函数（CubeMX 增量模式用）。"""
    builder = _Builder(spec)
    return builder._gpio_init(gpios, add_loop_pin=False, name=name)


def spec_to_dict(spec: AppSpec) -> dict:
    """序列化规格（供增量模式持久化多轮需求）。"""
    return {
        "device": spec.device.name,
        "sysclk": spec.sysclk,
        "hse": spec.hse,
        "title": spec.title,
        "pwm": [
            {
                "pin": item.pin.name,
                "timer": item.timer,
                "channel": item.channel,
                "freq_hz": item.freq_hz,
                "duty": item.duty,
            }
            for item in spec.pwm
        ],
        "gpio": [
            {
                "pin": item.pin.name,
                "mode": item.mode,
                "initial": item.initial,
                "label": item.label,
            }
            for item in spec.gpios
        ],
        "uart": [
            {
                "instance": item.instance,
                "baud": item.baud,
                "tx": item.tx.name if item.tx else "",
                "rx": item.rx.name if item.rx else "",
                "dma": item.dma,
                "printf": item.printf,
            }
            for item in spec.uarts
        ],
        "timers": [
            {
                "timer": item.timer,
                "freq_hz": item.freq_hz,
                "toggle_pin": item.toggle_pin.name if item.toggle_pin else "",
            }
            for item in spec.timers
        ],
        "loop": (
            {
                "kind": spec.loop.kind,
                "pin": spec.loop.pin.name if spec.loop.pin else "",
                "interval_ms": spec.loop.interval_ms,
                "message": spec.loop.message,
            }
            if spec.loop
            else None
        ),
        "keys": (
            {
                "pins": [pin.name for pin in spec.keys.pins],
                "active_low": spec.keys.active_low,
                "debounce_ms": spec.keys.debounce_ms,
            }
            if spec.keys
            else None
        ),
        "buzzer": (
            {"pin": spec.buzzer.pin.name, "timer": spec.buzzer.timer, "channel": spec.buzzer.channel}
            if spec.buzzer
            else None
        ),
        "serial_cmd": {"instance": spec.serial_cmd.instance} if spec.serial_cmd else None,
        "adc": (
            {"pin": spec.adc.pin.name, "samples": spec.adc.samples, "label": spec.adc.label}
            if spec.adc
            else None
        ),
        "oled": (
            {
                "instance": spec.oled.instance,
                "address": spec.oled.address,
                "width": spec.oled.width,
                "height": spec.oled.height,
            }
            if spec.oled
            else None
        ),
        "sched": spec.sched,
    }


def spec_from_payload(data: dict, device: str | None = None) -> AppSpec:
    """从 JSON 字典构造 AppSpec（与 spec_to_dict 互逆）。"""
    dev = resolve_device(data.get("device") or device or "STM32F103RB")
    spec = AppSpec(
        device=dev,
        sysclk=int(data.get("sysclk") or 72_000_000),
        hse=int(data.get("hse") or 8_000_000),
        title=str(data.get("title") or "app"),
    )
    for item in data.get("pwm") or []:
        pin = _pin_from(str(item.get("pin", "")))
        if pin is None:
            continue
        spec.pwm.append(
            PwmSpec(
                pin=pin,
                timer=str(item.get("timer") or ""),
                channel=str(item.get("channel") or ""),
                freq_hz=float(item.get("freq_hz") or 1000),
                duty=float(item.get("duty") if item.get("duty") is not None else 0.5),
            )
        )
    for item in data.get("gpio") or []:
        pin = _pin_from(str(item.get("pin", "")))
        if pin is None:
            continue
        spec.gpios.append(
            GpioSpec(
                pin=pin,
                mode=str(item.get("mode") or "output_pp"),
                initial=str(item.get("initial") or ""),
                label=str(item.get("label") or ""),
            )
        )
    for item in data.get("uart") or []:
        instance = str(item.get("instance") or "USART1").upper()
        if instance not in USART_PINS:
            continue
        tx = _pin_from(str(item.get("tx") or "")) or USART_PINS[instance]["TX"][0]
        rx = _pin_from(str(item.get("rx") or "")) or USART_PINS[instance]["RX"][0]
        spec.uarts.append(
            UartSpec(
                instance=instance,
                baud=int(item.get("baud") or 115200),
                tx=tx,
                rx=rx,
                dma=bool(item.get("dma")),
                printf=bool(item.get("printf", True)),
            )
        )
    for item in data.get("timers") or []:
        spec.timers.append(
            TimerSpec(
                timer=str(item.get("timer") or "TIM2").upper(),
                freq_hz=float(item.get("freq_hz") or 1000),
                toggle_pin=_pin_from(str(item.get("toggle_pin") or "")),
            )
        )
    loop = data.get("loop")
    if loop:
        spec.loop = LoopSpec(
            kind=str(loop.get("kind") or "none"),
            pin=_pin_from(str(loop.get("pin") or "")),
            interval_ms=int(loop.get("interval_ms") or 500),
            message=str(loop.get("message") or ""),
        )
    keys = data.get("keys")
    if keys is not None:
        pins_list = [_pin_from(str(item)) for item in keys.get("pins") or []]
        spec.keys = KeySpec(
            pins=[pin for pin in pins_list if pin is not None],
            active_low=bool(keys.get("active_low", True)),
            debounce_ms=int(keys.get("debounce_ms") or 20),
        )
    buzzer = data.get("buzzer")
    if buzzer is not None:
        pin = _pin_from(str(buzzer.get("pin") or ""))
        if pin is not None:
            spec.buzzer = BuzzerSpec(
                pin=pin,
                timer=str(buzzer.get("timer") or ""),
                channel=str(buzzer.get("channel") or ""),
            )
    cmd = data.get("serial_cmd")
    if cmd is not None:
        spec.serial_cmd = SerialCmdSpec(instance=str(cmd.get("instance") or "USART1").upper())
    adc = data.get("adc")
    if adc is not None:
        pin = _pin_from(str(adc.get("pin") or ""))
        if pin is not None:
            spec.adc = AdcSpec(
                pin=pin,
                samples=int(adc.get("samples") or 8),
                label=str(adc.get("label") or "旋钮"),
            )
    oled = data.get("oled")
    if oled is not None:
        spec.oled = OledSpec(
            instance=str(oled.get("instance") or "I2C1").upper(),
            address=int(oled.get("address") or 0x78),
            width=int(oled.get("width") or 128),
            height=int(oled.get("height") or 64),
        )
    spec.sched = bool(data.get("sched"))
    return spec


def merge_specs(old: AppSpec | None, new: AppSpec) -> AppSpec:
    """把新需求合并到已有规格上：同键覆盖，其余保留。"""
    if old is None:
        return new
    merged = AppSpec(device=new.device, sysclk=new.sysclk, hse=new.hse, title=new.title)

    new_uart_keys = {item.instance for item in new.uarts}
    merged.uarts = list(new.uarts) + [item for item in old.uarts if item.instance not in new_uart_keys]

    new_pwm_keys = {item.pin.name for item in new.pwm}
    merged.pwm = list(new.pwm) + [item for item in old.pwm if item.pin.name not in new_pwm_keys]

    new_timer_keys = {item.timer for item in new.timers}
    merged.timers = list(new.timers) + [item for item in old.timers if item.timer not in new_timer_keys]

    used_pins = (
        {item.pin.name for item in merged.pwm}
        | {item.pin.name for item in merged.timers if item.toggle_pin}
        | {pin.name for item in merged.uarts for pin in (item.tx, item.rx) if pin}
    )
    new_gpio_keys = {item.pin.name for item in new.gpios}
    merged.gpios = list(new.gpios) + [
        item
        for item in old.gpios
        if item.pin.name not in new_gpio_keys and item.pin.name not in used_pins
    ]
    merged.gpios = [item for item in merged.gpios if item.pin.name not in used_pins]

    if new.loop is not None and new.loop.kind != "none":
        merged.loop = new.loop
    elif old.loop is not None:
        merged.loop = old.loop

    merged.keys = new.keys or old.keys
    merged.buzzer = new.buzzer or old.buzzer
    merged.serial_cmd = new.serial_cmd or old.serial_cmd
    merged.adc = new.adc or old.adc
    merged.oled = new.oled or old.oled
    merged.sched = new.sched or old.sched
    return merged


def _pin_from(text: str) -> Pin | None:
    match = re.search(r"P([A-E])(\d{1,2})", text.upper())
    if not match:
        return None
    return Pin(match.group(1), int(match.group(2)))


def spec_summary(spec: AppSpec) -> dict:
    """给报告 / Agent 看的结构化摘要。"""
    return {
        "device": spec.device.name,
        "sysclk": spec.sysclk,
        "pwm": [
            {
                "pin": item.pin.name,
                "timer": item.timer or (timer_for_pin(item.pin, spec.device) or ("", ""))[0],
                "channel": item.channel or (timer_for_pin(item.pin, spec.device) or ("", ""))[1],
                "freq_hz": item.freq_hz,
                "duty": item.duty,
            }
            for item in spec.pwm
        ],
        "gpio": [{"pin": item.pin.name, "mode": item.mode, "initial": item.initial} for item in spec.gpios],
        "uart": [
            {
                "instance": item.instance,
                "baud": item.baud,
                "tx": (item.tx or USART_PINS[item.instance]["TX"][0]).name,
                "rx": (item.rx or USART_PINS[item.instance]["RX"][0]).name,
                "dma": item.dma,
            }
            for item in spec.uarts
        ],
        "timers": [{"timer": item.timer, "freq_hz": item.freq_hz} for item in spec.timers],
        "loop": (
            {"kind": spec.loop.kind, "pin": spec.loop.pin.name if spec.loop.pin else "", "interval_ms": spec.loop.interval_ms}
            if spec.loop
            else None
        ),
    }
