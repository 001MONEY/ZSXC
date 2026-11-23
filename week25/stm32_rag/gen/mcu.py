"""STM32F103 的器件 / 引脚复用 / DMA / 时钟静态数据与计算。

「代码生成工具」在下笔前先过一遍这里的表，能在写代码之前就把错误挡住：

- 引脚能不能复用成目标外设：PA8 只能是 TIM1_CH1，写成 TIM2_CH1 会直接报错；
- 定时器时钟：TIM1 挂在 APB2（72 MHz），TIM2~4 挂 APB1，APB1 分频不为 1 时
  定时器时钟 = PCLK1 × 2 = 72 MHz（RM0008 的 TIMxCLK 规则）；
- 目标频率能不能分频出来：PSC / ARR 都是 16 位，1 MHz 与 1 Hz 的预算完全不同；
- UART 的 BRR 误差：9600 与 115200 在 36 MHz 上的误差不一样，超过 3% 通不稳。

数据来源：RM0008 第 9 章（GPIO/AF 表）、第 14 章（DMA 请求映射）、
STM32F103x8/B 数据手册引脚定义表。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# ---------------------------------------------------------------- 器件

@dataclass(frozen=True)
class Device:
    """一个可以用来自动生成工程的 F103 型号。"""

    name: str
    define: str  # 工程里的器件宏（stm32f1xx.h 靠它选择寄存器定义）
    flash_kb: int
    ram_kb: int
    ports: str  # 封装引出的 GPIO 端口，如 "ABC"
    startup: str  # vendor 里的启动文件名

    @property
    def flash_bytes(self) -> int:
        return self.flash_kb * 1024

    @property
    def ram_bytes(self) -> int:
        return self.ram_kb * 1024


DEVICES: dict[str, Device] = {
    "STM32F103C8": Device("STM32F103C8", "STM32F103xB", 64, 20, "ABC", "startup_stm32f103xb.s"),
    "STM32F103RB": Device("STM32F103RB", "STM32F103xB", 128, 20, "ABC", "startup_stm32f103xb.s"),
    "STM32F103RC": Device("STM32F103RC", "STM32F103xE", 256, 48, "ABCD", "startup_stm32f103xe.s"),
    "STM32F103VE": Device("STM32F103VE", "STM32F103xE", 512, 64, "ABCDE", "startup_stm32f103xe.s"),
    "STM32F103ZE": Device("STM32F103ZE", "STM32F103xE", 512, 64, "ABCDE", "startup_stm32f103xe.s"),
}

# 完整型号 -> 前缀（STM32F103C8T6 / STM32F103RBT6 都能识别）
_CHIP_RE = re.compile(r"STM32F10[0-9]([A-Z])([0-9])[A-Z]?", re.IGNORECASE)

# 每个封装里可用的定时器
TIMERS_BY_DEVICE = {
    "STM32F103C8": ("TIM1", "TIM2", "TIM3", "TIM4"),
    "STM32F103RB": ("TIM1", "TIM2", "TIM3", "TIM4"),
    "STM32F103RC": ("TIM1", "TIM2", "TIM3", "TIM4", "TIM5", "TIM6", "TIM7", "TIM8"),
    "STM32F103VE": ("TIM1", "TIM2", "TIM3", "TIM4", "TIM5", "TIM6", "TIM7", "TIM8"),
    "STM32F103ZE": ("TIM1", "TIM2", "TIM3", "TIM4", "TIM5", "TIM6", "TIM7", "TIM8"),
}

# 封装真正引出的引脚（数据手册引脚定义表）。
# 只按端口字母判断会误判：LQFP48 的 C8 只引出 PC13~PC15，PC10/PC11 没焊出来，
# 所以 UART4（PC10/PC11）在 C8 上不可用；LQFP64 的 RB/RC 才有 PC0~PC15 + PD0~PD2。
PIN_LAYOUT: dict[str, dict[str, tuple[int, ...]]] = {
    "STM32F103C8": {"A": tuple(range(16)), "B": tuple(range(16)), "C": (13, 14, 15)},
    "STM32F103RB": {"A": tuple(range(16)), "B": tuple(range(16)), "C": tuple(range(16)), "D": (0, 1, 2)},
    "STM32F103RC": {"A": tuple(range(16)), "B": tuple(range(16)), "C": tuple(range(16)), "D": (0, 1, 2)},
    "STM32F103VE": {
        "A": tuple(range(16)),
        "B": tuple(range(16)),
        "C": tuple(range(16)),
        "D": tuple(range(16)),
        "E": tuple(range(16)),
    },
    "STM32F103ZE": {
        "A": tuple(range(16)),
        "B": tuple(range(16)),
        "C": tuple(range(16)),
        "D": tuple(range(16)),
        "E": tuple(range(16)),
    },
}


def device_pins(device: Device | None = None) -> list[Pin]:
    """封装实际引出的引脚；表里没有的器件按端口字母 × 16 兜底。"""
    device = device or DEVICES["STM32F103RB"]
    layout = PIN_LAYOUT.get(device.name) or {port: tuple(range(16)) for port in device.ports}
    return [Pin(port, number) for port, numbers in layout.items() for number in numbers]


def device_has_pin(pin: Pin, device: Device | None = None) -> bool:
    return pin.name in {item.name for item in device_pins(device)}


def device_usarts(device: Device | None = None) -> list[str]:
    """这颗片子能用的串口：TX/RX 引脚都在封装上才算有。"""
    names = {item.name for item in device_pins(device)}
    usable: list[str] = []
    for instance, roles in USART_PINS.items():
        if all(pin.name in names for pins in roles.values() for pin in pins):
            usable.append(instance)
    return usable



def resolve_device(name: str | None = None) -> Device:
    """把 `STM32F103C8T6` 之类的完整型号收敛到表中的器件。"""
    if not name:
        return DEVICES["STM32F103RB"]
    text = name.strip().upper()
    if text in DEVICES:
        return DEVICES[text]
    match = _CHIP_RE.search(text)
    if match:
        key = f"STM32F103{match.group(1)}{match.group(2)}"
        if key in DEVICES:
            return DEVICES[key]
    for key, device in DEVICES.items():
        if key in text:
            return device
    return DEVICES["STM32F103RB"]


# ---------------------------------------------------------------- 引脚

@dataclass(frozen=True)
class Pin:
    port: str  # "A"
    number: int

    @property
    def name(self) -> str:
        return f"P{self.port}{self.number}"

    @property
    def gpio(self) -> str:
        return f"GPIO{self.port}"

    @property
    def pin_macro(self) -> str:
        return f"GPIO_PIN_{self.number}"

    @property
    def rcc_enable(self) -> str:
        return f"__HAL_RCC_GPIO{self.port}_CLK_ENABLE()"

    def __str__(self) -> str:  # pragma: no cover - 调试用
        return self.name


def parse_pin(text: str) -> Pin | None:
    """从 `PA8` / `pa8` / `GPIOA_PIN_8` 里取引脚；取不到返回 None。"""
    if not text:
        return None
    match = re.search(r"P([A-E])\s*_?\s*(\d{1,2})\b", text.upper())
    if not match:
        return None
    return Pin(match.group(1), int(match.group(2)))


# ---------------------------------------------------------------- 定时器复用

# 无重映射时的默认复用（RM0008 表 9-2 ~ 9-4 / 定时器章节）
TIM_CHANNELS: dict[str, dict[str, tuple[str, ...]]] = {
    "TIM1": {"CH1": ("PA8",), "CH2": ("PA9",), "CH3": ("PA10",), "CH4": ("PA11",)},
    "TIM2": {"CH1": ("PA0",), "CH2": ("PA1",), "CH3": ("PA2",), "CH4": ("PA3",)},
    "TIM3": {"CH1": ("PA6",), "CH2": ("PA7",), "CH3": ("PB0",), "CH4": ("PB1",)},
    "TIM4": {"CH1": ("PB6",), "CH2": ("PB7",), "CH3": ("PB8",), "CH4": ("PB9",)},
}

ADVANCED_TIMERS = ("TIM1", "TIM8")
APB2_TIMERS = ("TIM1", "TIM8", "TIM9", "TIM10", "TIM11")


def pin_timer_options(pin: Pin, device: Device | None = None) -> list[tuple[str, str]]:
    """这个引脚能做哪些 (定时器, 通道)；按器件可用定时器过滤。"""
    available = TIMERS_BY_DEVICE.get((device or DEVICES["STM32F103RB"]).name, ("TIM1", "TIM2", "TIM3", "TIM4"))
    options: list[tuple[str, str]] = []
    for timer, channels in TIM_CHANNELS.items():
        if timer not in available:
            continue
        for channel, pins in channels.items():
            if pin.name in pins:
                options.append((timer, channel))
    return options


def timer_for_pin(pin: Pin, device: Device | None = None, prefer: str | None = None) -> tuple[str, str] | None:
    """给一个引脚挑定时器通道：优先用户指定，其次 TIM1（高级定时器精度高）。"""
    options = pin_timer_options(pin, device)
    if not options:
        return None
    if prefer:
        for timer, channel in options:
            if timer.upper() == prefer.upper():
                return timer, channel
    order = {name: i for i, name in enumerate(("TIM1", "TIM8", "TIM2", "TIM3", "TIM4"))}
    return sorted(options, key=lambda item: order.get(item[0], 99))[0]


def channel_of_pin(pin: Pin, timer: str) -> str | None:
    for channel, pins in TIM_CHANNELS.get(timer, {}).items():
        if pin.name in pins:
            return channel
    return None


# ---------------------------------------------------------------- USART

USART_PINS: dict[str, dict[str, tuple[Pin, ...]]] = {
    "USART1": {"TX": (Pin("A", 9),), "RX": (Pin("A", 10),)},
    "USART2": {"TX": (Pin("A", 2),), "RX": (Pin("A", 3),)},
    "USART3": {"TX": (Pin("B", 10),), "RX": (Pin("B", 11),)},
    "UART4": {"TX": (Pin("C", 10),), "RX": (Pin("C", 11),)},
    "UART5": {"TX": (Pin("C", 12),), "RX": (Pin("D", 2),)},
}
USART_ON_APB2 = ("USART1",)
USART_IRQ = {
    "USART1": "USART1_IRQn",
    "USART2": "USART2_IRQn",
    "USART3": "USART3_IRQn",
    "UART4": "UART4_IRQn",
    "UART5": "UART5_IRQn",
}
TIMER_IRQ = {
    "TIM1": "TIM1_UP_IRQn",
    "TIM2": "TIM2_IRQn",
    "TIM3": "TIM3_IRQn",
    "TIM4": "TIM4_IRQn",
}


def usart_of_pin(pin: Pin) -> tuple[str, str] | None:
    """PA9 -> ("USART1", "TX")。"""
    for instance, roles in USART_PINS.items():
        for role, pins in roles.items():
            if pin in pins:
                return instance, role
    return None


# ---------------------------------------------------------------- DMA

# F103 的 DMA 请求是硬连线的（RM0008 表 14-4）
DMA_MAP: dict[tuple[str, str], str] = {
    ("USART1", "TX"): "DMA1_Channel4",
    ("USART1", "RX"): "DMA1_Channel5",
    ("USART2", "TX"): "DMA1_Channel7",
    ("USART2", "RX"): "DMA1_Channel6",
    ("USART3", "TX"): "DMA1_Channel2",
    ("USART3", "RX"): "DMA1_Channel3",
    ("UART4", "TX"): "DMA2_Channel5",
    ("UART4", "RX"): "DMA2_Channel3",
    ("UART5", "TX"): "DMA2_Channel2",
    ("UART5", "RX"): "DMA2_Channel1",
    ("SPI1", "RX"): "DMA1_Channel2",
    ("SPI1", "TX"): "DMA1_Channel3",
    ("SPI2", "RX"): "DMA1_Channel4",
    ("SPI2", "TX"): "DMA1_Channel5",
    ("I2C1", "TX"): "DMA1_Channel6",
    ("I2C1", "RX"): "DMA1_Channel7",
    ("I2C2", "TX"): "DMA1_Channel4",
    ("I2C2", "RX"): "DMA1_Channel5",
    ("ADC1", "RX"): "DMA1_Channel1",
    ("TIM1", "CH1"): "DMA1_Channel2",
    ("TIM1", "CH2"): "DMA1_Channel3",
    ("TIM1", "CH3"): "DMA1_Channel6",
    ("TIM1", "CH4"): "DMA1_Channel4",
    ("TIM2", "CH1"): "DMA1_Channel5",
    ("TIM2", "CH2"): "DMA1_Channel7",
    ("TIM2", "CH3"): "DMA1_Channel1",
    ("TIM2", "CH4"): "DMA1_Channel7",
    ("TIM3", "CH1"): "DMA1_Channel6",
    ("TIM3", "CH2"): "DMA1_Channel3",
    ("TIM3", "CH3"): "DMA1_Channel2",
    ("TIM3", "CH4"): "DMA1_Channel3",
    ("TIM4", "CH1"): "DMA1_Channel1",
    ("TIM4", "CH2"): "DMA1_Channel4",
    ("TIM4", "CH3"): "DMA1_Channel5",
}


def dma_channel(peripheral: str, role: str) -> str | None:
    return DMA_MAP.get((peripheral, role))


def dma_irq(channel: str) -> str:
    """DMA1_Channel4 -> DMA1_Channel4_IRQn。"""
    return f"{channel}_IRQn"


# ---------------------------------------------------------------- 时钟

def timer_clock(timer: str, sysclk: int = 72_000_000, apb1_div: int = 2, apb2_div: int = 1) -> tuple[int, str]:
    """定时器输入时钟与它的来源说明。

    APB 预分频不为 1 时，定时器时钟是 PCLK 的 2 倍（RM0008 §7.2）。
    """
    if timer in APB2_TIMERS:
        pclk = sysclk // apb2_div
        clk = pclk * 2 if apb2_div != 1 else pclk
        return clk, f"{timer}CLK = PCLK2×{2 if apb2_div != 1 else 1} = {clk / 1e6:g} MHz（APB2）"
    pclk = sysclk // apb1_div
    clk = pclk * 2 if apb1_div != 1 else pclk
    return clk, f"{timer}CLK = PCLK1×{2 if apb1_div != 1 else 1} = {clk / 1e6:g} MHz（APB1）"


def peripheral_clock(peripheral: str, sysclk: int = 72_000_000, apb1_div: int = 2, apb2_div: int = 1) -> int:
    if peripheral in USART_ON_APB2 or peripheral in APB2_TIMERS:
        return sysclk // apb2_div
    return sysclk // apb1_div


# ---------------------------------------------------------------- 分频计算

@dataclass(frozen=True)
class Timing:
    """16 位 PSC/ARR 的定时参数。"""

    psc: int  # 寄存器值（实际分频 = psc + 1）
    arr: int  # 寄存器值（实际计数 = arr + 1）
    pulse: int  # CCR
    actual_hz: float
    resolution: int  # 一个周期有多少个计数，决定占空比精度

    def describe(self, target_hz: float) -> str:
        err = (self.actual_hz - target_hz) / target_hz * 100 if target_hz else 0.0
        return (
            f"PSC={self.psc}（分频 {self.psc + 1}）ARR={self.arr}（周期 {self.arr + 1}）"
            f"CCR={self.pulse} → 实际 {self.actual_hz / 1000:g} kHz，误差 {err:+.3f}%，"
            f"占空比分辨率 1/{self.resolution}"
        )


def pwm_timing(tim_clk: int, target_hz: float, duty: float = 0.5, max_arr: int = 65_535) -> Timing:
    """给目标频率算 PSC/ARR/CCR。

    先取 PSC=0（分频 1）吃满分辨率；一个周期超过 65536 个计数时再抬 PSC。
    频率超出能力范围时抛 ValueError，让调用方在生成代码前就报错。
    """
    if target_hz <= 0:
        raise ValueError("PWM 频率必须大于 0")
    if not 0.0 <= duty <= 1.0:
        raise ValueError("占空比必须在 0~1 之间")

    ticks = tim_clk / target_hz  # 一个输出周期需要的定时器计数
    if ticks < 2:
        raise ValueError(
            f"{target_hz / 1e6:g} MHz 超出能力：定时器时钟 {tim_clk / 1e6:g} MHz，"
            f"最高只能到 {tim_clk / 2 / 1e6:g} MHz（ARR 最小为 1）"
        )
    if ticks <= max_arr + 1:
        psc = 1
        arr = max(1, int(round(ticks)) - 1)
    else:
        psc = int(math.ceil(ticks / (max_arr + 1)))
        if psc > 65_536:
            raise ValueError(f"{target_hz:g} Hz 太低：PSC 需要 {psc}，超过 16 位上限")
        arr = max(1, int(round(ticks / psc)) - 1)

    period = psc * (arr + 1)
    actual = tim_clk / period
    pulse = int(round((arr + 1) * duty))
    pulse = max(0, min(arr + 1, pulse))
    return Timing(psc=psc - 1, arr=arr, pulse=pulse, actual_hz=actual, resolution=arr + 1)


def uart_brr(peripheral_hz: int, baud: int, oversampling: int = 16) -> tuple[int, float, float]:
    """返回 (BRR 值, 实际波特率, 误差百分比)。"""
    if baud <= 0:
        raise ValueError("波特率必须大于 0")
    if oversampling == 8:
        div = peripheral_hz / (8 * baud)
        mantissa = int(div)
        fraction = int(round((div - mantissa) * 8)) & 0x7
        brr = (mantissa << 4) | fraction
        actual = peripheral_hz / (8 * (mantissa + fraction / 8))
    else:
        div = peripheral_hz / (16 * baud)
        mantissa = int(div)
        fraction = int(round((div - mantissa) * 16)) & 0xF
        brr = (mantissa << 4) | fraction
        actual = peripheral_hz / (16 * (mantissa + fraction / 16))
    error = (actual - baud) / baud * 100
    return brr, actual, error


def gpio_clock_ok(pin: Pin, device: Device) -> bool:
    return pin.port in device.ports
