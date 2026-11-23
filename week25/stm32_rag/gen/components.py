"""可复用外设组件的 C 代码模板（不依赖外部库，只用 HAL + 标准 C）。

与 `codegen.py` 的分工：codegen 负责「外设初始化 + main.c 骨架」，
这里负责「应用层组件」——每个组件都是独立的 .c/.h，直接落盘到 `Core/{Src,Inc}`。

生成的组件一律遵循三条约定：
1. 初始化函数 `Xxx_Init()` 只做参数保存 / 启动，不阻塞；
2. 主循环里调用 `Xxx_Task()`（非阻塞），时间基准统一用 `HAL_GetTick()`（1 ms）；
3. 句柄由 main.c 定义，组件用 `extern` 引用，避免重复定义。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# 江协科技例程里的 8x16 字库（用户本地资料，默认从这里提取）
FONT_SRC = Path(
    os.getenv(
        "STM32_FONT_SRC",
        r"G:\BaiduNetdiskDownload\STM32入门教程资料\程序源码\STM32Project-有注释版"
        r"\4-1 OLED显示屏\Hardware\OLED_Font.h",
    )
)

_HEADER = "/* {name} —— 由 stm32_rag 生成 */\n#ifndef __{guard}_H\n#define __{guard}_H\n"


@dataclass
class ComponentFile:
    path: str
    content: str


def _h(name: str, guard: str, body: str, includes: str = "") -> ComponentFile:
    text = _HEADER.format(name=name, guard=guard)
    if includes:
        text += includes + "\n"
    text += body.rstrip() + "\n\n#endif\n"
    return ComponentFile(f"Core/Inc/{name}", text)


def _c(name: str, body: str, includes: list[str]) -> ComponentFile:
    head = f"/* {name} —— 由 stm32_rag 生成 */\n" + "".join(f'#include "{inc}"\n' for inc in includes)
    return ComponentFile(f"Core/Src/{name}", head + "\n" + body.rstrip() + "\n")


# ---------------------------------------------------------------- 调度器

def sched_files() -> list[ComponentFile]:
    h = _h(
        "app_sched.h",
        "APP_SCHED",
        """
#include <stdint.h>

#define SCHED_MAX_TASKS 8

void Sched_Init(void);
/* 注册周期任务：period_ms 必须 > 0；最多 8 个 */
void Sched_Add(void (*fn)(void), uint32_t period_ms);
/* 主循环里调用：扫描到期任务并执行，非阻塞 */
void Sched_Run(void);
""",
    )
    c = _c(
        "app_sched.c",
        """
typedef struct
{
  void (*fn)(void);
  uint32_t period;
  uint32_t last;
} SchedTask;

static SchedTask s_tasks[SCHED_MAX_TASKS];
static uint8_t s_count;

void Sched_Init(void)
{
  for (uint8_t i = 0; i < SCHED_MAX_TASKS; i++)
  {
    s_tasks[i].fn = 0;
    s_tasks[i].period = 0;
    s_tasks[i].last = 0;
  }
  s_count = 0;
}

void Sched_Add(void (*fn)(void), uint32_t period_ms)
{
  if ((fn == 0) || (period_ms == 0) || (s_count >= SCHED_MAX_TASKS))
  {
    return;
  }
  s_tasks[s_count].fn = fn;
  s_tasks[s_count].period = period_ms;
  s_tasks[s_count].last = HAL_GetTick();
  s_count++;
}

void Sched_Run(void)
{
  uint32_t now = HAL_GetTick();
  for (uint8_t i = 0; i < s_count; i++)
  {
    if ((now - s_tasks[i].last) >= s_tasks[i].period)
    {
      s_tasks[i].last += s_tasks[i].period;
      /* 任务执行超时导致落后太多时重新对齐，避免追帧风暴 */
      if ((now - s_tasks[i].last) >= s_tasks[i].period)
      {
        s_tasks[i].last = now;
      }
      s_tasks[i].fn();
    }
  }
}
""",
        ["main.h", "app_sched.h"],
    )
    return [h, c]


# ---------------------------------------------------------------- 按键

def key_files(pins: list[tuple[str, str]], debounce_ms: int = 20, active_low: bool = True) -> list[ComponentFile]:
    """pins: [(GPIOx, GPIO_PIN_n), ...]，序号 bit0 起。"""
    rows = ",\n".join(f"  {{{port}, {pin}}}" for port, pin in pins)
    count = len(pins)
    active = "0" if active_low else "1"
    h = _h(
        "key.h",
        "KEY",
        f"""
#include <stdint.h>

#define KEY_COUNT {count}

void Key_Init(void);
/* 扫描一次：返回消抖后的按下位图（bit0 = 第 1 个键），在主循环里调用 */
uint8_t Key_Scan(void);
/* 读取某一位（未消抖的原始电平），调试用 */
uint8_t Key_Raw(uint8_t index);
""",
    )
    c = _c(
        "key.c",
        f"""
typedef struct
{{
  GPIO_TypeDef *port;
  uint16_t pin;
}} KeyPin;

static const KeyPin s_keys[KEY_COUNT] = {{
{rows}
}};

static uint8_t s_stable;      /* 消抖后的状态：1 = 按下 */
static uint8_t s_raw_last;    /* 上一次采样 */
static uint32_t s_change_tick[KEY_COUNT];

static uint8_t Key_ReadRaw(uint8_t i)
{{
  return (HAL_GPIO_ReadPin(s_keys[i].port, s_keys[i].pin) == GPIO_PIN_RESET) ? 1U : 0U;
}}

void Key_Init(void)
{{
  GPIO_InitTypeDef GPIO_InitStruct = {{0}};
  __HAL_RCC_AFIO_CLK_ENABLE();
{"".join(f"  __HAL_RCC_{port}_CLK_ENABLE();" + chr(10) for port in dict.fromkeys(p for p, _ in pins))}  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  for (uint8_t i = 0; i < KEY_COUNT; i++)
  {{
    GPIO_InitStruct.Pin = s_keys[i].pin;
    HAL_GPIO_Init(s_keys[i].port, &GPIO_InitStruct);
  }}
  s_stable = 0;
  s_raw_last = 0;
  for (uint8_t i = 0; i < KEY_COUNT; i++)
  {{
    s_change_tick[i] = HAL_GetTick();
  }}
}}

uint8_t Key_Raw(uint8_t index)
{{
  if (index >= KEY_COUNT)
  {{
    return 0;
  }}
  return Key_ReadRaw(index);
}}

uint8_t Key_Scan(void)
{{
  uint8_t raw = 0;
  uint32_t now = HAL_GetTick();
  for (uint8_t i = 0; i < KEY_COUNT; i++)
  {{
    uint8_t level = Key_ReadRaw(i);
    uint8_t pressed = ({active}U == level) ? 1U : 0U;
    raw = (uint8_t)(raw | (pressed << i));
  }}
  /* 只有电平稳定超过 debounce 才更新消抖结果 */
  if (raw != s_raw_last)
  {{
    s_raw_last = raw;
    for (uint8_t i = 0; i < KEY_COUNT; i++)
    {{
      s_change_tick[i] = now;
    }}
  }}
  else
  {{
    for (uint8_t i = 0; i < KEY_COUNT; i++)
    {{
      uint8_t pressed = (uint8_t)((raw >> i) & 1U);
      uint8_t stable = (uint8_t)((s_stable >> i) & 1U);
      if ((pressed != stable) && ((now - s_change_tick[i]) >= {debounce_ms}U))
      {{
        s_stable = (uint8_t)((s_stable & ~(1U << i)) | (pressed << i));
      }}
    }}
  }}
  return s_stable;
}}
""",
        ["main.h", "key.h"],
    )
    return [h, c]


# ---------------------------------------------------------------- 蜂鸣器（无源，PWM）

def buzzer_files(timer: str, tim_clk: int, channel: str = "CH1") -> list[ComponentFile]:
    handle = f"htim{timer[3:]}"
    h = _h(
        "buzzer.h",
        "BUZZER",
        f"""
#include <stdint.h>

/* 十二平均律音名（A4 = 440 Hz） */
#define NOTE_C4  262
#define NOTE_D4  294
#define NOTE_E4  330
#define NOTE_F4  349
#define NOTE_G4  392
#define NOTE_A4  440
#define NOTE_B4  494
#define NOTE_C5  523
#define NOTE_D5  587
#define NOTE_E5  659
#define NOTE_F5  698
#define NOTE_G5  784
#define NOTE_A5  880
#define NOTE_B5  988
#define NOTE_SILENT 0

void Buzzer_Init(void);
/* 非阻塞发声：hz = 0 表示停；ms 后自动停 */
void Buzzer_Tone(uint16_t hz, uint16_t ms);
void Buzzer_Stop(void);
/* 主循环里调用，负责到时停音 */
void Buzzer_Task(void);
""",
    )
    c = _c(
        "buzzer.c",
        f"""
#define BUZZER_TIM_CLK {tim_clk}UL

extern TIM_HandleTypeDef {handle};

static uint32_t s_stop_tick;
static uint8_t s_active;

void Buzzer_Init(void)
{{
  Buzzer_Stop();
}}

void Buzzer_Tone(uint16_t hz, uint16_t ms)
{{
  if (hz == 0)
  {{
    Buzzer_Stop();
    return;
  }}
  /* 预分频固定到 1 MHz，ARR 直接等于 (1MHz / hz) - 1，精度足够听感 */
  uint32_t psc = (BUZZER_TIM_CLK / 1000000UL);
  psc = (psc == 0) ? 0 : (psc - 1);
  uint32_t arr = (1000000UL / hz);
  if (arr < 2)
  {{
    arr = 2;
  }}
  __HAL_TIM_SET_PRESCALER(&{handle}, psc);
  __HAL_TIM_SET_AUTORELOAD(&{handle}, arr - 1);
  __HAL_TIM_SET_COMPARE(&{handle}, TIM_CHANNEL_{channel[2:]}, arr / 2);
  HAL_TIM_PWM_Start(&{handle}, TIM_CHANNEL_{channel[2:]});
  s_active = 1;
  s_stop_tick = HAL_GetTick() + ms;
}}

void Buzzer_Stop(void)
{{
  HAL_TIM_PWM_Stop(&{handle}, TIM_CHANNEL_{channel[2:]});
  s_active = 0;
}}

void Buzzer_Task(void)
{{
  if (s_active && ((int32_t)(HAL_GetTick() - s_stop_tick) >= 0))
  {{
    Buzzer_Stop();
  }}
}}
""",
        ["main.h", "buzzer.h"],
    )
    return [h, c]


# ---------------------------------------------------------------- 串口命令行

def serial_cmd_files(instance: str, baud: int) -> tuple[list[ComponentFile], str, str]:
    """返回 (文件列表, 中断向量名, UART 句柄名)。"""
    handle = f"huart{instance[5:]}"
    irq = f"{instance}_IRQn"
    h = _h(
        "serial_cmd.h",
        "SERIAL_CMD",
        f"""
#include <stdint.h>

#define SERIAL_CMD_RX_SIZE 256
#define SERIAL_CMD_LINE_SIZE 96

void SerialCmd_Init(void);
/* 主循环调用：从环形缓冲取字节、拼行、执行命令 */
void SerialCmd_Task(void);
void SerialCmd_Send(const char *text);
/* 注册自定义命令处理：返回 1 表示已处理，0 交给内置命令 */
void SerialCmd_SetHandler(uint8_t (*handler)(const char *line));
""",
    )
    c = _c(
        "serial_cmd.c",
        f"""
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

extern UART_HandleTypeDef {handle};

static volatile uint8_t s_rx_buf[SERIAL_CMD_RX_SIZE];
static volatile uint16_t s_head;
static volatile uint16_t s_tail;
static uint8_t s_byte;                       /* HAL 接收中断的落脚点 */
static char s_line[SERIAL_CMD_LINE_SIZE];
static uint16_t s_len;
static uint8_t (*s_handler)(const char *line);

/* 中断里调用：单字节入环形缓冲，满了就丢最新字节 */
void SerialCmd_RxISR(uint8_t byte)
{{
  uint16_t next = (uint16_t)((s_head + 1U) % SERIAL_CMD_RX_SIZE);
  if (next != s_tail)
  {{
    s_rx_buf[s_head] = byte;
    s_head = next;
  }}
}}

/* HAL 的回调：收到一个字节后立刻重新挂上接收 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{{
  if (huart->Instance == {instance})
  {{
    SerialCmd_RxISR(s_byte);
    HAL_UART_Receive_IT(&{handle}, &s_byte, 1);
  }}
}}

void SerialCmd_Init(void)
{{
  s_head = 0;
  s_tail = 0;
  s_len = 0;
  s_handler = 0;
  HAL_UART_Receive_IT(&{handle}, &s_byte, 1);
  SerialCmd_Send("\\r\\n[cmd] ready, type 'help'\\r\\n");
}}

void SerialCmd_Send(const char *text)
{{
  HAL_UART_Transmit(&{handle}, (const uint8_t *)text, (uint16_t)strlen(text), HAL_MAX_DELAY);
}}

static void SerialCmd_Exec(char *line)
{{
  if (line[0] == '\\0')
  {{
    return;
  }}
  if ((s_handler != 0) && (s_handler(line) != 0))
  {{
    return;
  }}
  if (strcmp(line, "help") == 0)
  {{
    SerialCmd_Send("help          显示命令\\r\\n");
    SerialCmd_Send("info          打印系统节拍\\r\\n");
    SerialCmd_Send("echo <text>   原样回显\\r\\n");
  }}
  else if (strcmp(line, "info") == 0)
  {{
    char buf[64];
    (void)snprintf(buf, sizeof(buf), "tick = %lu ms\\r\\n", (unsigned long)HAL_GetTick());
    SerialCmd_Send(buf);
  }}
  else if (strncmp(line, "echo ", 5) == 0)
  {{
    SerialCmd_Send(line + 5);
    SerialCmd_Send("\\r\\n");
  }}
  else
  {{
    SerialCmd_Send("unknown: ");
    SerialCmd_Send(line);
    SerialCmd_Send("\\r\\n");
  }}
}}

void SerialCmd_SetHandler(uint8_t (*handler)(const char *line))
{{
  s_handler = handler;
}}

void SerialCmd_Task(void)
{{
  while (s_tail != s_head)
  {{
    char ch = (char)s_rx_buf[s_tail];
    s_tail = (uint16_t)((s_tail + 1U) % SERIAL_CMD_RX_SIZE);
    if ((ch == '\\r') || (ch == '\\n'))
    {{
      s_line[s_len] = '\\0';
      SerialCmd_Exec(s_line);
      s_len = 0;
    }}
    else if (s_len < (SERIAL_CMD_LINE_SIZE - 1U))
    {{
      s_line[s_len++] = ch;
    }}
    else
    {{
      s_len = 0;   /* 超长行直接丢弃 */
      SerialCmd_Send("\\r\\nline too long\\r\\n");
    }}
  }}
}}
""",
        ["main.h", "serial_cmd.h"],
    )
    return [h, c], irq, handle


# ---------------------------------------------------------------- ADC 旋钮

def adc_files(instance: str, channel: int, samples: int, label: str) -> list[ComponentFile]:
    handle = f"h{instance.lower()}"
    h = _h(
        "adc_knob.h",
        "ADC_KNOB",
        f"""
#include <stdint.h>

#define ADC_AVG_SAMPLES {samples}
/* {label}：ADC1_IN{channel}，DMA 循环采样 + 滑动平均 */

void Adc_Init(void);
void Adc_Task(void);
/* 最近一次滑动平均（0 ~ 4095） */
uint16_t Adc_GetAvg(void);
/* 换算成百分比（0 ~ 100） */
uint8_t Adc_GetPercent(void);
""",
    )
    c = _c(
        "adc_knob.c",
        f"""
extern ADC_HandleTypeDef {handle};

/* DMA 循环模式持续刷新这组样本 */
static volatile uint16_t s_samples[ADC_AVG_SAMPLES];
static uint8_t s_ready;
static uint16_t s_avg;

void Adc_Init(void)
{{
  s_ready = 0;
  s_avg = 0;
  if (HAL_ADCEx_Calibration_Start(&{handle}) != HAL_OK)
  {{
    Error_Handler();
  }}
  if (HAL_ADC_Start_DMA(&{handle}, (uint32_t *)s_samples, ADC_AVG_SAMPLES) != HAL_OK)
  {{
    Error_Handler();
  }}
}}

void Adc_Task(void)
{{
  uint32_t sum = 0;
  for (uint8_t i = 0; i < ADC_AVG_SAMPLES; i++)
  {{
    sum += s_samples[i];
  }}
  s_avg = (uint16_t)(sum / ADC_AVG_SAMPLES);
  s_ready = 1;
}}

uint16_t Adc_GetAvg(void)
{{
  return s_ready ? s_avg : 0;
}}

uint8_t Adc_GetPercent(void)
{{
  return (uint8_t)(((uint32_t)Adc_GetAvg() * 100U) / 4095U);
}}
""",
        ["main.h", "adc_knob.h"],
    )
    return [h, c]


# ---------------------------------------------------------------- SSD1306 OLED

def _font_body() -> tuple[str, str]:
    """从本地字库文件提取 8x16 ASCII 表；找不到时返回占位表 + 警告文案。"""
    if FONT_SRC.is_file():
        text = FONT_SRC.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"OLED_F8x16\s*\[\]\[16\]\s*=\s*\{(.*?)\n\};", text, re.S)
        if match:
            body = match.group(1)
            data_lines = [line for line in body.splitlines() if "0x" in line]
            if len(data_lines) >= 96:
                return "\n".join(data_lines), f"字库来源：{FONT_SRC.name}（96 个 ASCII 字符）"
    placeholder = "\n".join("  " + ",".join(["0x00"] * 16) + "," for _ in range(96))
    return placeholder, f"未找到字库文件 {FONT_SRC}，已生成空白字库（文字不显示，但不影响编译）"


def oled_files(instance: str, address: int, width: int, height: int) -> list[ComponentFile]:
    font, note = _font_body()
    pages = height // 8
    fh = _h(
        "oled_font.h",
        "OLED_FONT",
        f"""
/* 8x16 ASCII 字库（0x20 ~ 0x7F，共 96 个字符）。{note} */
#include <stdint.h>

const uint8_t OLED_F8x16[][16] = {{
{font}
}};
""",
    )
    h = _h(
        "oled_ssd1306.h",
        "OLED_SSD1306",
        f"""
#include <stdint.h>

#define OLED_WIDTH  {width}
#define OLED_HEIGHT {height}
#define OLED_ADDR   0x{address:02X}

/* 硬件 I2C{instance[3:]}（100% 兼容 SSD1306 0.96" 128x64） */
void OLED_Init(void);
void OLED_Clear(void);
void OLED_ClearLine(uint8_t page);
/* x: 列（0~127，按 8 像素对齐用 (x/8)*8），page: 行（0~7），内容为 8x16 字符 */
void OLED_ShowString(uint8_t x, uint8_t y, const char *text);
void OLED_Printf(uint8_t x, uint8_t y, const char *fmt, ...);
/* 基础图形：像素 / 水平进度条（0~100） */
void OLED_DrawPixel(uint8_t x, uint8_t y, uint8_t on);
void OLED_ProgressBar(uint8_t x, uint8_t y, uint8_t w, uint8_t percent);
""",
    )
    c = _c(
        "oled_ssd1306.c",
        f"""
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

extern I2C_HandleTypeDef hi2c{instance[3:]};

/* 显存按 SSD1306 的页结构组织：8 页 x 128 列 */
static uint8_t s_gram[OLED_HEIGHT / 8][OLED_WIDTH];

static void OLED_WriteCmd(uint8_t cmd)
{{
  uint8_t buf[2] = {{0x00, cmd}};
  HAL_I2C_Master_Transmit(&hi2c{instance[3:]}, (uint16_t)(OLED_ADDR), buf, 2, 100);
}}

static void OLED_WriteData(const uint8_t *data, uint16_t len)
{{
  static uint8_t buf[1 + 32];
  while (len > 0)
  {{
    uint16_t chunk = (len > 32) ? 32 : len;
    buf[0] = 0x40;
    memcpy(&buf[1], data, chunk);
    HAL_I2C_Master_Transmit(&hi2c{instance[3:]}, (uint16_t)(OLED_ADDR), buf, (uint16_t)(chunk + 1), 100);
    data += chunk;
    len = (uint16_t)(len - chunk);
  }}
}}

static void OLED_SetCursor(uint8_t page, uint8_t col)
{{
  OLED_WriteCmd((uint8_t)(0xB0 | (page & 0x07)));
  OLED_WriteCmd((uint8_t)(0x00 | (col & 0x0F)));
  OLED_WriteCmd((uint8_t)(0x10 | ((col >> 4) & 0x0F)));
}}

static void OLED_Flush(void)
{{
  for (uint8_t page = 0; page < (OLED_HEIGHT / 8); page++)
  {{
    OLED_SetCursor(page, 0);
    OLED_WriteData(s_gram[page], OLED_WIDTH);
  }}
}}

void OLED_Init(void)
{{
  HAL_Delay(100);
  OLED_WriteCmd(0xAE);            /* 关显示 */
  OLED_WriteCmd(0xD5); OLED_WriteCmd(0x80);
  OLED_WriteCmd(0xA8); OLED_WriteCmd(0x3F);   /* 复用率 1/64 */
  OLED_WriteCmd(0xD3); OLED_WriteCmd(0x00);
  OLED_WriteCmd(0x40);            /* 起始行 0 */
  OLED_WriteCmd(0x8D); OLED_WriteCmd(0x14);   /* 充电泵 */
  OLED_WriteCmd(0x20); OLED_WriteCmd(0x02);   /* 页寻址 */
  OLED_WriteCmd(0xA1);            /* 左右反置 */
  OLED_WriteCmd(0xC8);            /* 上下反置 */
  OLED_WriteCmd(0xDA); OLED_WriteCmd(0x12);
  OLED_WriteCmd(0x81); OLED_WriteCmd(0xCF);   /* 对比度 */
  OLED_WriteCmd(0xD9); OLED_WriteCmd(0xF1);
  OLED_WriteCmd(0xDB); OLED_WriteCmd(0x40);
  OLED_WriteCmd(0xA4);
  OLED_WriteCmd(0xA6);            /* 正常显示 */
  OLED_WriteCmd(0xAF);            /* 开显示 */
  OLED_Clear();
}}

void OLED_Clear(void)
{{
  memset(s_gram, 0x00, sizeof(s_gram));
  OLED_Flush();
}}

void OLED_ClearLine(uint8_t page)
{{
  if (page >= (OLED_HEIGHT / 8))
  {{
    return;
  }}
  memset(s_gram[page], 0x00, OLED_WIDTH);
  OLED_SetCursor(page, 0);
  OLED_WriteData(s_gram[page], OLED_WIDTH);
}}

void OLED_ShowString(uint8_t x, uint8_t y, const char *text)
{{
  uint8_t page = (uint8_t)(y / 8);
  if (page >= (OLED_HEIGHT / 8))
  {{
    return;
  }}
  while ((*text != '\\0') && (x <= (OLED_WIDTH - 8)))
  {{
    uint8_t ch = (uint8_t)*text;
    if ((ch < 0x20) || (ch > 0x7F))
    {{
      ch = '?';
    }}
    const uint8_t *glyph = OLED_F8x16[ch - 0x20];
    for (uint8_t col = 0; col < 8; col++)
    {{
      s_gram[page][x + col] = glyph[col];
      if ((page + 1) < (OLED_HEIGHT / 8))
      {{
        s_gram[page + 1][x + col] = glyph[col + 8];
      }}
    }}
    OLED_SetCursor(page, x);
    OLED_WriteData(&s_gram[page][x], 8);
    if ((page + 1) < (OLED_HEIGHT / 8))
    {{
      OLED_SetCursor((uint8_t)(page + 1), x);
      OLED_WriteData(&s_gram[page + 1][x], 8);
    }}
    x = (uint8_t)(x + 8);
    text++;
  }}
}}

void OLED_Printf(uint8_t x, uint8_t y, const char *fmt, ...)
{{
  char buf[32];
  va_list args;
  va_start(args, fmt);
  (void)vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  OLED_ShowString(x, y, buf);
}}

void OLED_DrawPixel(uint8_t x, uint8_t y, uint8_t on)
{{
  if ((x >= OLED_WIDTH) || (y >= OLED_HEIGHT))
  {{
    return;
  }}
  uint8_t page = (uint8_t)(y / 8);
  uint8_t mask = (uint8_t)(1U << (y % 8));
  if (on)
  {{
    s_gram[page][x] |= mask;
  }}
  else
  {{
    s_gram[page][x] &= (uint8_t)~mask;
  }}
  OLED_SetCursor(page, x);
  OLED_WriteData(&s_gram[page][x], 1);
}}

void OLED_ProgressBar(uint8_t x, uint8_t y, uint8_t w, uint8_t percent)
{{
  if (percent > 100)
  {{
    percent = 100;
  }}
  uint8_t filled = (uint8_t)(((uint16_t)w * percent) / 100U);
  for (uint8_t i = 0; i < w; i++)
  {{
    OLED_DrawPixel((uint8_t)(x + i), y, (i < filled) ? 1 : 0);
    OLED_DrawPixel((uint8_t)(x + i), (uint8_t)(y + 1), (i < filled) ? 1 : 0);
  }}
}}
""",
        ["main.h", "oled_ssd1306.h", "oled_font.h"],
    )
    return [fh, h, c]
