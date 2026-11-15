"""跑 10 个覆盖五类语料的问题，记录路由、命中来源、答案与耗时。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stm32_rag.answer import answer  # noqa: E402
from stm32_rag.router import MultiSourceRetriever  # noqa: E402

QUESTIONS = [
    "RCC_CFGR 里 PLLMUL 位怎么配置成 9 倍频？",          # RM 寄存器
    "TIMx_PSC 预分频寄存器的作用是什么？",                  # RM 寄存器
    "STM32F103C8T6 的 Flash 和 SRAM 各有多大？",          # DS 器件参数
    "STM32F103C8T6 是什么封装，有多少个引脚？",             # DS 封装
    "HAL_UART_Transmit 的参数分别是什么？",               # UM API
    "HAL_GPIO_Init 的 Mode 成员可以取哪些值？",            # UM/HAL typedef
    "__HAL_RCC_GPIOA_CLK_ENABLE 宏最终操作哪个寄存器？",    # HAL 源码/宏
    "HAL_ADC_Start 可能返回哪些状态？",                    # HAL 源码
    "GPIO 例程里 LED 闪烁是怎么实现的？",                   # 例程
    "用 HAL 让 PA5 输出高电平，底层写的是哪个寄存器？",       # 跨源 HAL+RM
]

retriever = MultiSourceRetriever()
lines = [f"# 10 题实测", "", retriever.describe(), ""]

for i, question in enumerate(QUESTIONS, start=1):
    plan = retriever.plan(question)
    t0 = time.perf_counter()
    try:
        result = answer(question, retriever=retriever)
    except Exception as exc:  # noqa: BLE001
        lines.append(f"## {i}. {question}")
        lines.append(f"- 路由：{' → '.join(plan)}")
        lines.append(f"- **失败**：{type(exc).__name__}: {exc}")
        lines.append("")
        print(f"[{i}] FAIL {type(exc).__name__}: {exc}")
        continue
    cost = time.perf_counter() - t0
    print(f"[{i}] ok {cost:.1f}s")

    sources = result.sources()
    docs = sorted({h.metadata.get("doc", "") for h in result.hits})
    lines.append(f"## {i}. {question}")
    lines.append(f"- 路由：{' → '.join(plan)}")
    lines.append(f"- 命中来源：{', '.join(d for d in docs if d)}；片段 {len(result.hits)} 个；耗时 {cost:.1f}s")
    lines.append(f"- 答案：\n\n{result.text.strip()}")
    lines.append(f"- 出处：{'；'.join(sources)}")
    lines.append("")

Path("_qa10.md").write_text("\n".join(lines), encoding="utf-8")
print("written")
