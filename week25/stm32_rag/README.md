# STM32 AI 开发助手（`stm32_rag`）

把一句中文需求，变成**可编译的 STM32F103 工程**；写代码之前，先用手册和 HAL 源码把参数核对清楚。

它由两部分组成，共用一套知识库：

- **知识库（RAG）**：RM0008 参考手册、数据手册、UM1850 HAL 说明书、HAL 源码、Nucleo 例程，
  混合检索 + 路由 + LLM 重排，回答带页码/文件出处；
- **开发助手**：需求分析 → 规格校验 → 代码生成 → Keil 编译 → 错误修复 → 交付报告，
  既能从零建工程，也能在 CubeMX 工程上做增量修改。

```text
自然语言需求
   │
   ▼
Agent / Orchestrator ──► STM32 RAG（RM0008 / DS / UM1850 / HAL / 例程）
   │                 ├─► 工程分析（.uvprojx / 源码 / 启动文件 / .ioc）
   │                 ├─► 代码生成（时钟 / GPIO / TIM / USART / DMA / NVIC）
   │                 └─► Keil MCP Server（工程操作 / 编译 / 日志分析）
   ▼
Keil MDK-ARM（UV4 命令行编译）
   ▼
交付：源码 + 工程配置 + 编译日志 + 报告 + 修改记录
```

---

## 1. 快速开始

### 1.1 环境

| 依赖 | 位置 | 说明 |
| --- | --- | --- |
| Python | `d:\project\step1\week23\.venv` | langchain / chromadb / pymilvus 等都装在这里 |
| 模型配置 | `week23/.env` | 百炼 `text-embedding-v4`（1024 维）+ `kimi-k3.1-pro` |
| Keil | `C:\Keil_v5\UV4\UV4.exe` | 命令行构建，可用 `STM32_UV4` 覆盖 |
| 知识库 | `chroma_db/` | 没有就按第 7 节建库 |
| 图谱（可选） | `../neo4j`（端口 7687） | 器件事实图 `graph` + 文档关系图 `graph-docs`；不启动也能跑，只是少了 `graph`/`--graph-expand` 能力 |

```powershell
& d:\project\step1\week23\.venv\Scripts\Activate.ps1
cd d:\project\step1\week25
python -m stm32_rag --help
```

> 如果 `python` 不识别，直接写全路径：
> `& 'd:\project\step1\week23\.venv\Scripts\python.exe' -m stm32_rag ...`

### 1.2 三条最常用的命令

```powershell
# 1) 从零生成一个新工程（不需要 CubeMX，自动拷 CMSIS/启动文件/HAL）
python -m stm32_rag develop "配置 PA8 输出 1MHz PWM，占空比 50%" --workspace workspaces/pwm_pa8

# 2) 在已有 CubeMX 工程上增量修改（自动识别 .ioc，不覆盖原代码）
python -m stm32_rag develop "USART1 115200 每秒打印 Hello" --workspace "E:\Codex\...\STM32F103C8T6_Base"

# 3) 需求说法比较自由时用 LLM 编排（失败自动回退到 develop）
python -m stm32_rag agent "让 PC13 每 500ms 翻转一次" --workspace workspaces/led_demo
```

### 1.3 三种调用方式

| 入口 | 怎么用 | 适合 |
| --- | --- | --- |
| 命令行 | `python -m stm32_rag <子命令>` | 日常、脚本、CI |
| MCP | VS Code / Claude 里自然语言调 `keil_*` 工具 | 在 IDE 里边看代码边改（配置见第 9 节） |
| 代码库 | `from stm32_rag import develop, DevAssistant` | 二次开发、批量任务 |

`develop` 与 `agent` 的分工：前者是规则解析，不花 token、结果可复现；
后者让模型理解更自由的说法，但**落盘和编译始终走同一套工具**，模型不能绕过校验。

### 1.4 当成库用

```python
from stm32_rag import answer, develop, DevAssistant, HybridRetriever, Stm32Index

# 知识库问答
retriever = HybridRetriever(index=Stm32Index())
result = answer("USART 波特率怎么算？", retriever=retriever)
print(result.text)
print("\n".join(result.sources()))

# 开发助手（确定性 / LLM）
delivery = develop("配置 PA8 输出 1MHz PWM", workspace=r"D:\tmp\pwm_demo")
print(delivery.summary())

assistant = DevAssistant(workspace=r"D:\tmp\pwm_demo")
print(assistant.run("再把占空比改成 30%").summary())
```

### 1.5 一次运行长什么样

```text
> python -m stm32_rag develop "配置 PA8 输出 1 MHz PWM，占空比 50%"

需求：配置 PA8 输出 1 MHz PWM，占空比 50%
✅ 编译通过（退出码 0：无错误、无警告）
Program Size: Code=7752 RO-data=304 RW-data=12 ZI-data=1108
产物: ...\MDK-ARM\Objects\stm32f103rb-pwm_pa8.axf
工程：...\MDK-ARM\stm32f103rb-pwm_pa8.uvprojx
报告：...\REPORT.md
修改记录：...\CHANGES.md
```

生成的关键代码：

```c
htim1.Init.Prescaler = 0;      /* ÷1 */
htim1.Init.Period    = 71;     /* 72 MHz / 72 = 1 MHz */
sConfigOC.Pulse      = 36;     /* 36/72 = 50% */
HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
```

`REPORT.md` 会把账算给你看：

```text
- PA8 → TIM1_CH1；TIM1CLK = PCLK2×1 = 72 MHz（APB2）；
  PSC=0（分频 1）ARR=71（周期 72）CCR=36 → 实际 1000 kHz，误差 +0.000%
- USART1：PCLK=72 MHz，BRR=0x271，实际 115207.5 bps（误差 +0.18%）
```

---

## 2. 能力总览

| 能力 | 需求示例 | 状态 |
| --- | --- | --- |
| GPIO 输出 | `PC13 每 500ms 翻转` | ✅ |
| GPIO 输入 / 外部中断 | `按键接 PA0，下降沿中断` | ✅ |
| TIM PWM | `PA8、PA11 输出 1kHz PWM` | ✅ 同一定时器多通道（同频） |
| TIM 更新中断 | `TIM2 产生 1kHz 中断，翻转 PA5` | ✅ |
| USART 发送 | `USART1 115200 每秒打印 Hello` | ✅ 轮询 / DMA |
| USART 接收 / 命令行 | `USART1 串口命令行` | ✅ 中断接收 + 256 B 环形缓冲 + 行解析 |
| 按键（消抖） | `按键 PB12 PB13 PB14 PB15` | ✅ 20 ms 消抖组件 |
| 无源蜂鸣器 | `蜂鸣器 PB0` | ✅ 音调表 + 非阻塞 `Buzzer_Tone` |
| ADC（电位器） | `电位器 PA0` | ✅ ADC + DMA + 滑动平均 |
| I2C / OLED | `OLED 显示` | ✅ SSD1306 驱动 + 8x16 ASCII 字库 + 绘图 API |
| 1 ms 调度骨架 | `1ms 调度` | ✅ `Sched_Add/Run`（基于 Hal_GetTick） |
| SPI / CAN | — | ❌ 待做 |
| 编码器 / 输入捕获 | — | ❌ 待做 |

工程类型：

| 类型 | 入口 | 说明 |
| --- | --- | --- |
| 新建工程 | `develop` / `scaffold` | 从 vendor 拷 CMSIS + HAL，生成可直接编译的 `.uvprojx` |
| CubeMX 工程 | `develop` / `agent` | 读 `.ioc`，只往 USER CODE 段增量插入 |
| 已有工程 | `compile` / `project` | 编译、看结构、加文件/宏/头文件路径 |

---

## 3. 架构与模块

```
stm32_rag/
├─ __init__.py        # 对外 API（from stm32_rag import ...）
├─ __main__.py        # python -m stm32_rag
├─ cli.py             # 全部子命令入口
├─ mcp_keil_server.py # Keil MCP Server（11 个工具）
├─ core/              # 全局配置 + 文本清洗 + 切块
├─ corpus/            # 语料加载（PDF / HAL / Markdown）+ 插图 + Chroma 建库
├─ rag/               # 混合检索 / 路由 / 回答 / 上传入库 / 评测
├─ graphs/            # 器件事实图 + 文档关系图（Neo4j）
├─ gen/               # 器件表 / 代码生成 / 脚手架 / .uvprojx / CubeMX
├─ build/             # UV4 编译闭环 + 确定性流水线
└─ assistant/         # 工作区会话（工具实现）+ LangChain Agent
```

| 能力 | 模块 | 职责 |
| --- | --- | --- |
| 手册知识库 | `corpus/loader*.py` / `core/chunker.py` / `corpus/index.py` | PDF/源码/例程 → 切块 → Chroma |
| 混合检索 | `rag/retrieve.py` / `rag/router.py` / `rag/answer.py` | 向量 + BM25 → RRF → LLM 重排 → 带出处回答 |
| 上传与评测 | `rag/ingest.py` / `rag/evaluate.py` | MinerU 入库、评测集与 A/B |
| 器件事实库 | `gen/mcu.py` | 引脚复用、DMA 映射、TIMxCLK、PSC/ARR、UART BRR |
| 工程分析 | `gen/project.py` | 读写 `.uvprojx`：器件/分组/文件/宏/头文件路径 |
| 代码生成 | `gen/codegen.py` | 先校验再生成 `main.c / main.h / stm32f1xx_it.c` |
| 工程脚手架 | `gen/scaffold.py` | 从 vendor 生成可编译工程（HAL 按模块裁剪） |
| CubeMX 兼容 | `gen/cubemx.py` | 解析 `.ioc`、增量插入、多轮规格累积 |
| 编译诊断 | `build/keil.py` | `UV4 -b`、日志解析、错误→修复动作映射 |
| 确定性编排 | `build/pipeline.py` | 需求解析 → 生成 → 编译 → 自修复 → 报告 |
| 知识图谱 | `graphs/graph.py` / `graphs/graph_docs.py` | 器件事实图 / 文档关系图 |
| LLM 编排 | `assistant/agent.py` + `assistant/tools.py` | LangChain `create_agent`，9 个工具 |
| MCP Server | `mcp_keil_server.py` | 11 个 MCP 工具 |
| CLI | `cli.py` | 全部子命令入口 |

---

## 4. 代码生成：先校验，再下笔

`codegen.validate()` 在写文件前拦下这些问题（校验不过不产生任何代码）：

| 检查 | 例子 |
| --- | --- |
| 封装有没有这个端口 | STM32F103RB 是 64 脚，没有 GPIOE |
| 引脚复用对不对 | PA8 只有 TIM1_CH1，要求 TIM3 会报“可用：TIM1_CH1” |
| 频率能不能分出来 | PSC/ARR 都是 16 位，1 MHz 与 1 Hz 的预算不同 |
| 占空比范围 | 0~100%，CCR 会夹在 0~ARR+1 |
| UART 引脚匹配 | USART1 的 TX/RX 必须是 PA9/PA10（重映射要另改 AFIO） |
| 波特率误差 | BRR 误差 > 3% 给警告 |
| DMA 映射存在 | USART1_TX 只能是 DMA1_Channel4 |
| 定时器时钟来源 | TIM1 挂 APB2；TIM2~4 挂 APB1，APB1 分频≠1 时 ×2 |

生成的代码沿用 CubeMX 命名（`MX_xxx_Init` / `HAL_xxx_MspInit`），和 UM1850、HAL 源码里的函数名一一对应，检索到的资料可以直接验证。

---

## 5. 编译闭环

### 5.1 UV4 命令行的六个坑

1. **UV4 是 GUI 程序**：`&` / Popen 不会等它结束，必须同步等待并读退出码，否则永远读到上一次的日志；
2. **工程文件带 UTF-8 BOM 会 exit 15**：不编译、不写日志。`KeilProject.save()` 固定写无 BOM；
3. **`<Define>` 用逗号分隔**（`USE_HAL_DRIVER,STM32F103xB`），**`<IncludePath>` 用分号**；
4. **μVision 正占用工程时命令行构建静默失败**：拿不到日志时会结束残留 `UV4.exe` 重试一次（`STM32_KILL_STALE_UV4=0` 可关闭）；
5. **`UV4 -r` 会拉起 GUI 卡住**：全量重编改成删 `Objects/*.o` 再 `-b`；
6. **日志里的错误行是 include 链**，用 `.....` 串联，要按 `....` 切分取最深一段。

退出码对照（`keil.EXIT_CODES`）：

| 码 | 含义 |
| --- | --- |
| 0 / 1 | 无错误无警告 / 仅有警告（都算通过） |
| 2 / 3 | 有错误 / 致命错误 |
| 11 / 15 / 20 | 打不开工程 / 读取工程失败（BOM、被占用）/ License |

### 5.2 错误 → 修复建议 → 重编译

`keil.suggest_fixes()` 把常见错误映射成动作，`develop` 自动跑最多 `--max-fix` 轮：

| 现象 | 自动动作 |
| --- | --- |
| `Please select first the target STM32F1xx device` | 加宏 `STM32F103xB` |
| `stm32f1xx_hal.h: No such file` | 加头文件路径 `..\Drivers\STM32F1xx_HAL_Driver\Inc` |
| `Undefined symbol HAL_TIM_PWM_Init` | 补 `stm32f1xx_hal_tim.c` + 打开 `HAL_TIM_MODULE_ENABLED` |
| `Undefined symbol SystemInit` | 把 `system_stm32f1xx.c` 加进工程 |
| `use of undeclared identifier 'HAL_'` | 加宏 `USE_HAL_DRIVER` |
| `No space in execution regions` | 标记人工处理（裁剪功能或换型号） |

语法/逻辑错误不在自动修复范围内——那属于 Agent 用 `read_source` + `write_source` 逐处修改、再编译验证的场景。

---

## 6. 在 CubeMX 工程上增量修改

CubeMX 的 `main.c / stm32f1xx_hal_msp.c / stm32f1xx_it.c` 里有 `/* USER CODE BEGIN/END */` 和已生成的
`MX_GPIO_Init` / `HAL_MspInit`。直接整文件生成会丢掉这些，所以助手识别到 `.ioc` 会自动切到增量模式。

### 6.1 处理规则

| 需求 vs `.ioc` 现状 | 处理方式 |
| --- | --- |
| 引脚/外设未配置 | 生成 `MX_xxx_Init`，插进 USER CODE 段，调用加到 `USER CODE 2` |
| 串口已配置，要求改波特率 | 改 `huart1.Init.BaudRate`，同步 `.ioc` 的 `USART1.BaudRate` |
| 引脚已配成同一信号 | 跳过，不重复生成 |
| 引脚已被别的信号占用 | 报冲突并列出实际信号，不猜 |
| 定时器已是 CubeMX 外设 | 不重复初始化 |
| USER CODE 标记被手工删掉 | 拒绝自动改，提示改用 `read_source` + `write_source` |

生成代码用标记包起来，重复执行是替换而不是叠加：

```c
/* stm32_rag:functions:BEGIN */
static void MX_TIM1_Init(void) { ... }
/* stm32_rag:functions:END */
```

### 6.2 为什么用「内联 MSP」

CubeMX 已经把 `HAL_TIM_Base_MspInit` / `HAL_UART_MspInit` 写在 `stm32f1xx_hal_msp.c`，
再定义一次就是重复符号。增量模式把时钟使能与引脚复用写进 `MX_xxx_Init()`，不碰 MSP 回调。

### 6.3 多轮需求是累积的

规格会合并到 `<工程>/.stm32rag/spec.json` 再整体重算，同键覆盖、其余保留：

```text
第 1 轮  配置 PA8 输出 1MHz PWM        → Code=4268
第 2 轮  USART1 115200 每秒打印 Hello  → Code=7412（PWM 仍在）
第 3 轮  把 USART1 波特率改成 9600     → Code=7412（PWM 仍在）
```

想从零重来：删掉 `.stm32rag/`，或在 CubeMX 里重新生成工程。

> 助手**不回写 `.ioc`**（除波特率字段）。CubeMX 重新生成会保留 USER CODE，代码不丢，
> 但图形界面里看不到助手新增的外设。要图形化维护，建议外设也在 CubeMX 里配好。

---

## 7. 知识库（RAG）

### 7.1 语料与集合

| 来源 | 内容 | 集合 | 命令 |
| --- | --- | --- | --- |
| `rm` | RM0008 参考手册（寄存器级） | `stm32_rm0008_text_embedding_v4` | `build --source rm --rebuild` |
| `ds` | STM32F103x8/B 数据手册 | `stm32_ds_f103xb_text_embedding_v4` | `build --source ds` |
| `um` | UM1850 HAL/LL 说明书（2591 块 / 60 章） | `stm32_um1850_v3_text_embedding_v4` | `build --source um` |
| `hal` | F1 HAL 源码（889 函数 / 241 类型 / 1057 宏组） | `stm32f1_hal_v1110_text_embedding_v4` | `build --source hal` |
| `examples` | Nucleo 例程 `main.c`（180 块） | `stm32f1_hal_examples_text_embedding_v4` | `build --source examples` |

### 7.2 切块与检索

| 环节 | 通用做法 | 本模块 |
| --- | --- | --- |
| 加载 | 每页一个 Document | 1623 条书签当锚点，按 y 坐标切小节（一页可切多节） |
| 清洗 | 中文折行正则 | 英文折行 + 连字符断词 + 按坐标裁页眉页脚 |
| 表格 | 纯文本（位表变 `31 30 29 …`） | `find_tables()` 转 Markdown，剔除被表格覆盖的碎文本 |
| 插图 | 看不到 | `Figure N.` 题注定位 → 裁 PNG → `kind=figure` 块，`--with-figures` 发给视觉模型 |
| 切块 | 50 字符 | tiktoken 计数、480 token/块、表格原子不切、短节合并 |
| 噪声 | 全收 | 整章排除 `Revision history` |
| 检索 | 纯向量 top-3 | 向量 + BM25 → RRF → LLM 重排 → 按小节奏扑 → 邻居补全 |
| 型号 | — | 识别 F101~104 / F105~107，优先对应密度章节 |

跨集合路由（`--sources auto`）：

| 问题特征 | 路由顺序 |
| --- | --- |
| `HAL_` / `LL_` 符号 | hal → um → examples |
| 寄存器 / 位域 / 复位值 | rm → hal |
| 芯片型号 / 电气参数 | ds → rm → hal |
| 函数 / 宏 / 回调 / 怎么配置 | um → hal → examples |
| 其他 | rm → hal → ds |

融合：各源内先 RRF，再按路由顺序加权（1.0 / 0.6 / 0.4 …）做第二次 RRF，最后统一 LLM 重排。
两个提升召回的小机制：**类型规则分**（问函数抬 `kind=function`）、**中文→API 扩展**（高电平→`SET/high`、串口→`UART/USART`）。

### 7.3 上传 PDF（MinerU → Markdown → 建库）

```powershell
python -m stm32_rag ingest D:\docs\应用笔记.pdf --tier basic   # 全自动
python -m stm32_rag ingest --md D:\docs\笔记.md                # 已有 Markdown
python -m stm32_rag ingest 手册.pdf --pages 1-10 --force       # 先试 10 页
python -m stm32_rag ingest a.pdf b.pdf                         # 批量排队
```

流程：`PDF → MinerU → mineru_out/<stem>.md → 按标题切节 → 清洗 → 嵌入 → chroma_db/stm32_doc_<slug>_... → custom_sources.json`。
上传文档自动参与检索（权重 0.8，且每个来源至少保留 1 个重排候选），也可 `ask "..." --sources docs:<slug>` 只查这份。

MinerU 默认在 `D:\project\step3\llm\Scripts\mineru.exe`（服务 `127.0.0.1:15980`），
可用 `MINERU_EXE` / `MINERU_TIER` 覆盖；解析产物留在 `mineru_out/`，重跑建库不会重复解析。
批量时多本先入队再并发解析；嵌入按批并发（`--workers`），切块仍是单线程（CPU 密集）。

### 7.4 实测数据

| 指标 | 数值 |
| --- | --- |
| RM0008 | 1133 页 / 12.5 MB / 1623 书签 / 1334 小节 / 474 张表格 |
| 切块 | 2016 正文 + 364 插图 = 2380 块 |
| 建库耗时 | 首次解析 979 s + 向量化 146 s（之后走缓存） |
| 评测（20 题，锚点 + MRR） | 全中 100%，平均覆盖 97.5%，MRR 1.000 |
| 多语料评测（8 题跨 RM/DS/HAL/例程） | 全中 100%，覆盖 100%，MRR 1.000 |

### 7.5 关键参数

`config.py`：

- `BuildConfig.max_tokens / min_tokens / overlap_tokens`：块大小与短节合并；
- `BuildConfig.exclude_chapter_patterns`：整章排除（默认 `Revision history`）；
- `RetrieveConfig.vector_k / bm25_k / rerank_pool / final_k`：两路召回与最终小节数；
- `RetrieveConfig.max_per_section`：单节最多保留几块（寄存器位定义表常跨 3 块，别调到 2）；
- `RetrieveConfig.max_context_tokens`：进 Prompt 的正文上限（默认 6000）；
- `RetrieveConfig.use_rerank / expand_neighbors / device_boost`：重排、邻居补全、按型号偏置。

环境变量：

| 变量 | 作用 |
| --- | --- |
| `LAB_CONFIG_DIR` | 模型配置目录（默认 `week23/20261027`，同时读 `week23/.env`） |
| `CHAT_MODEL` | 生成与重排模型（默认 `kimi-k3.1-pro`） |
| `STM32_EMBED_MODEL / STM32_EMBED_DIM` | 嵌入模型与维度（默认 `text-embedding-v4` / 1024，**换它必须换集合名重建**） |
| `STM32_PDF / STM32_DS_PDF / STM32_UM_PDF` | 手册路径 |
| `STM32_HAL_ROOT` | `vendor/STM32CubeF1` 路径 |
| `STM32_DB_DIR` | Chroma 目录 |
| `STM32_WORKSPACE_DIR` | 默认交付目录（`week25/workspaces`） |
| `STM32_UV4` | `UV4.exe` 路径 |
| `STM32_BUILD_TIMEOUT / STM32_KILL_STALE_UV4` | 编译超时 / 是否清理残留 µVision 进程 |
| `STM32_DEVICE / STM32_SYSCLK / STM32_HSE` | 默认器件与时钟 |
| `MINERU_EXE / MINERU_TIER` | MinerU 可执行文件与解析档位 |

---

## 8. 知识图谱（Neo4j）

这里有两张图，数据源不同、用途也不同：

| | 器件事实图（8.1~8.2） | 文档关系图（8.3） |
| --- | --- | --- |
| 数据来源 | `mcu.py` 静态表 | Chroma 的 chunk metadata |
| 节点 | Device / Pin / Timer / Usart / DmaChannel / Bus / Irq | Doc / Chapter / Section / Chunk / Symbol |
| 回答 | 「PA8 只能是 TIM1_CH1」这类**确定事实** | 「这个符号在哪讲、谁定义了它、哪块和它相关」 |

### 8.1 器件事实图：建图

```powershell
# 先确认 Neo4j 5.x 社区版在跑（部署见 ../neo4j/README.md）
& d:\project\step1\week23\.venv\Scripts\python.exe -m stm32_rag graph --build
```

```text
建图完成
节点：Device 5、Pin 80、Timer 8、Usart 5、DmaChannel 11、Bus 2、Irq 9
关系：CLOCKED_BY 13、DMA 25、HAS_PIN 297、HAS_TIMER 32、HAS_USART 23、IRQ 9、TIM_CH 16、USART_ROLE 10
校验：图与 mcu.py 一致 ✓
```

图直接由 `mcu.py` 的静态表生成（单一数据源），`graph --verify` 会把图里读回来的结果
和 `mcu.py` 现算的结果逐条对比，表改了忘了重建会立刻报出来。
建图幂等，关系每次全量重建，所以 `mcu.py` 改小（比如 C8 去掉 UART4）不会残留旧边。

### 8.2 器件事实图：常用查询

| 命令 | 回答什么 |
| --- | --- |
| `graph --pin PA8` | PA8 只能复用成 TIM1_CH1（高级定时器） |
| `graph --pin PC13` | 没有定时器/串口复用，只能当普通 GPIO |
| `graph --usart USART1` | TX=PA9、RX=PA10、DMA1_Channel4/5、USART1_IRQn、APB2 |
| `graph --timer TIM3` | CH1~CH4 各在哪个引脚、DMA 通道、中断 |
| `graph --pwm-pins` | 全部 16 个可出 PWM 的引脚 |
| `graph --device STM32F103C8` | 64KB/20KB、4 个定时器、**只有 3 个串口**（UART4/5 的 PC10~PC12 在 LQFP48 上没引出） |
| `graph --stats` / `--verify` | 图规模 / 与 `mcu.py` 互验 |
| `graph --cypher "MATCH ..."` | 只读 Cypher（`CREATE/MERGE/DELETE/SET` 会被拒绝） |

### 8.3 文档关系图（GraphRAG）

向量检索只拿到“语义相近的块”；图能把**同一个符号在不同语料里的块**串起来：
`HAL_GPIO_Init` 在 UM1850 的 API 说明、HAL 源码的实现、例程的用法，一次就都拿到。

**不需要 LLM 抽实体** —— 建库时 metadata 里已经有 `symbols`/`symbol`/`chapter`/`section_id`/
`chunk_index`/`page`，直接读一遍就是图：

```powershell
& d:\project\step1\week23\.venv\Scripts\python.exe -m stm32_rag graph-docs --build
```

```text
建图完成
节点：Doc 5、Chapter 180、Section 5999、Chunk 8154、Symbol 3855、Peripheral 52
关系：IN_DOC 8154、IN_CHAPTER 8154、IN_SECTION 8154、NEXT 2155、MENTIONS 9026、DEFINES 4866、CALLS 947、USES_PERIPHERAL 2071、SAME_AS 24
语料：rm 2380、ds 176、um 2591、hal 2763、examples 244
```

图的边全是确定性生成的：

| 关系 | 含义 | 来源 |
| --- | --- | --- |
| `IN_DOC` / `IN_CHAPTER` / `IN_SECTION` | 块 → 文档 / 章 / 小节 | `doc`、`chapter`、`section_id` |
| `NEXT` | 同小节相邻块 | `chunk_index` 排序 |
| `MENTIONS` | 块提到了哪些符号 | `symbols` / `symbol` |
| `DEFINES` | 块定义了这个符号 | `kind=function\|typedef\|macro` |
| `CALLS` | 函数 → 它调用的 HAL/LL 符号 | 函数体正则 `HAL_xxx(`（947 条） |
| `USES_PERIPHERAL` | 块/符号 → 外设（TIM1、GPIOA、DMA1_Channel4…） | 正文 + 标题正则（2071 条） |
| `SAME_AS` | `Peripheral` → 器件事实图的 `Timer`/`Usart`/`DmaChannel` | 同名对齐（跨图 JOIN） |

检索时开一个开关就会做**跨语料扩展**（种子块 → 共同符号 → 其它语料的块）：

```powershell
python -m stm32_rag ask "HAL_GPIO_Init 的参数和返回值是什么？" --graph-expand -k 4 --show-context
python -m stm32_rag eval --graph-expand          # A/B 对比
```

实测（关闭重排、20 道寄存器题）：

| 配置 | 全中率 | 命中覆盖率 | MRR |
| --- | --- | --- | --- |
| 基线 | 95.0% | 95.0% | 0.893 |
| `--graph-expand` | 95.0% | 95.0% | 0.893 |

寄存器题本来就不靠跨语料，所以指标不变（说明没有退化）；真正的收益在函数类问题：
问 `HAL_GPIO_Init` 时，扩展会把 UM1850 §20.2.5 的 API 说明补进上下文（`+图 via HAL_GPIO_Init`）。

调试/巡检命令：

| 命令 | 作用 |
| --- | --- |
| `graph-docs --stats` | 节点/关系数量 |
| `graph-docs --symbol HAL_GPIO_Init` | 符号出现在哪些块（带来源、页码、是否定义处） |
| `graph-docs --chunk <chunk-id>` | 块详情 + 小节/章 + 相邻块 |
| `graph-docs --expand <chunk-id>` | 单块能扩出哪些跨语料邻块 |
| `graph-docs --calls HAL_UART_Init` | 调用链（2 跳）；`--callers` 反向看谁调用它 |
| `graph-docs --peripheral TIM1` | 哪些块讲这个外设 + 器件图的通道/DMA/中断（跨图） |
| `graph-docs --cypher "MATCH ..."` | 只读 Cypher |

三个索引细节：

1. **Chunk 主键是「来源 + Chroma id」**：RM0008 和 DS 的小节号会撞（`5.1.1#0` 在两边都有），
   只按 chunk id 建节点会丢 14 个块；
2. `router.search` 的合并键也一并改成了 `(source, id)` —— 原来只按 id 合并，
   撞号的块会被后一个源**静默覆盖**；
3. **`loader_hal._scan_units` 的函数边界用大括号配对**（原来取「第一个 `strip()=='}'` 的行」，
   会把 `HAL_UART_Init` 截成 718 字符、丢掉 `HAL_UART_MspInit` 调用）。修复后
   `hal` 语料从 2187 块变 2763 块、`examples` 从 180 变 244，**已重建这两个集合**。

重建 Chroma 语料后记得重跑 `graph-docs --build`（chunk id 会变）。

### 8.4 跨语料 A/B（`eval-graph`）

`eval_cases.json` 里 20 题全是寄存器题，测不出图扩展的价值，所以另开一组
`eval_cases_functions.json`（12 题，问 HAL/LL 函数参数，期望来源是 UM1850）：

```powershell
python -m stm32_rag eval-graph          # 默认 k=6，关重排
eval-graph --json                       # 结构化输出
```

实测：

```text
跨语料命中（期望来源进 top-6）：12 题 | 基线 0.0% → 图扩展 100.0% | 平均新增块 4.42
```

原因很直接：函数类问题里 BM25/向量都被 HAL 源码块制霸（关键词命中太强），
UM1850 的 API 说明永远挤不进 top-k；图按符号把 UM 块直接补进来，12 题全部命中。

寄存器题那 20 题指标不变（95.0% / MRR 0.893），说明没有退化。

### 8.4 Agent 怎么用

`stm32_graph_query` 是器件事实图工具，`stm32_doc_graph` 是文档关系图工具（共 9 个工具）。
系统提示里已经把分工写清楚：**手册原文**找 `stm32_knowledge_search`；
**引脚/DMA/总线/中断**找 `stm32_graph_query`；**某个函数在哪讲、调用了谁、哪个章节讲这个外设**找 `stm32_doc_graph`。

```text
stm32_graph_query("pin", "PA8")           → PA8 → TIM1_CH1（高级定时器）
stm32_graph_query("usart", "USART1")      → TX=PA9 RX=PA10 DMA1_Channel4/5 APB2
stm32_doc_graph("calls", "HAL_UART_Init") → HAL_UART_MspInit（1 跳，p.553）
stm32_doc_graph("peripheral", "TIM3")     → 讲 TIM3 的块 + 器件图通道/DMA/中断
```

Neo4j 没启动时工具只回一句“图谱不可用”，不阻断生成与编译——图谱是**增强**，不是依赖。

### 8.5 为什么先不用 LLM 抽实体

LLM 抽实体/关系贵、不可复现，而你这批语料里 **6033/7514 个块本来就带符号**，
`chunker._symbols()` 和 `loader_hal` 已经把正则抽好的结果写进了 metadata。
所以 P0 只做确定性建图；等需要「参数说明里的自然语言关系」（如“PB6 默认是 I2C1_SCL”）时，
再对新增文档做 LLM 关系抽取，已有层不用改。

---

## 9. Keil MCP Server

### 9.1 VS Code（Copilot）

`week25/.vscode/mcp.json`：

```json
{
  "servers": {
    "keil-mdk": {
      "type": "stdio",
      "command": "d:\\project\\step1\\week23\\.venv\\Scripts\\python.exe",
      "args": [
        "-m", "stm32_rag.mcp_keil_server",
        "--workspace", "E:\\Codex\\2026-11-16\\new-chat\\outputs\\STM32F103C8T6_Base"
      ],
      "env": { "PYTHONPATH": "d:\\project\\step1\\week25" }
    }
  }
}
```

### 9.2 Claude Desktop

```json
{
  "mcpServers": {
    "keil-mdk": {
      "command": "d:\\project\\step1\\week23\\.venv\\Scripts\\python.exe",
      "args": ["-m", "stm32_rag.mcp_keil_server", "--workspace", "E:\\...\\工程目录"],
      "cwd": "d:\\project\\step1\\week25"
    }
  }
}
```

### 9.3 工具清单

`keil_scaffold`、`keil_project_info`、`keil_list_sources`、`keil_read_file`、`keil_write_file`、
`keil_add_source`、`keil_set_defines`、`keil_add_include_paths`、`keil_build`、
`keil_last_errors`、`keil_build_log`。

`--workspace` 是启动参数；换工程改 JSON 重启即可。

---

## 10. Agent 模式评测

`develop` 是规则解析：认“引脚 + 频率 + 关键词”，口语化、缺引脚、同一定时器多通道这类需求它解不出来。
agent 值不值得，要用同一批需求对比，而不是看单次演示。

- **用例**：`stm32_rag/eval_cases_agent.json`，26 条自由说法；含 2 条当前不支持的外设（SPI / CAN），考“该拒绝时能否明确拒绝”
- **判分**：编译通过 **且** 结构匹配 —— 引脚、PWM 的定时器/通道数/频率、占空比、串口实例与波特率、组件文件
- **参数**：`qwen3.8-flash`（`STM32_CHAT_MODEL`），`max_steps=25`，单条 300 s 超时；UV4 全量编译
- **复现**：`python -m stm32_rag agent-eval --max-steps 25 --out agent_eval_report.json`
- 注：首批用默认的 `deepseek-v4.1-flash` 跑到第 6 条额度耗尽（403），改用 `qwen3.8-flash` 跑完 26 条；
  工具轨迹与工程在 `workspaces/agent_eval/`，逐条明细（缺失项 / token / 耗时）在 `stm32_rag/agent_eval_report.json`

### 10.1 汇总

| 指标 | develop | agent |
| --- | --- | --- |
| 编译通过 | 24/26 | **26/26** |
| 需求匹配 | **20/26** | 18/26 |
| 一次成功（无修复、无回退） | — | 16/26 |
| 回退 develop | — | 2/26（递归上限后自动回退） |
| 自修复总轮数 | — | 0 |
| 平均耗时 / 条 | **3.8 s** | 28.3 s |
| 平均 token / 条 | 0 | 25.2 k（26 条共 654 k） |
| 平均 LLM 调用 / 条 | 0 | 5.9 |

### 10.2 谁多解了什么

| 情况 | 用例 | 说明 |
| --- | --- | --- |
| 两边都对 | a01 a02 a03 a05 a06 a07 a08 a09 a13 a17 a19 a20 a22 a23 a24（15 条） | 规则恰好能覆盖的说法 |
| **只有 agent 对** | **a04 · a18 · a21** | a04：要求 TIM3_CH1+CH2，却没给引脚 → agent 查复用表选 PA6/PA7；a18：PWM + 定时翻转 + 串口三合一；a21：TIM1 四通道 250 Hz |
| 只有 develop 对 | a11 a12 a14 a15 a16（5 条） | 需要写 `keys / adc / oled / sched / serial_cmd` 组件字段，agent 漏填 |
| 两边都没对 | a10 a25 a26 | a25 的“脉宽 7.5%”不在规则词表；a10/a26 是多组件组合，agent 也只写了一部分 |

### 10.3 怎么读这组数

1. **agent 的增量在“隐含信息”**：需求没给引脚、要求同一定时器多通道、多外设拼在一起时，agent 会先查复用/时钟再下笔，
   develop 只能在默认引脚上直接失败。那 3 条“只有 agent 对”就是 agent 存在的理由；
2. **agent 的短板在“规格完整性”**：碰到按键/ADC/OLED/命令行/调度，模型经常只写 `uart`/`pwm` 而漏掉组件字段。
   提示词已加“关键词 → 字段”映射表，仍不稳；下一步应在工具侧做前置校验（需求命中组件词但规格缺字段 → 拒绝生成并回给建议字段）；
3. **成本差 7 倍**：agent 28.3 s / 25 k token vs develop 3.8 s / 0 token。所以线上流程保持
   “先 develop，规则解不出再上 agent，agent 失败自动回退 develop”；
4. **自修复本次未被触发**：26 条全部一次编译通过，`auto_fix_build` 0 轮；历史真实案例是 6 条 L6218E 补 TIM 模块（见 5.2）。

---

## 11. 交付物

```text
workspaces/pwm_pa8/
  Core/{Inc,Src,Startup}/        源码与启动文件
  Drivers/{CMSIS,STM32F1xx_HAL_Driver}/
  MDK-ARM/<name>.uvprojx         可直接用 µVision 打开
  MDK-ARM/Objects/<name>.axf     编译产物
  MDK-ARM/build.log              编译日志
  REPORT.md                      需求 / 规格 / 计算核对 / 接线 / 编译结果
  CHANGES.md                     每次需求追加一条变更记录
  AGENT.md                       （仅 agent 模式）工具轨迹与交付说明
```

CubeMX 工程里额外有 `.stm32rag/spec.json`（多轮累积的规格）。

---

## 12. 已知边界与后续路线

### 12.1 边界

1. **外设范围**：GPIO（输出/输入/外部中断）、TIM PWM（同一定时器多通道，共频）、TIM 更新中断、
   USART（发送轮询/DMA，接收中断+命令行）、ADC1（DMA + 滑动平均）、I2C1/I2C2 主发（SSD1306）、
   以及按键 / 蜂鸣器 / 1 ms 调度等应用层组件。SPI、CAN、编码器接口、输入捕获、USART 的 DMA 接收未支持；
   **同一定时器的不同通道必须同频**（共享 PSC/ARR）；
2. **应用层组件只在新建工程模式生成**：CubeMX 增量模式（往 USER CODE 段插）不支持组件文件；
3. **整文件生成**：`generate_application` 适合从零建工程；CubeMX 工程用增量模式，其他已有工程用
   `read_source` + `write_source` + 编译验证；
4. **不回写 `.ioc`**（波特率除外）；
5. **不做烧录**：本机 DFP 的 Flash 算法目录为空，`UV4 -f` 不可用；工程保留 ST-Link 配置，用 µVision 下载即可
   （网盘 `工具软件\STLINK Utility` / `FlyMcu` 可代替）；
6. **器件表只覆盖 F103 常见型号**（C8 / RB / RC / VE / ZE）；
7. **`develop` 的需求解析是规则驱动**：只认“引脚 + 频率/占空比/波特率 + 关键词”，
   组件引脚取“关键词之后出现的引脚”；更自由的说法用 `agent` 或直接给 JSON 规格；
8. **嵌入维度固定**：换 `STM32_EMBED_MODEL` / `STM32_EMBED_DIM` 必须换集合名重建。

### 12.2 后续路线（按板子需求排序）

板子配置：4 个按键（低电平有效）、电位器（ADC）、无源蜂鸣器（PWM）、SSD1306 0.96" OLED（I2C）。

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| P0 | 同一定时器多通道 PWM | ✅ 按 TIM 归并，共享 PSC/ARR |
| P0 | CubeMX 已配外设的追加通道 | ❌ 仍需在 CubeMX 里加通道 |
| P0 | USART 接收 + 环形缓冲 + 命令行 | ✅ `serial_cmd.c`（help/info/echo + 自定义 handler） |
| P0 | 按键消抖组件 | ✅ `key.c`（20 ms 消抖，位图返回） |
| P0 | 蜂鸣器音调表 | ✅ `buzzer.c`（十二平均律音名，非阻塞） |
| P1 | ADC 旋钮（DMA + 滑动平均） | ✅ `adc_knob.c` |
| P1 | SSD1306 I2C 驱动（ASCII） | ✅ `oled_ssd1306.c` + 8x16 字库（来源可用 `STM32_FONT_SRC` 覆盖） |
| P1 | `app.c` 1 ms 节拍调度骨架 | ✅ `app_sched.c`（`Sched_Add` 注册周期任务） |
| P2 | 菜单/页面框架 | ❌ 已有像素/进度条/字符串基础绘图，菜单未做 |
| P2 | 串口自检固件 | ✅ 命令行内置 `help/info/echo`，可注册自定义命令 |
| P2 | 烧录工具接入 | ❌ 按需求暂不做 |

---

## 附录：命令速查

| 命令 | 作用 |
| --- | --- |
| `inspect [--source rm\|ds\|um\|hal\|examples]` | 体检语料并预览切块 |
| `build [--source ...] [--rebuild]` | 建 RAG 索引（Chroma） |
| `stats [--source ...]` | 看集合状态 |
| `ask "问题" [--sources auto] [--with-figures] [--graph-expand]` | 知识库问答（带出处） |
| `eval [--cases ...] [--check]` | 锚点式评测 |
| `ingest <pdf>... [--md] [--pages] [--rebuild]` | PDF → MinerU → 建库 |
| `figures [--show N]` | 只抽插图（调试） |
| `scaffold --dest DIR [--device]` | 生成可编译工程骨架 |
| `codegen "需求" [--json]` | 只解析并打印代码，不落盘 |
| `develop "需求" --workspace DIR` | 生成 + 编译 + 自修复 + 报告 |
| `agent "需求" --workspace DIR` | LLM 编排，失败回退 `develop` |
| `project --workspace DIR` | 查看 `.uvprojx` + `.ioc` 结构 |
| `compile --workspace DIR [--clean]` | 用 UV4 编译（别名 `keil`） |
| `graph --build` / `--pin PA8` / `--usart USART1` / `--timer TIM3` / `--pwm-pins` / `--stats` / `--verify` | 器件知识图谱（Neo4j） |
| `graph-docs --build` / `--symbol HAL_GPIO_Init` / `--chunk ID` / `--expand ID` | 文档关系图（GraphRAG） |
| `graph-docs --calls X` / `--callers X` / `--peripheral TIM1` | 调用链 / 反向调用 / 外设相关章节 |
| `eval-graph [-k 6]` | 跨语料 A/B：期望来源能否进 top-k |
| `agent-eval [--limit N] [--only a03,a21] [--out x.json]` | Agent 评测：成功率 / 自修复 / 回退 / 耗时 / token |
| `mcp --workspace DIR` | 启动 Keil MCP Server |

---

> 面试讲稿（30 秒 / 2 分钟 / 5 分钟版本、技术钩子、Q&A、数字卡）见 [`INTERVIEW.md`](INTERVIEW.md)。
