# stm32_rag —— STM32F10xxx 参考手册（RM0008）RAG

把 `STM32F10xxx参考手册（英文）.pdf`（1133 页）做成可检索、可溯源的知识库。
嵌入与生成都走云端：`text-embedding-v4` 做嵌入（1024 维），`kimi-k3.1-pro` 做生成与重排，
配置统一从 `week23/.env` 读取，代码里不出现明文 Key。

## 一、和之前那套切法的区别

| 环节 | 通用做法（`RAG2.ipynb`） | 本模块 |
| --- | --- | --- |
| 加载 | `PyMuPDFLoader` 每页一个 Document | 用 1623 条书签当锚点，按 y 坐标把小节切开，一页多个小节也能分开 |
| 清洗 | 中文折行正则 | 英文折行 + 连字符断词 + 页眉页脚（`DocID…` / `RM0008` / `x/1133`）按坐标裁掉 |
| 表格 | 纯文本抽出，位表变成 `31 30 29 … rw rw` | `find_tables()` 转 Markdown 表格，并剔除被表格覆盖的碎文本 |
| 插图 | 完全看不到 | `Figure N.` 题注定位 → 矢量图裁成 PNG → 建成 `kind=figure` 的块；命中后由 `--with-figures` 把原图发给视觉模型 |
| 切块 | `chunk_size=50` 字符 | tiktoken 计数、480 token/块、段落装包、表格原子不切、短节前后合并 |
| 噪声 | 全部入匓 | 整章排除 `Revision history`（24 页改版记录，只会污染召回） |
| 检索 | 纯向量 top-3 | 向量 + BM25 双路 RRF → LLM 重排 → 按小节奏扑（每节先留最相关一块）→ 前后块补全 |
| 型号 | — | 识别 F101~104 / F105~107，优先对应密度章节（同一寄存器两章各有一份） |
| 元数据 | `source/page` | 章节号、章节路径、器件密度、页码区间、块序号、是否含表、寄存器符号 |

## 二、目录

```text
stm32_rag/
  config.py      # 路径、模型、建库/检索参数（改这里就够）
  cleaner.py     # 英文排版清洗
  loader.py      # PDF → 按书签切小节 + 表格转 Markdown
  chunker.py     # 小节 → token 装包切块
  index.py       # Chroma 建库（显式向量、分批、可断点续跑）
  retrieve.py    # BM25 + 向量 + RRF + LLM 重排 + 邻居补全
  answer.py      # Prompt 与带出处的答案
  evaluate.py    # 锚点式评测（命中率 / 覆盖率 / MRR）
  cli.py         # 命令行
  eval_cases.json
```

## 三、命令

```powershell
# 0. 体检：解析全本，打印小节数、块数、章节分布，不写库
python -m stm32_rag inspect

# 1. 先小范围试跑（第 9 章 GPIO 附近），确认切块与检索正常
python -m stm32_rag build --pages 160-200 --collection stm32_smoke --rebuild

# 2. 全量建库（首次解析约 16 分钟，后续走缓存秒开；向量化约 150 秒，可重复执行）
#    --rebuild 会先写进 <集合名>__rebuild，写完后才替换旧集合；解析/嵌入中途失败不会弄挂现有库
python -m stm32_rag build --rebuild

# 3. 看库状态
python -m stm32_rag stats

# 4. 提问（默认带重排；加 --with-figures 把命中插图一并发给视觉模型）
python -m stm32_rag ask "看 Figure 8 的时钟树，SDIO 的时钟怎么来？" --with-figures

# 5. 只看插图抽取（调试用）
python -m stm32_rag figures --show 10

# 6. 评测（--check 会先校验题目里的锚点是否真在语料中）
python -m stm32_rag eval --check
```

## 四、实测数据

| 指标 | 数值 |
| --- | --- |
| 页数 / 体积 | 1133 页 / 12.5 MB |
| 书签 | 1623 条（5 级），过滤图表标题后用作锚点 |
| 解析出的小节 | 1334 个（印刷页 40–1133，目录页与 Revision history 已排除） |
| 转成 Markdown 的表格 | 474 张 |
| 切块 | **2016 块**正文 + **364 块**插图 = 2380 块，538 块含表格 |
| 插图 | 364 张 PNG（20 MB），按 150 dpi 从矢量图裁出 |
| 建库耗时 | 首次解析 979 s + 向量化写入 146 s（8 并发）；之后走解析缓存 |
| 评测（20 题，锚点+MRR） | 全中 **100%**，平均覆盖 97.5%，MRR **1.000**（kimi 重排） |

评测口径：锚点尽量收紧到“寄存器名 + 位域名”（如 `RCC_APB2ENR` + `IOPAEN`），
只提到寄存器名、没有位定义的片段不再能拿到分。评测与 `answer()` 用**同一份证据列表**，
不在评测侧另行截断 `k`，否则分数和实际进 Prompt 的内容对不上。

命令行的实测样例：

```text
> python -m stm32_rag ask "GPIOx_CRL 的地址偏移和复位值分别是多少？"
【答】GPIOx_CRL 的地址偏移为 0x00，复位值为 0x4444 4444 [1]。
【出处】[1] … > 9.2.1 Port configuration register low (GPIOx_CRL) (x=A..G)｜p.170
```

## 五、当成库用

```python
from stm32_rag import Stm32Index, HybridRetriever, answer

index = Stm32Index()
index.build()                     # 首次；之后重复执行只补新增块

retriever = HybridRetriever(index=index)
result = answer("USART 波特率怎么算？", retriever=retriever)
print(result.text)
print("\n".join(result.sources()))
```

只想要检索结果：

```python
from stm32_rag import HybridRetriever
hits = HybridRetriever().search("RCC_APB2ENR", )
for h in hits:
    print(h.label, h.vector_rank, h.bm25_rank, h.rerank_score)
```

## 六、关键参数在哪调

`config.py`：

- `BuildConfig.max_tokens / overlap_tokens`：块大小。手册公式多、表格多，480 token 比较稳；
  调大召回更完整但精度下降。
- `BuildConfig.min_tokens`：小节小于这个值会尝试并入相邻小节（只允许并入同级或更细的小节）。
- `BuildConfig.exclude_chapter_patterns`：整章排除的正则，默认剔除 `Revision history`。
- `RetrieveConfig.vector_k / bm25_k`：两路召回条数。
- `RetrieveConfig.final_k`：进入 Prompt 的**小节数**，默认 6；每个小节再由邻居补全扩成 1~3 块。
- `RetrieveConfig.max_per_section`：单个小节最多保留几块（默认 3）。寄存器位定义表常跨 3 块，
  调到 2 会让 `PLLMUL=0111` 这类编码表掉出去。
- `RetrieveConfig.max_context_tokens`：拼进 Prompt 的正文上限，默认 6000。
- `RetrieveConfig.max_figures_in_context`：上下文里最多保留几张插图（默认 2），
  问题里点名的图不受限；不设上限的话 364 个图块会挤掉寄存器题的正文证据。
- `RetrieveConfig.device_boost`：按问题里的型号（F101~104 / F105~107）优先对应密度章节。
- `RetrieveConfig.use_rerank`：LLM 重排开关。关掉更快，命中率约掉 5 个点。
- `RetrieveConfig.expand_neighbors`：把同一小节的**后一块优先、前一块其次**补进来，
  解决表格/列表被切在块边界的问题。

`.env`（week23）：

- `CHAT_MODEL`：生成与重排用的对话模型，默认 `kimi-k3.1-pro`；换模型只改这一行。
- `DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL`：对话模型走的百炼兼容接口。
- 嵌入模型由 `STM32_EMBED_MODEL` 控制（默认 `text-embedding-v4`），**换它必须换集合名重建**。

## 七、已知边界

1. **图里的文字没做 OCR**：插图只按“题注 + 章节 + 页码”建块，图内的信号名/数值要靠
   `--with-figures` 现场看图；插图多的时候每次问答会多 1~2k token。
2. **跨页表格**：寄存器表一页放不下时会被拆成两张表，第二张没有表头；靠邻居补全缓解。
3. **目录页与改版记录已跳过**：PDF 前 39 页是点线目录，第 32 章是 Revision history，都不建库。
4. **嵌入维度固定**：集合用 1024 维建库，换 `STM32_EMBED_MODEL`/`STM32_EMBED_DIM` 必须换集合名重建。
5. **答案里可能出现“资料中未给出该编码”**：这是故意的——宁可说没给，也不让模型推算位域编码。
