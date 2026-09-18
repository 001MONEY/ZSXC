# Seq2Seq 中英翻译：0918 课堂整合版总结与两版对比

> 日期：2026-09-18
> 相关目录：`week18/0918`（本文档所在，课堂整合版）、`week18/0914`（作业版）、`week18/teacher_ref`（老师原始代码快照）

## 1. 本次做了什么

老师在 09-18 课上给出了一整套课堂管线代码（`train` / `evaluate` / `predict` / `utils` + 新版 `config` / `model`），与此前的 0914 作业版是两代代码。本次完成三件事：

1. **整合**：把老师这套管线与我们已验证过的 `data.py` / `model.py` 合并成**一套可运行的完整工程**（统一数据接口、损失对齐、词表复用）；
2. **跑通**：完整训练（50 轮上限 + 早停）→ 全量验证集 BLEU 评估 → 交互/批量推理；
3. **对比**：与 0914 作业版做**同句对照**与**全量 BLEU 对比**，用数字回答"新版到底好多少"。

设计取舍：模型架构保留我们验证过的版本（Decoder 隐藏维度 = 2 × Encoder 隐藏维度），未采用老师 `model(1).py` 中新增的 `fc(2H→H)` 投影，留作后续对照实验。

---

## 2. 目录与文件结构

```text
week18/
├─ 0914/              作业版（自研全流程，单层单向 RNN）
├─ 0918/              课堂整合版（本文档所在）
│  ├─ config.py       全局配置：数据路径 / 模型结构 / 训练策略 / 日志目录 + get_device()
│  ├─ data.py         语料读取、分词、Vocabulary、Dataset、create_dataloaders()
│  ├─ model.py        Encoder / Decoder / Seq2Seq + build_model()（按 Config 组装并初始化）
│  ├─ utils.py        set_seed / init_weights / greedy_decode / translate_sentence
│  ├─ train.py        AdamW + TF 线性衰减 + 早停 + TensorBoard + 每 5 轮翻译示例
│  ├─ evaluate.py     验证集 BLEU-4（NLTK corpus_bleu，method1 平滑）+ 示例展示
│  ├─ predict.py      交互式翻译（输入中文 → 输出英文）
│  ├─ raw/            cmn.txt + zh.json / en.json（训练集词表）
│  └─ _demo_translate.py  批量示例推理小脚本（改列表即可换句子）
└─ teacher_ref/       老师原始代码快照（仅作对照，其中已知 bug 未修，不建议直接运行）
```

执行顺序：

```text
data.py → model.py → train.py → evaluate.py → predict.py
```

---

## 3. 与 0914 作业版的关键区别

| 方面 | 0914 作业版 | 0918 课堂整合版 |
| --- | --- | --- |
| 编码器 | 单层**单向** RNN，hidden 256 | 双层**双向** GRU，hidden 64（拼接后 128） |
| 解码器 | 单层 RNN + Linear | 双层 GRU + Linear |
| 权重初始化 | PyTorch 默认 | `weight_hh` 正交 + 其余 Xavier；PAD 嵌入置零 |
| Teacher Forcing | 固定 1.0（全程喂真值） | **1.0 → 0.5 线性衰减**（后期学习纠错） |
| 优化器 | Adam + ReduceLROnPlateau | **AdamW**（权重衰减与梯度更新解耦） |
| 早停 | 无（固定 20 轮） | PATIENCE 8 |
| 日志 | 控制台打印 | TensorBoard 曲线 + tqdm + 每 5 轮翻译示例 |
| 评估 | 人工看几个例子 | **全量验证集 BLEU-4** |
| 推理入口 | `translate.py` | `predict.py` 交互 + 批量小脚本 |

---

## 4. 数据与超参数

**数据**

- `raw/cmn.txt`：26388 对中英平行句；
- 划分：SEED=10、9:1 → 训练 23749 / 验证 2639；
- 词表：仅用训练集构建（防泄漏），中文 11250 / 英文 7462，复用 `raw/zh.json`、`en.json`；
- Batch 64 → 训练 372 个 batch / 验证 42 个 batch。

**模型**

```text
Embedding 128 → 双向 GRU ×2（hidden 64）→ Decoder 单向 GRU ×2（hidden 128）→ Linear(7462)
Dropout 0.3（层间 + 词嵌入）
```

**训练策略**

| 项 | 取值 |
| --- | --- |
| 优化器 | AdamW，lr 1e-3，weight_decay 1e-4 |
| 梯度裁剪 | clip = 1.0 |
| Teacher Forcing | 1.0 → 0.5（按 epoch/总轮数 线性衰减） |
| 早停 | PATIENCE = 8（验证 loss 连续 8 轮不降） |
| 轮数上限 | 50（原配置 70，本次调小） |

---

## 5. 训练结果

**结论一览**：早停于第 **32** 轮；最佳验证 loss **2.8050**（第 24 轮）已保存为 `weights/best.pt`；总耗时约 **11 分钟**（15–24 s/轮，GPU）。

抽样训练曲线：

| Epoch | train | val | tf 比例 |
| --- | --- | --- | --- |
| 01 | 5.3173 | 4.7922 | 1.00 |
| 05 | 3.2981 | 3.4616 | 0.96 |
| 10 | 2.5606 | 3.0705 | 0.91 |
| 16 | 1.9925 | 2.8800 | 0.85 |
| 21 | 1.7080 | 2.8219 | 0.80 |
| **24（best）** | 1.5188 | **2.8050** | 0.77 |
| 32（早停） | 1.1990 | 2.8390 | 0.69 |

曲线特征：

- 验证 loss 前 6 轮快速下降，21 轮后基本走平（~2.80–2.84）；
- 训练 loss 一路降到 1.20 仍在下降 → **过拟合/暴露偏差仍在**；
- 每 5 轮的翻译示例肉眼可见地在变好（"请帮我找一下我的钱包"从乱编到 `Please help me my wallet`）。

---

## 6. 推理效果（best.pt）

**表现好的（常见短句，全部正确）：**

| 中文 | 模型翻译 |
| --- | --- |
| 你叫什么名字？ | `What's your name?` |
| 今天天气很好。 | `It's a nice day.` |
| 我在学英语。 | `I'm learning English.` |
| 这个多少钱？ | `How much does it cost?` |

**翻车/跑偏的（典型错误）：**

| 中文 | 模型翻译 | 问题 |
| --- | --- | --- |
| 我爱你。 | `I owe you.` | 意思翻了 |
| 富士山顶盖满了雪。 | `Mt. Fuji has been stolen.` | 主题词对、动作乱编 |
| 明天我要去北京。 | `Tomorrow I'll go to see tomorrow.` | 开头对、尾巴崩 |
| 请帮我找一下我的钱包。 | `Please help me my wallet and please.` | 开头对、尾巴崩 |

错误规律：**开头主语/短语基本正确，后半段逐渐漂移**——典型的无 Attention 信息瓶颈 + 暴露偏差表现。

---

## 7. 两版对比实验（重点）

### 7.1 同一批句子对照（10 句）

| 中文 | 0914 作业版（val 3.2554） | 0918 课堂版（val 2.8050） |
| --- | --- | --- |
| 我爱你。 | `I'm not busy.` ❌ | `I owe you.` ❌ |
| 你叫什么名字？ | `What are you doing?` ❌ | `What's your name?` ✅ |
| 今天天气很好。 | `It's very cold.` ❌ | `It's a nice day.` ✅ |
| 我在学英语。 | `I'm a member of the tennis club.` ❌ | `I'm learning English.` ✅ |
| 这个多少钱？ | `How much is this pen?` 🟡 | `How much does it cost?` ✅ |
| 请帮我找一下我的钱包。 | `Please show me your phone number.` ❌ | `Please help me my wallet and please.` 🟡 |
| 你看上去不太好。 | `You're very brave.` ❌ | `You don't really very well.` ❌ |
| 汤姆不想让玛丽失望。 | `Tom didn't know what to do.` ❌ | `Tom doesn't want Mary to help Mary.` 🟡 |
| 富士山顶盖满了雪。 | `The boy got in the mirror.` ❌ | `Mt. Fuji has been stolen.` ❌ |
| 明天我要去北京。 | `I'm not a junior.` ❌ | `Tomorrow I'll go to see tomorrow.` 🟡 |

计分：**0914 ≈ 0/10（仅 1 句沾边）；0918 = 4/10 全对 + 4/10 开头对。**

### 7.2 全量验证集 BLEU-4

| 版本 | 架构 | val CE | BLEU-4 |
| --- | --- | --- | --- |
| 0914 作业版 | 单层单向 RNN（hidden 256） | 3.2554 | **0.032** |
| 0918 课堂版 | 双层双向 GRU + TF 衰减 + 早停 | **2.8050** | **0.1297** |

（各自在自身 10% 验证集上评估：0914 用 2638 句，0918 用 2639 句；BLEU 口径均为 NLTK `corpus_bleu` + `method1` 平滑。）

**BLEU 相差约 4 倍。**

---

## 8. 结论与教学点

1. **编码质量的差距是真实的**：单层单向 RNN → 双层双向 GRU，短句准确率发生质变（"你叫什么名字 / 今天天气 / 我在学英语 / 多少钱"从全错到全对）；
2. **CE 与 BLEU 的错位（重要）**：teacher-forcing 的验证 CE 只差 0.45 nats（3.26 → 2.81，约 14%），BLEU 却差 4 倍。因为 CE 是"老师喂着答案"逐步计算的，而生成时要自己走：前面的小误差会被**暴露偏差**滚雪球放大。**结论：评估生成模型不能只看 loss，必须看 BLEU 与实际生成**；
3. **"感觉差不多"是错觉**：两版错误风格相似（都是 fluent but wrong 的"语法通顺但内容乱编"），加上 0914 输出经过 detokenize 排版更像样、失败案例更容易被记住——只有同句对照 + 定量指标才能看出真实差距；
4. **天花板仍在 Attention**：0918 的进步来自"编码变强"，而不是"瓶颈消失"——无 Attention 时整句信息仍压在一个固定向量里，长句后半段依旧崩。这正好作为下一课 Attention 版的 Before 对照组。

---

## 9. 复现步骤

```powershell
cd d:\project\step1\week18\0918

# 训练（本次实测约 11 分钟，GPU）
D:\project\step3\llm\python.exe train.py

# 全量验证集 BLEU（几分钟）
D:\project\step3\llm\python.exe evaluate.py

# 交互式翻译（输入中文回车，q 退出）
D:\project\step3\llm\python.exe predict.py

# 批量示例推理（改 _demo_translate.py 里的句子列表后运行）
D:\project\step3\llm\python.exe _demo_translate.py

# TensorBoard（实时查看 loss 曲线与翻译示例）
D:\project\step3\llm\python.exe -m tensorboard.main --logdir=runs --port 6006
# 浏览器打开 http://localhost:6006
```

注意事项：

- 依赖 `tensorboard`（本次已在 llm 环境安装）；
- 词表复用 `raw/zh.json`、`raw/en.json`，**改动 SEED 会使划分与已有词表不一致**；
- 产物：`weights/best.pt`（最佳权重）、`runs/`（TensorBoard 事件文件）。

---

## 10. 后续计划

- [ ] Attention 版实现后，用同一套流程做**三方对照**（同句 + 全量 BLEU）；
- [ ] 可选实验：老师 `model(1).py` 的 `fc(2H→H)` 投影；beam search 解码；
- [ ] 对比时注意统一展示格式（detokenize）与评估口径。
