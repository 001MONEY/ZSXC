# 基于 RNN 的 Seq2Seq 中英翻译

## 1. 作业目标

本作业使用 PyTorch 从零实现一个基础的中英机器翻译系统，重点不是追求翻译质量，而是理解 Seq2Seq 的完整工作流程：

```text
中文句子 → Encoder → 语义隐藏状态 → Decoder → 英文句子
```

通过本作业，需要掌握以下内容：

1. 中英文平行语料的读取、分词和编码；
2. 中文词表和英文词表的独立构建；
3. Dataset、DataLoader、动态 Padding 和 train/val 划分；
4. Encoder-Decoder 模型的基本结构；
5. 训练阶段的教师强制；
6. 推理阶段的自回归生成；
7. `<SOS>`、`<EOS>`、`<PAD>` 和 `<UNK>` 的作用；
8. 基础 Seq2Seq 的局限，以及后续 Attention 的必要性。

---

## 2. 项目文件

| 文件                         | 主要作用                              |
| -------------------------- | --------------------------------- |
| `config.py`                | 保存原始语料文件路径等配置                     |
| `data.py`                  | 原始数据处理代码，保留作为参考                   |
| `data_process.py`          | 读取语料、分词、构建词表、编码、解码及保存词表           |
| `dataset.py`               | 构建 Dataset 和 DataLoader，划分训练集与验证集 |
| `model.py`                 | 定义 Encoder、Decoder 和完整 Seq2Seq 模型 |
| `train.py`                 | 完成训练、验证、学习率调度和最佳模型保存              |
| `translate.py`             | 加载最佳模型，进行中文到英文的自回归推理              |
| `train_val_split.json`     | 保存固定的训练集和验证集索引                    |
| `zh_train_word2index.json` | 只根据训练集构建的中文词表                     |
| `en_train_word2index.json` | 只根据训练集构建的英文词表                     |
| `best_seq2seq.pt`          | 验证损失最低时保存的模型 checkpoint           |

整个程序的执行顺序是：

```text
data_process.py
      ↓
dataset.py
      ↓
model.py
      ↓
train.py
      ↓
translate.py
```

---

## 3. 数据处理：data_process.py

### 3.1 读取平行语料

原始 `cmn.txt` 每行包含英文、中文和来源信息：

```text
I love you.\t我爱你。\t来源信息
```

程序读取前两个字段，并将每一组数据保存为：

```python
(中文句子, 英文句子)
```

最终得到：

```python
pairs = [
    ("我爱你。", "I love you."),
    ...
]
```

### 3.2 中英文分词

中文首先通过 `zhconv` 转换为简体中文，再使用 `jieba` 分词：

```text
我是一名学生。 → 我 / 是 / 一名 / 学生 / 。
```

英文使用 NLTK 的 `TreebankWordTokenizer` 分词：

```text
I love you. → I / love / you / .
```

### 3.3 特殊 Token

中英文词表都预留了四个特殊 Token：

| Token   | 编号  | 作用                       |
| ------- | ---:| ------------------------ |
| `<PAD>` | 0   | 将一个 Batch 内不同长度的句子补成相同长度 |
| `<SOS>` | 1   | 表示 Decoder 开始生成句子        |
| `<EOS>` | 2   | 表示一句话结束                  |
| `<UNK>` | 3   | 表示词表中不存在的词               |

可以把 `<SOS>` 和 `<EOS>` 类比为通信协议中的开始和结束标志：它们不是句子本身的内容，而是用于告诉 Decoder 何时开始、何时停止。

### 3.4 为什么中英文要使用两个词表

中文和英文的 Token 完全不同，因此分别建立：

```text
中文词表：中文 Token ↔ 中文编号
英文词表：英文 Token ↔ 英文编号
```

当前训练词表大小为：

```text
中文词表：11,265
英文词表： 7,453
```

两个词表长度不同是正常现象，因为中英文的分词方式、词汇种类和出现频率都不同。

### 3.5 encode 与 decode

`encode()` 将文本转换成编号：

```text
文本 → 分词 → 查 word2index → 编号列表
```

不在词表中的 Token 会被映射为 `<UNK>`。

`decode()` 执行相反过程：

```text
编号列表 → 查 index2word → Token 列表
```

解码遇到 `<EOS>` 时停止，并跳过 `<PAD>`、`<SOS>` 等控制符。

---

## 4. 数据集：dataset.py

### 4.1 数据集划分

原始数据共有：

```text
26,388 对中英句子
```

使用固定随机种子将数据划分为：

```text
训练集索引：23,750
验证集索引： 2,638
```

划分结果保存在 `train_val_split.json` 中。以后再次训练时直接读取相同索引，保证不同实验使用同一份训练集和验证集。

### 4.2 过滤超长句子

为了避免个别超长句子导致整个 Batch 都被补到很长，源序列和目标完整序列的最大长度都设为 30。

当前过滤结果为：

```text
训练集过滤：3 条
验证集过滤：1 条
```

最终实际进入 Dataset 的数量为：

```text
训练样本：23,747
验证样本： 2,637
```

### 4.3 为什么在初始化时完成分词和编码

Dataset 在 `__init__()` 中一次性完成全部分词和编码，并保存 Tensor。

这样 `__getitem__()` 只需要读取已经处理好的结果，避免每个 epoch 都重新执行 `jieba + zhconv`，节省训练时间。

### 4.4 Encoder 和 Decoder 的序列构造

Encoder 输入为：

```text
中文内容 + <EOS>
```

Decoder 的完整英文序列为：

```text
<SOS> + 英文内容 + <EOS>
```

通过错开一位，构造教师强制输入和监督标签。假设完整序列为：

```text
<SOS> I love you <EOS>
```

则：

```text
decoder_input：<SOS> I    love you
target：          I  love you  <EOS>
```

模型在每个位置学习“根据前面的正确单词预测下一个单词”。

### 4.5 动态 Padding

每个 Batch 只补到当前 Batch 中最长句子的长度，而不是把全量数据都补到统一长度。

`collate_fn` 返回：

```python
{
    "src": src_batch,
    "src_lengths": src_lengths,
    "decoder_input": decoder_batch,
    "target": target_batch,
}
```

`src_lengths` 在补 PAD 之前记录，用于让 Encoder 跳过句尾的 `<PAD>`。

---

## 5. 模型：model.py

### 5.1 Encoder

Encoder 的计算过程为：

```text
中文编号
  ↓ Embedding
中文词向量
  ↓ Dropout
  ↓ RNN
最终隐藏状态 hidden
```

其核心递推关系可以写成：

$$
h_t = \operatorname{RNN}(x_t, h_{t-1})
$$

输入和输出形状为：

```text
src：      [batch_size, src_length]
embedded： [batch_size, src_length, embedding_dim]
hidden：   [num_layers, batch_size, hidden_size]
```

当前 `hidden_size=256`，因此 Encoder 要把整句中文的信息压缩到最后的 256 维隐藏状态中。

`pack_padded_sequence()` 用于跳过 Padding，使最终隐藏状态对应每句话真实的 `<EOS>` 位置，而不是 `<PAD>` 位置。

### 5.2 Decoder

Decoder 接收 Encoder 的最终隐藏状态作为初始状态：

```text
Encoder hidden
      +
英文输入 Token
      ↓
Decoder RNN
      ↓
Linear
      ↓
英文词表上每个词的 logits
```

输出形状为：

```text
logits：[batch_size, target_length, target_vocab_size]
```

当前英文词表大小为 7,453，因此 Decoder 在每个时间步都会输出 7,453 个分数。

### 5.3 Dropout

Encoder 和 Decoder 都在 Embedding 后使用：

```python
nn.Dropout(0.2)
```

训练时随机屏蔽约 20% 的词向量特征，减轻过拟合；验证和推理时通过 `model.eval()` 自动关闭。

当前 RNN 只有一层。PyTorch 的 `nn.RNN(dropout=...)` 只作用于多层 RNN 的层间连接，因此这里必须显式对词向量使用 Dropout。

---

## 6. 训练：train.py

### 6.1 主要超参数

```text
batch_size：       32
embedding_dim：   128
hidden_size：     256
num_layers：        1
epoch：             20
初始学习率：      0.001
dropout：          0.2
梯度裁剪：         1.0
```

### 6.2 教师强制

训练时将完整的 `decoder_input` 一次交给 Decoder：

```python
logits = model(src, src_lengths, decoder_input)
```

由于 `decoder_input` 中是正确的前一个英文词，因此属于教师强制。

### 6.3 损失函数

使用交叉熵损失：

```python
nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD_INDEX)
```

`<PAD>` 只是为了对齐 Batch，不是需要预测的内容，因此不参与损失计算。

损失可以表示为：

$$
L=-\frac{1}{N}\sum_{t=1}^{N}\log p(y_t\mid y_{<t},h)
$$

其中 $h$ 是 Encoder 的最终隐藏状态，$y_{<t}$ 是目标句中位置 $t$ 之前的正确单词。

### 6.4 梯度裁剪

普通 RNN 可能发生梯度爆炸，因此反向传播后执行：

```python
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

它将整体梯度范数限制在 1.0 以内。

### 6.5 学习率调度

使用 `ReduceLROnPlateau` 监视验证损失：

```text
验证损失继续降低 → 保持学习率
验证损失连续多轮不改善 → 学习率乘 0.5
最低学习率 → 0.00001
```

本次训练中学习率变化为：

```text
Epoch 01～13：0.001
Epoch 14～18：0.0005
Epoch 19～20：0.00025
```

第一次降低学习率后，验证损失从第 14 轮的 3.3059 改善到第 16 轮的 3.2554，说明调度器发挥了作用。

### 6.6 最佳模型保存

每轮验证结束后，只在 `val_loss` 创造新低时保存模型：

```text
best_seq2seq.pt
```

checkpoint 中包含：

```text
model_state_dict
embedding_dim
hidden_size
num_layers
dropout
```

这样推理时可以恢复与训练阶段完全一致的模型结构。

---

## 7. 训练结果分析

### 7.1 未使用 Dropout 和学习率调度

第一次训练的最佳结果为：

```text
Epoch 06/20
train_loss=2.3886
val_loss=3.2319
```

第 6 轮后训练损失继续下降，但验证损失持续上升：

```text
Epoch 20
train_loss=1.2055
val_loss=3.7162
```

这说明模型对训练集拟合得越来越好，但对未见数据的表现越来越差，出现明显过拟合。

### 7.2 加入 Dropout 和学习率调度

第二次训练的最佳结果为：

```text
Epoch 16/20
train_loss=2.0071
val_loss=3.2554
lr=0.000500
```

两次实验对比：

| 实验            | 最佳轮次 | 最佳 train loss | 最佳 val loss | 曲线表现                |
| ------------- | ----:| -------------:| -----------:| ------------------- |
| 基础版本          | 6    | 2.3886        | 3.2319      | 第 7 轮后验证损失持续恶化      |
| Dropout + 调度器 | 16   | 2.0071        | 3.2554      | 验证损失长期稳定在 3.25～3.30 |

第二次实验的最佳验证损失并没有超过第一次，差值为 0.0235；但验证曲线更稳定，过拟合速度明显减慢，并且学习率下降后能够再次刷新最佳结果。

因此更准确的结论是：

> Dropout 和学习率调度改善了训练稳定性与抗过拟合能力，但没有从根本上解决基础 RNN Seq2Seq 的翻译能力限制。

---

## 8. 推理：translate.py

### 8.1 加载模型

推理脚本依次完成：

```text
加载中文词表和英文词表
          ↓
读取 checkpoint 中的模型参数
          ↓
重新创建 Encoder 和 Decoder
          ↓
load_state_dict() 加载训练权重
          ↓
model.eval() 关闭 Dropout
```

### 8.2 自回归生成

推理阶段没有英文正确答案，因此不能再使用教师强制，而是从 `<SOS>` 开始逐词生成：

```text
<SOS>
  ↓
预测第一个英文词 y₁
  ↓
把 y₁ 作为下一步输入
  ↓
预测 y₂
  ↓
不断重复，直到预测出 <EOS>
```

这就是自回归：每一步的输入来自模型上一步自己的预测。

训练与推理的区别可以概括为：

| 阶段  | Decoder 上一步输入 | 特点                |
| --- | ------------- | ----------------- |
| 训练  | 真实英文单词        | 教师强制，可以并行计算整个目标序列 |
| 推理  | 模型自己预测的单词     | 自回归，必须按时间步逐个生成    |

### 8.3 解码英文

模型生成的是英文词表编号：

```text
[编号1, 编号2, 编号3, ..., <EOS>]
```

英文词表的 `decode()` 将其恢复为 Token，遇到 `<EOS>` 停止；`TreebankWordDetokenizer` 再将英文缩写和标点恢复成正常格式：

```text
I / do / n't / know / . → I don't know.
```

---

## 9. 实际翻译结果与现象

当前模型能够完成端到端推理，但翻译质量不稳定。例如：

```text
我想你了
→ I don't want to fail my exams.

今天天气很好，非常适合逛公园
→ What a wonderful, you can download it.
```

对标点还表现出明显敏感性：

```text
我不想挂科
→ I'm not sure Tom.

我不想挂科。
→ I don't want to fail my exams.
```

训练语料中存在完全一致的：

```text
我不想挂科。 → I don't want to fail my exams.
```

因此，加上句号后翻译正确，说明模型记住了训练样本；去掉句号后立即失败，则说明它对训练格式依赖较强，泛化能力不足。

较长输入：

```text
今天天气很好，非常适合逛公园
```

分词后包含训练词表中没有的“非常适合”和“逛公园”，它们会被转换为 `<UNK>`，导致后半句的重要语义丢失。

---

## 10. 基础 Seq2Seq 的局限

### 10.1 固定长度语义瓶颈

当前 Encoder 只把最后一个隐藏状态传给 Decoder：

```text
x₁ → x₂ → x₃ → ... → xₙ → 最终 hidden → Decoder
```

无论输入句子多长，全部信息都必须压缩到一个 256 维向量中。句子越长，前面信息越容易丢失。

### 10.2 教师强制与推理不一致

训练时 Decoder 总能看到正确的前一个英文词；推理时只能看到自己的预测。一旦前面预测错误，后面的错误可能不断累积。

### 10.3 Decoder 可能更像英文语言模型

Decoder 在训练中容易学会“常见英文词后面通常接什么”，却不一定充分利用 Encoder 的中文语义。

因此可能生成语法结构看似合理、但和中文意思无关的句子。

### 10.4 `<UNK>` 会丢失具体语义

词级词表无法表示未登录词，所有未知 Token 都被映射为同一个 `<UNK>`。模型无法区分两个不同的未知词分别代表什么。

### 10.5 语料重复和一对多

同一个中文句子可能对应多个英文翻译，同一个英文句子也可能重复对应多种中文表达。这会增加监督目标的不确定性，并使模型偏向高频英文模板。

---

## 11. 为什么下一步需要 Attention

基础 Seq2Seq 的主要问题是 Decoder 只能获得 Encoder 最后一个隐藏状态。

```text
基础 Seq2Seq：
整句中文 → 一个最终 hidden → Decoder
```

Attention 会保留 Encoder 在所有位置的输出：

```text
中文每个位置的 Encoder 输出
             ↓
Decoder 每生成一个词，都计算当前应该关注哪些中文位置
```

例如翻译“今天天气很好，非常适合逛公园”时：

```text
生成 weather → 重点关注“天气”
生成 suitable → 重点关注“适合”
生成 park → 重点关注“公园”
```

Decoder 不必再依赖一个固定向量保存整句话，这正是下一节 Attention 要解决的核心问题。

---

## 12. 运行方法

### 12.1 构建数据集和词表

```powershell
python dataset.py
```

### 12.2 训练模型

```powershell
python train.py
```

### 12.3 进行翻译

```powershell
python translate.py
```

运行推理脚本后：

```text
推理设备： cuda
输入中文进行翻译，直接回车退出。
中文：我不想挂科。
英文：I don't want to fail my exams.
```

---

## 13. 总结

本作业完成了一个基础 RNN Seq2Seq 中英翻译系统，从原始文本一直走到了模型推理：

```text
平行语料
  ↓
分词与词表
  ↓
encode
  ↓
Dataset 和动态 Padding
  ↓
Encoder-Decoder
  ↓
教师强制训练
  ↓
自回归推理
  ↓
decode 得到英文
```

虽然翻译效果有限，但它已经完整说明了 Seq2Seq 如何处理“输入序列到输出序列”的问题，也通过实际错误展示了固定隐藏状态、未知词和训练/推理差异带来的局限。

本节的重点是理解 Seq2Seq 的基本机制，而不是得到成熟的翻译系统。下一节 Attention 将在此基础上，让 Decoder 在生成每个词时动态关注输入序列的不同位置。
