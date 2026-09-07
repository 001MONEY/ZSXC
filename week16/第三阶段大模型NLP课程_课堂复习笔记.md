# 第三阶段大模型 / NLP 课程复习笔记

> 课堂日期：2026-09-04  
> 主线：为什么语言需要专门建模 → 文本预处理 → 中文分词 → 子词 Tokenizer → 数字编码 → 词向量 → Word2Vec。  
> 整理依据：《第三阶段大模型课程》的完整对话文本与本次可读取的课堂截图。保留课堂例子和电子信息工程类比；对容易产生误解的表述作了必要校正。公式补全、伪代码和“后续衔接”用于复习，不代表老师已逐项展开讲授。

## 目录

1. [先看整堂课的主线](#s1)
2. [DNN 局限与时序数据的三大差异](#s2)
3. [NLP 定义、自然语言特点与四次演进](#s3)
4. [中文分词的三代方法](#s4)
5. [HMM：把分词变成概率状态机](#s5)
6. [Viterbi：寻找最可能的完整状态路径](#s6)
7. [子词分割：BPE、WordPiece、Unigram](#s7)
8. [工程工具：jieba 与 Hugging Face tokenizers](#s8)
9. [课堂代码：文本到 Token ID](#s9)
10. [词向量、One-Hot、余弦相似度与分布假设](#s10)
11. [Word2Vec：CBOW 与 Skip-gram](#s11)
12. [CBOW 的完整训练流程](#s12)
13. [Embedding 矩阵：词向量到底存在哪里](#s13)
14. [为什么经典 CBOW 的隐藏层不加 ReLU](#s14)
15. [电子信息工程类比与后续课程衔接](#s15)
16. [复习速查与自测](#s16)
17.  [CBOW训练脚本](#s17)

<a id="s1"></a>

## 1. 先看整堂课的主线

**今天主要解决的问题：怎样把一段变长、有顺序、依赖上下文的文字，变成神经网络能使用的向量？**

```text
语言：变长 + 有序 + 元素相关 + 离散符号
    ↓
先决定按什么单位处理文本
    ├─ 中文词级分词：词典 / HMM → 词
    └─ 子词 Tokenizer：BPE / WordPiece / Unigram → token
    ↓
建立词表：token ↔ ID
    ↓
Embedding：ID → 连续向量
    ↓
Word2Vec：借上下文预测任务训练这张向量表
    ├─ CBOW：上下文 → 中心词
    └─ Skip-gram：中心词 → 上下文
    ↓
得到能表达部分语义关系的静态向量
    ↓
后续：序列模型 / Attention / Transformer → 上下文化表示
```

这里有两条并行的逻辑：

| 逻辑    | 要解决的问题                   |
| ----- | ------------------------ |
| 文本预处理 | 文本切成什么单位？每个单位的 ID 是什么？   |
| 表示学习  | 每个 ID 对应什么向量？向量中的语义从哪里来？ |

**HMM 是一种分词路线，不是 BPE 的前置步骤；Word2Vec 是学习词向量的一种方法，也不是所有大模型都必须先执行的步骤。** 现代语言模型通常直接在自身的训练目标下学习 token embedding。

<a id="s2"></a>

## 2. DNN 局限与时序数据的三大差异

### 2.1 本课中的 DNN 指什么

课堂这里主要指普通前馈全连接网络（MLP）：

$$
\mathbf h=\sigma(W\mathbf x+\mathbf b),\qquad \mathbf x\in\mathbb R^n
$$

它很适合固定长度、各位置含义明确的输入，例如：

```text
[年龄, 身高, 体重, 血压] → 分类结果
```

严格说，DNN 是“深度神经网络”的泛称，RNN、Transformer 等也可以属于深度网络。此处比较的是普通全连接结构与专门的序列建模结构。

### 2.2 三大差异

| 课堂重点  | 时序 / 语言数据的特点    | 普通全连接网络的困难           | 后续处理思路               |
| ----- | --------------- | -------------------- | -------------------- |
| 元素不独立 | 后面的元素往往依赖前文     | 没有内置的跨步状态传递机制        | 状态递推、上下文建模、Attention |
| 长度不固定 | 不同句子的 token 数不同 | 固定输入层不能直接接收任意长度的展平序列 | padding、mask、池化、序列结构 |
| 顺序有意义 | 相同元素换序可能改变含义    | 缺少适合序列的结构约束与参数共享方式   | 递推顺序、位置表示            |

**差异一：序列内部有依赖。**

```text
我去了银行取____
```

后一个词的分布依赖前文，语言建模常写成：

$$
P(x_t\mid x_1,x_2,\ldots,x_{t-1})
$$

需要分清两个层次：训练集中“不同句子能否近似视为独立样本”，与“一句话内部的词是否独立”，是两个问题。**MLP 本身并不要求输入特征互相独立，也可以学习特征间的关系。**

**差异二：长度可变。**

$$
X=(x_1,x_2,\ldots,x_T),\qquad T\text{ 随样本变化}
$$

固定矩阵形状的 batch 通常需要补齐：

```text
[2, 5, 8]       → [2, 5, 8, 0, 0]
[2, 7, 9, 6, 4] → [2, 7, 9, 6, 4]

0 是本例约定的 PAD ID；mask 标记哪些位置有效。
```

**差异三：顺序承载含义。**

```text
狗 咬 人
人 咬 狗
```

固定位置的 MLP 对输入顺序也敏感，不能说它“完全看不见顺序”。困难在于：它没有天然适合变长序列、跨位置复用规律的机制；如果先把词变成无序词袋，顺序则已经在预处理阶段丢失。

### 2.3 语言额外增加了“符号表示”问题

传感器电压、图像像素本来就有数值表示；文字虽能存成字符编码，但字符编码的大小不等于语义关系。

```text
字符编码：解决存储与传输
Token ID：解决词表索引
Embedding：提供可训练的连续表示
```

因此本课先解决“文本怎么表示”，再为后续“如何建模上下文”打基础。

<a id="s3"></a>

## 3. NLP 定义、自然语言特点与四次演进

### 3.1 什么是 NLP

**NLP（Natural Language Processing，自然语言处理）**研究如何让计算机处理、分析、理解和生成自然语言。它是一组任务、方法与工具，不是某一种网络。

| 任务   | 输入 → 输出      | 例子             |
| ---- | ------------ | -------------- |
| 文本分类 | 文本 → 类别      | 情感分析、垃圾邮件识别    |
| 序列标注 | 序列 → 逐位置标签   | 分词、词性标注、命名实体识别 |
| 机器翻译 | 一种语言 → 另一种语言 | 中文 → 英文        |
| 问答   | 问题及相关信息 → 答案 | 文档问答           |
| 文本生成 | 条件 / 提示 → 文本 | 摘要、对话、写作       |

### 3.2 自然语言为什么难

| 特点        | 例子                   | 对模型的要求     |
| --------- | -------------------- | ---------- |
| 歧义性、多义性   | “苹果”可以是水果，也可以指公司     | 结合语境判别词义   |
| 上下文依赖     | “小明告诉小李，他明天不来了”中的“他” | 跨位置、跨句关联   |
| 省略与隐含信息   | “吃了吗？”省略了主语和宾语       | 利用语境与常识    |
| 容错性与表达灵活性 | 错别字、口语、省略、语序变化       | 不能只按严格文法匹配 |
| 开放性       | 新人名、新术语、网络词不断出现      | 应对词表外内容    |
| 长距离依赖     | 代词可能指向很早之前的人物        | 不只看相邻几个词   |

Python、C++、Verilog 等形式语言的语法和语义规则由规范定义；自然语言的理解常常还依赖语用、背景与常识。即使有上下文，也不保证所有歧义都能唯一消除。

### 3.3 NLP 技术的四次演进

这是一种课堂用的概括，各路线在实际应用中长期并存。

| 阶段                | 核心方式            | 典型方法                                    | 主要局限 / 推动力         |
| ----------------- | --------------- | --------------------------------------- | ------------------ |
| ① 规则系统            | 人写词典、文法、知识与推理规则 | 词典匹配、专家系统                               | 规则维护成本高，新情况覆盖困难    |
| ② 统计机器学习          | 从语料估计概率或学习决策边界  | N-gram、朴素贝叶斯、SVM、HMM、CRF；配合词袋、TF-IDF 特征 | 依赖特征设计，表示能力有限      |
| ③ 神经网络 / 深度学习 NLP | 学习连续表示及任务模型     | Word2Vec、RNN、LSTM、GRU、Seq2Seq、Attention | 数据需求、长距离依赖、任务迁移等问题 |
| ④ 预训练模型与大模型       | 大规模预训练，再适配多种任务  | Transformer、BERT、GPT 等                  | 计算与数据成本、可靠性及适配问题   |

Word2Vec 本身是浅层模型，放在第三阶段是因为它推动了神经表示学习，不是因为网络层数很深。

**课堂问题：“一个 if 语句也算 AI 吗？”**

单独一条 `if` 不能据此认定为 AI。但知识库配合推理引擎的专家系统，属于人工智能发展史中的符号主义路线。判断重点是系统完成什么任务、如何表示知识与推理，而不是有没有 `if`、规则有多少条。

<a id="s4"></a>

## 4. 中文分词的三代方法

### 4.1 为什么中文需要分词

中文通常没有显式词间空格：

```text
我爱自然语言处理
```

可能按不同粒度切成：

```text
我 / 爱 / 自然 / 语言 / 处理
我 / 爱 / 自然语言 / 处理
我 / 爱 / 自然语言处理
```

切分受任务、标注规范和词表影响，并非任何场景都只有一种正确粒度。英文有空格，也仍需处理标点、缩写、复合词等问题。

### 4.2 按本课口径记三代

| 代际          | 核心思想         | 代表                    | 主要目标             |
| ----------- | ------------ | --------------------- | ---------------- |
| 第一代：规则 / 词典 | 用词典和匹配规则找词   | FMM、BMM、双向最大匹配        | 切出已有词            |
| 第二代：统计      | 利用语料概率选择切分   | HMM；相关方法还有 CRF        | 消歧、识别部分未登录词      |
| 第三代：子词分割    | 学习可复用片段及有限词表 | BPE、WordPiece、Unigram | 平衡词表规模、序列长度与覆盖能力 |

这里的“三代”是本课从传统分词走向现代 Tokenizer 的教学路线，不是中文分词领域唯一的历史分期。子词不要求对应语言学意义上的完整词，也不保证每个片段都有独立语义。

### 4.3 第一代：最大匹配

**FMM（正向最大匹配）**：从左往右，当前位置优先选择词典中能匹配的最长词。

```text
当前位置 = 句首
while 尚有字符:
    从允许的最大词长开始尝试匹配
    如果词典命中：输出该词，位置向后移动
    如果一直不命中：退回单字并前进
```

**BMM（逆向最大匹配）**：从右往左做类似处理，再恢复输出顺序。

**双向最大匹配**：分别运行 FMM、BMM；结果冲突时，按较少词数、较少单字等启发式规则选择，具体规则取决于实现。

```text
南京市长江大桥 → 南京市 / 长江大桥
```

最大匹配是局部贪心策略，并不保证语义正确。例如“乒乓球拍卖完了”可能在“球拍 / 卖”和“球 / 拍卖”之间发生歧义。

这引出第二代的问题：**能不能用数据给整条切分路径打分，而不只问词典里有没有？**

<a id="s5"></a>

## 5. HMM：把分词变成概率状态机

### 5.1 先把分词改写成 BMES 序列标注

| 标签       | 含义     | 例子         |
| -------- | ------ | ---------- |
| B：Begin  | 多字词的词首 | “鱼香肉丝”的“鱼” |
| M：Middle | 多字词的中间 | “香”“肉”     |
| E：End    | 多字词的词尾 | “丝”        |
| S：Single | 单字成词   | 单独成词的“的”   |

```text
单字词：S
双字词：B E
三字词：B M E
四字词：B M M E
```

保留课堂三句已分词语料及其标注：

```text
今天 / 的 / 天气 / 很热
今 天   的   天 气   很 热
B  E    S    B  E    B  E

今天 / 的 / 鱼香肉丝 / 味道 / 很好
今 天   的   鱼 香 肉 丝   味 道   很 好
B  E    S    B  M  M  E    B  E    B  E

你 / 今天 / 吃饭 / 了 / 吗
你   今 天   吃 饭   了   吗
S    B  E    B  E    S    S
```

这套标注遵循课堂给定切分，例如“很热”在此被作为一个词；换标注规范，标签也会改变。

于是：

$$
\text{寻找词边界}\quad\Longleftrightarrow\quad\text{给每个汉字预测 BMES 标签}
$$

### 5.2 HMM 五元组

HMM：Hidden Markov Model，隐马尔可夫模型。

$$
\lambda=(\mathcal S,\mathcal O,\pi,A,B_{\mathrm{emit}})
$$

课堂常把最后一项直接记作 $B$；这里加下标以区别 BMES 中的词首标签 B。

| 成分                           | 定义                       | 中文分词中的对应       |
| ---------------------------- | ------------------------ | -------------- |
| $\mathcal S$                 | 隐藏状态集合                   | $\{B,M,E,S\}$  |
| $\mathcal O$                 | 观测符号集合                   | 可观察到的汉字等符号     |
| $\pi_i$                      | $P(s_1=i)$               | 第一个字对应各状态的概率   |
| $A=[a_{ij}]$                 | $P(s_t=j\mid s_{t-1}=i)$ | 标签之间的转移概率      |
| $B_{\mathrm{emit}}=[b_j(c)]$ | $P(o_t=c\mid s_t=j)$     | 给定标签时观察到某个字的概率 |

注意：$\mathcal O$ 是“可能出现的符号集合”，而 $o_{1:T}$ 是某一次实际观察到的字符序列。

```text
隐藏状态：s₁ ──→ s₂ ──→ s₃ ──→ … ──→ sT
          ↓      ↓      ↓              ↓
观测字符：o₁     o₂     o₃             oT
```

预测时，我们只看到字，不知道标签，所以叫“隐藏状态”。训练语料中若已有人工分词，就能转换出标签来监督估计参数。

### 5.3 两个关键假设

**一阶马尔可夫假设：下一状态只依赖当前状态。**

$$
P(s_t\mid s_1,\ldots,s_{t-1})=P(s_t\mid s_{t-1})
$$

**观测条件独立假设：给定当前状态，当前观测的分布不再依赖其他时刻。**

$$
P(o_t\mid s_{1:T},o_{1:t-1})=P(o_t\mid s_t)
$$

这是模型的简化假设，并不是说真实语言只有一阶关系。它也不意味着观测到的汉字序列本身必然满足一阶马尔可夫性。

### 5.4 BMES 的合法转移

| 当前状态 | 合法下一状态 | 理由              |
| ---- | ------ | --------------- |
| B    | M、E    | 多字词开始后，要继续或结束   |
| M    | M、E    | 词内部继续或结束        |
| E    | B、S    | 一个词已结束，下一个词重新开始 |
| S    | B、S    | 单字词已结束，下一个词重新开始 |

标准完整句子还应满足：句首只能是 B 或 S，句尾只能是 E 或 S。

$$
\pi_M=\pi_E=0
$$

非法转移不是“概率较低”，而是按 BMES 结构约束置为 0。平滑也应保留这些结构性零值。

### 5.5 初始概率：统计每句的第一个状态

三句课堂语料的开头依次为 B、B、S，因此：

$$
\pi=(\pi_B,\pi_M,\pi_E,\pi_S)
=\left(\frac23,0,0,\frac13\right)
$$

课堂图中的 0.66、0.34 可作为近似示意；按这三句精确计算是 $2/3$ 与 $1/3$。

### 5.6 状态转移概率：计数，再按行归一化

**下表是次数，不是概率。** 行是现态，列是次态；只统计句内相邻标签，不跨句连边。

| 现态 → 次态 | B   | M   | E   | S   | 出边总次数 |
| ------- | ---:| ---:| ---:| ---:| -----:|
| B       | 0   | 1   | 8   | 0   | 9     |
| M       | 0   | 1   | 1   | 0   | 2     |
| E       | 4   | 0   | 0   | 3   | 7     |
| S       | 3   | 0   | 0   | 1   | 4     |

$$
a_{ij}=\frac{C(i\to j)}{\sum_k C(i\to k)}
$$

$$
A=
\begin{bmatrix}
0&1/9&8/9&0\\
0&1/2&1/2&0\\
4/7&0&0&3/7\\
3/4&0&0&1/4
\end{bmatrix},\qquad \sum_j a_{ij}=1
$$

分母是“状态 $i$ 有后继时的转移总数”。句尾没有句内下一状态，因此不能直接拿语料里所有 $i$ 的出现次数作分母。本例没有额外建立 END 状态。

### 5.7 发射概率：给定状态，看到哪个字

$$
b_j(c)=P(o_t=c\mid s_t=j)
=\frac{C(s_t=j,o_t=c)}{C(s_t=j)}
$$

按照这三句语料，B 状态共 9 次，其中“今”对应 B 共 3 次：

$$
P(\text{今}\mid B)=\frac39=\frac13
$$

但三次“今”全都标为 B，所以同一份样本统计还得到：

$$
P(B\mid\text{今})=\frac33=1
$$

**两者条件方向不同，不能混用。** HMM 的发射概率采用前者；后者不是本模型的发射表。

对每个固定状态，发射分布在观测符号集合上归一化：

$$
\sum_{c\in\mathcal O}b_j(c)=1
$$

实际语料会出现未见过的“状态—字符”组合，需要未知字符处理或平滑，否则零概率会让整条路径直接归零。三句话只够演示统计流程，不足以训练可靠的分词器。

### 5.8 HMM 的训练与推断要分开

```text
训练（课堂的有标注版本）：
人工分词语料
→ 转成字符及 BMES 标签
→ 统计句首、状态转移、状态—字符共现次数
→ 归一化，必要时平滑
→ 得到 π、A、发射表

推断：
新句子的字符序列
→ 用 π、A、发射表给候选路径评分
→ Viterbi 找最优 BMES 路径
→ 按标签还原词边界
```

HMM 不一定靠神经网络反向传播训练。本课的有标签版本可直接用频数估计；无标签参数学习是另一类问题。

<a id="s6"></a>

## 6. Viterbi：寻找最可能的完整状态路径

### 6.1 一条路径如何评分

观测序列固定为 $o_{1:T}$，某条状态路径为 $s_{1:T}$：

$$
P(s_{1:T},o_{1:T}\mid\lambda)
=\pi_{s_1}b_{s_1}(o_1)
\prod_{t=2}^{T}a_{s_{t-1},s_t}b_{s_t}(o_t)
$$

记忆顺序：

```text
初始概率 × 第一个字的发射概率
         × 转移概率 × 第二个字的发射概率
         × 转移概率 × 第三个字的发射概率 × …
```

我们要找：

$$
\hat s_{1:T}
=\arg\max_{s_{1:T}}P(s_{1:T}\mid o_{1:T})
=\arg\max_{s_{1:T}}P(s_{1:T},o_{1:T})
$$

第二个等号成立，是因为给定观测后，贝叶斯公式的分母 $P(o_{1:T})$ 对所有候选状态路径都相同。

### 6.2 为什么不能逐字贪心

当前一个字最合适的标签，未必属于全句最优路径；它还必须与前后的标签合法衔接。

暴力枚举最多需要考察约 $4^T$ 条标签组合。Viterbi 则在每一步、每个状态上只保留一条最优历史。

**为什么能丢掉其他历史？** 如果两条历史都到达同一时刻的同一状态，在 HMM 假设下，它们未来的可选转移与评分规则相同。已有分数较差的那条，接上任何共同后缀也不会反超。

### 6.3 动态规划公式

定义 $\delta_t(j)$：解释前 $t$ 个观测、且在时刻 $t$ 结束于状态 $j$ 的最佳路径联合概率。

初始化：

$$
\delta_1(j)=\pi_jb_j(o_1)
$$

递推：

$$
\delta_t(j)=b_j(o_t)\max_i\left[\delta_{t-1}(i)a_{ij}\right]
$$

保存前驱：

$$
\psi_t(j)=\arg\max_i\left[\delta_{t-1}(i)a_{ij}\right]
$$

终止与回溯：

$$
\hat s_T=\arg\max_{j\in\{E,S\}}\delta_T(j),\qquad
\hat s_{t-1}=\psi_t(\hat s_t)
$$

最后必须回溯，才能得到完整路径；仅知道终点最大分数并不够。

### 6.4 实现时通常用对数

长句中连续相乘容易数值下溢，取对数后变成加法：

$$
D_t(j)=\log b_j(o_t)+\max_i\left[D_{t-1}(i)+\log a_{ij}\right]
$$

```text
将非法转移的 log 概率设为 -∞
对每个状态 j：
    score[1, j] = log π[j] + log emit[j, 第一个字]

for t = 2 ... T:
    for 当前状态 j:
        比较所有合法前驱 i 的 score[t-1, i] + log A[i, j]
        记录最大值对应的前驱 i
        再加上 log emit[j, 当前字]

在末尾 E、S 中选最大分数
沿前驱指针从后往前回溯
按 BMES 输出词：S 单独输出，B 开始，M 延续，E 结束
```

状态数为 $K$ 时，普通实现的时间复杂度是 $O(TK^2)$；本课 $K=4$。核心是“每个位置、每个状态保留最优历史”，不是“每个位置只保留一个状态”。上述递推可对照 [Jurafsky 与 Martin 的 HMM 教材附录](https://web.stanford.edu/~jurafsky/slp3/A.pdf)。

### 6.5 与数电、通信的对应

| 熟悉的概念            | 对应到 HMM / Viterbi   |
| ---------------- | ------------------- |
| 状态图              | BMES 合法转移图          |
| 现态—次态表           | 转移计数表、概率矩阵 $A$      |
| 给定现态和输入确定次态      | HMM 用当前状态条件下的分布描述次态 |
| 输出逻辑             | HMM 用发射分布描述观测       |
| 格形图 trellis、路径度量 | 不同时刻的候选状态和累计对数概率    |
| 保留幸存路径、回溯        | Viterbi 的最优前驱与路径恢复  |

你提到的卡诺图联系，准确落点是：**这里更像状态转移表；卡诺图用于化简布尔函数，HMM 表用于估计条件概率。**

“HMM ≈ 概率化有限状态机”是有效类比，但普通 Moore / Mealy 状态机本来也有输出，HMM 的关键是隐藏状态与概率性的转移、观测关系。

<a id="s7"></a>

## 7. 子词分割：BPE、WordPiece、Unigram

### 7.1 为什么从词走向子词

词级词表会遇到 **OOV（Out Of Vocabulary，词表外词 / 未登录词）**：

```text
词表里没有整个新词 → 映射成 <unk> → 原词信息丢失
```

| 粒度  | 优点                  | 代价              |
| --- | ------------------- | --------------- |
| 字符级 | 基础词表较小，片段可复用        | 序列较长；未覆盖字符仍可能未知 |
| 词级  | 常见词表示紧凑             | 词表大，新词容易 OOV    |
| 子词级 | 在覆盖能力、词表大小与序列长度之间折中 | token 不一定是完整词   |

示意：

```text
unhappiness
→ un + happi + ness
```

这只是可能的切法，必须由实际算法与词表决定。token 也可以是汉字、标点、空格片段或字节相关片段。

子词方法通常缓解 OOV，但不能一概保证完全没有未知符号；具有完整字节覆盖或相应回退机制的方案，才有更强的覆盖保证。

### 7.2 BPE：不断合并高频相邻片段

BPE：Byte Pair Encoding。用于子词学习时，可以从字符或字节等基础符号出发。

```text
训练：
1. 将语料按基础符号拆开
2. 统计相邻符号对的出现次数，计入词频
3. 选择频率最高的一对进行合并
4. 更新语料表示与词表，记录合并规则
5. 重复，直到达到目标词表大小或停止条件
```

$$
(a^*,b^*)=\arg\max_{(a,b)}C(a,b)
$$

示意合并链：

```text
l o w → lo w → low
```

编码新文本时，按照训练得到的合并规则优先级进行合并，**不是在新句子上重新训练、重新统计最高频词对**。

记忆：**BPE 从小片段出发，反复合并高频相邻组合。**

### 7.3 WordPiece：考虑组合的相对关联

课堂常用的训练直觉评分：

$$
\operatorname{score}(a,b)=\frac{C(a,b)}{C(a)C(b)}
$$

它体现的是：组合的共同出现，是否相对于两部分各自的频率足够突出。因此它与 BPE 单看相邻对次数不同。

这个评分是常见教学近似，不宜当作所有 WordPiece 实现的统一规格；Hugging Face 教程也明确说明，其训练算法讲解是依据公开文献作出的重建。[WordPiece 教程](https://huggingface.co/docs/course/chapter6/6)

训练后，经典 WordPiece 编码通常在最终词表中进行从左到右的最长匹配：

```text
playing → play + ##ing
```

`##` 是常见的词内部续接标记，不是原文真的含有两个井号。它主要依赖最终词表做切分，而不是重放 BPE 的合并序列。匹配失败如何返回未知词，需要看实现。[WordPiece 编码说明](https://huggingface.co/docs/course/chapter6/6)

### 7.4 Unigram：大候选词表中逐步删减

```text
1. 准备较大的候选子词集合
2. 为候选 token 估计概率
3. 评估删除某个 token 对语料似然的影响
4. 删掉影响较小的候选，并重新估计
5. 反复缩减至目标规模，保留必要的基础符号
```

对文本 $x$ 的一种切分 $z=(t_1,\ldots,t_m)$：

$$
P(z)=\prod_{i=1}^{m}p(t_i),\qquad
\log P(z)=\sum_{i=1}^{m}\log p(t_i)
$$

**一条切分路径的概率，不等于对所有切分求和后的文本概率。** 若 $\mathcal Z(x)$ 表示所有合法切分：

$$
P(x)=\sum_{z\in\mathcal Z(x)}P(z)
$$

确定性编码可寻找概率最大的切分；训练时也可利用多种候选切分。这里再次出现动态规划，是因为存在“多条候选切分路径”的问题。[Unigram 教程](https://huggingface.co/learn/llm-course/chapter6/7)

### 7.5 三者对照

| 方法        | 学习词表的直觉        | 编码直觉           | 记忆词   |
| --------- | -------------- | -------------- | ----- |
| BPE       | 按相邻对频率逐步合并     | 按已学习的合并优先级处理   | 高频合并  |
| WordPiece | 常用教学模型考虑组合关联收益 | 在最终词表中最长匹配     | 关联、匹配 |
| Unigram   | 概率估计后逐步删减候选    | 比较切分概率，选取或采样路径 | 概率剪枝  |

**SentencePiece 是工具库，不是与 BPE、WordPiece、Unigram 并列的第四种算法。** 它支持 BPE、Unigram 等方案，并处理原始文本到子词的相关流程。[Hugging Face 分词算法概览](https://huggingface.co/docs/transformers/tokenizer_summary)

<a id="s8"></a>

## 8. 工程工具：jieba 与 Hugging Face tokenizers

### 8.1 jieba：词典与统计方法结合

课堂可以简记为“词典分词 + HMM 补充”，更具体的默认路线是：

```text
前缀词典扫描
→ 生成候选成词的 DAG（有向无环图）
→ 根据词频，用动态规划找切分路径
→ 对未登录词相关片段，使用 HMM + Viterbi 补充识别
```

因此 jieba 不能简单等同于正向最大匹配。它支持自定义词典，`jieba.lcut(text)` 返回列表，`jieba.cut(text)` 返回可迭代的生成器。[jieba 官方说明](https://github.com/fxsjy/jieba)

```python
import jieba

words = jieba.lcut("我来到成都大学，学习人工智能。")
# 如需特定领域词，可按任务维护词典：
# jieba.add_word("电子信息工程")
```

具体切分受词典和设置影响，复习材料中的示意切分不应当作固定运行输出。

### 8.2 Hugging Face 工具链

| 工具                                | 主要角色                                 |
| --------------------------------- | ------------------------------------ |
| `tokenizers`                      | 底层快速分词库，支持 BPE、WordPiece、Unigram 等模型 |
| `transformers` 中的 `AutoTokenizer` | 载入与预训练模型配套的 tokenizer 配置，提供模型输入接口    |

一个常见处理流水线是：

```text
原始文本
→ 规范化（Normalization）
→ 预切分（Pre-tokenization）
→ 分词模型：BPE / WordPiece / Unigram
→ 后处理：特殊 token 等
→ IDs、mask 等编码结果
```

具体环节随配置变化。[Tokenizers 流水线文档](https://www.huggingface.co/docs/tokenizers/python/latest/pipeline.html)

使用配套模型时，常见调用形式为：

```python
# 示意代码：model_path 是已经准备好的配套模型目录。
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer(texts, padding=True, truncation=True,
                   return_tensors="pt")
# 常见结果包含 input_ids、attention_mask；其他字段依模型而定。
```

### 8.3 必须记住的工程边界

**Tokenizer 的 ID 映射必须与模型的 Embedding 表一致。**

```text
token → ID → Embedding 的某一行
```

如果任意替换词表，原来的某个 ID 可能指向另一个 token，模型就读错了输入。单纯把词切得更像人类理解，并不意味着适合某个已有模型。

`attention_mask` 中的有效位置标记，也不等于生成模型的因果约束；PAD mask 和“不能看未来”的 causal mask 解决不同问题。

<a id="s9"></a>

## 9. 课堂代码：文本到 Token ID

### 9.1 总体流程与变量含义

```text
corpus：原始句子列表
→ clean_words：分词、按课堂规则过滤标点与空白
→ tokenized：二维 token 列表
→ all_words：展平，仅用于统计词频
→ freq：Counter 词频表
→ vocab_words：加入 <pad>、<unk> 的词表
→ word2idx / idx2word：双向映射
→ encode / decode：句子与 ID 序列转换
```

### 9.2 整理后的课堂代码

```python
import jieba
from collections import Counter

corpus = [
    "我在成都大学欢迎2026级新同学",
    "我来到成都大学，学习人工智能和计算机视觉。",
]

PUNCT = set("，。、！？；：…—·《》【】（）“”‘’'\n\t ")

def clean_words(text):
    return [
        w for w in jieba.lcut(text)
        if w.strip() and not all(ch in PUNCT for ch in w)
    ]

tokenized = [clean_words(s) for s in corpus]
all_words = [w for sent in tokenized for w in sent]
freq = Counter(all_words)

PAD, UNK = "<pad>", "<unk>"
vocab_words = [PAD, UNK] + [
    w for w, _ in freq.most_common() if w not in {PAD, UNK}
]
word2idx = {w: i for i, w in enumerate(vocab_words)}
idx2word = {i: w for w, i in word2idx.items()}

def encode(text):
    return [word2idx.get(w, word2idx[UNK]) for w in clean_words(text)]

def decode(ids):
    return [idx2word.get(i, UNK) for i in ids]
```

### 9.3 逐项理解

| 对象         | 用处         | 容易混淆的点              |
| ---------- | ---------- | ------------------- |
| `Counter`  | 统计每个词出现多少次 | 没有训练词向量             |
| `word2idx` | 词 → 整数 ID  | ID 大小没有语义意义         |
| `idx2word` | ID → 词     | 用于读取预测结果或查看 token   |
| `<unk>`    | 词表外词的占位符   | 多个未知词会合并到同一个符号      |
| `<pad>`    | 为不同长度补齐    | 是占位符，不应当作为真实上下文贡献信息 |

补充注意：

- 原来的课堂代码只是定义了 `<pad>`，并没有真正执行 padding；CBOW 数据构造时才会用到。
- `decode(encode(text))` 得到的是处理后的词序列，无法恢复已经删除的标点，也无法恢复被 `<unk>` 替代的词。
- `PUNCT` 是手工列出的有限集合，不是完整的标点识别器；删除标点也不等于筛出了所有“有意义的词”。
- 是否删除标点由任务决定。语言生成中，标点和空白可能很重要，不应机械照搬此清洗流程。
- **展平用于词频统计，生成窗口时仍要保留句子边界，避免把上一句尾和下一句首当作上下文。**

<a id="s10"></a>

## 10. 词向量、One-Hot、余弦相似度与分布假设

### 10.1 ID 只是地址

```text
猫 → 3
狗 → 4
汽车 → 5
```

不能用 `|3 - 4|` 与 `|3 - 5|` 判断语义相似度。给词表重新编号，不应改变词本身的意义。

### 10.2 One-Hot：用正交向量表示身份

词表大小为 $V$，ID 为 $i$ 的词表示为：

$$
\mathbf x_i\in\{0,1\}^{V},\qquad
(\mathbf x_i)_k=\begin{cases}1,&k=i\\0,&k\ne i\end{cases}
$$

示意：

```text
猫   → [1, 0, 0]
狗   → [0, 1, 0]
汽车 → [0, 0, 1]
```

它没有人为引入编号大小关系，但有两个主要缺点：

1. **高维稀疏**：词表有 $V$ 个词，每个向量就有 $V$ 维；显式存储和计算很浪费。
2. **缺少语义几何结构**：任意两个不同词的向量都正交，无法区分近义词与无关词。

### 10.3 余弦相似度

对两个非零向量：

$$
\cos(\mathbf u,\mathbf v)
=\frac{\mathbf u^\top\mathbf v}{\|\mathbf u\|_2\|\mathbf v\|_2}
$$

| 值     | 几何意义 |
| ----- | ---- |
| 接近 1  | 方向接近 |
| 接近 0  | 近乎正交 |
| 接近 -1 | 方向相反 |

不同 One-Hot 向量的内积为 0、模长为 1，因此余弦相似度全部为 0。它们的欧氏距离也都是 $\sqrt2$。

**余弦相似度不是概率。** 负相似度不等于“反义词概率高”，反义词也可能因上下文相似而靠近。零向量的余弦没有定义，实际计算常加很小的稳定项。

### 10.4 词向量：低维、稠密、可学习

$$
\mathbf e_i\in\mathbb R^d,\qquad d\ll V
$$

```text
词 / token
→ 查到一个 d 维向量
→ 用训练任务不断更新它
→ 形成对任务有用的表示
```

向量各维通常没有人工指定的“爱情维度”“动物维度”。语义由多个维度共同表达，这是分布式表示的含义。

### 10.5 课堂的“分布式假设”

更常见的术语是 **分布假设（Distributional Hypothesis）**：出现在相似上下文中的词，往往具有相近的用法或语义。

```text
我 养了 一只 猫
我 养了 一只 狗

这只 猫 很 可爱
这只 狗 很 可爱
```

由此可以设计一个任务：通过上下文预测词，或通过词预测上下文，让网络从大量共现模式中学习表示。

需要区分：

| 名称    | 在说什么             |
| ----- | ---------------- |
| 分布假设  | 为什么上下文能提供语义线索    |
| 分布式表示 | 怎样用多个连续维度共同编码一个词 |

这是一种有效的建模依据，不是“上下文相同的词必然同义”的定理，也不保证某两个词训练后一定达到指定的余弦数值。

<a id="s11"></a>

## 11. Word2Vec：CBOW 与 Skip-gram

### 11.1 Word2Vec 的核心目的

**利用上下文预测任务学习词的向量表示。预测任务提供训练信号，Embedding 表是我们重点关心的训练成果。**

它不需要人工给每个词写“语义标签”：上下文和目标词直接来自语料，所以通常归为自监督表示学习。具体优化时仍采用输入—目标的监督式训练形式。

### 11.2 两种结构

设句子 token 为：

```text
我 今天 学习 自然语言 处理
```

中心词是“学习”，窗口半径 $r=2$，上下文为 `[我, 今天, 自然语言, 处理]`。

| 对照      | CBOW                     | Skip-gram            |
| ------- | ------------------------ | -------------------- |
| 全称 / 名称 | Continuous Bag-of-Words  | Skip-gram            |
| 方向      | 周围猜中间                    | 中间猜周围                |
| 输入      | 多个上下文 token              | 一个中心 token           |
| 目标      | 中心 token                 | 周围 token             |
| 示例      | `[我, 今天, 自然语言, 处理] → 学习` | `学习 → 我`、`学习 → 今天` 等 |
| 组合方式    | 上下文向量求和 / 平均             | 中心词与各上下文词组成预测对       |
| 常见经验    | 聚合上下文，训练通常较快             | 产生较多训练对，常有利于稀有词表示    |

速度与质量受数据、窗口、采样和实现影响，上述经验不是固定胜负关系。

数学方向：

$$
\text{CBOW:}\quad P(w_t\mid C_t)
$$

$$
\text{Skip-gram:}\quad
\sum_{j\in\{-r,\ldots,-1,1,\ldots,r\}}\log P(w_{t+j}\mid w_t)
$$

第二式表示对有效上下文位置的训练对累加对数概率，不意味着一次性生成一个有顺序的完整上下文句子。两种结构的原始说明见 [Word2Vec 论文](https://arxiv.org/abs/1301.3781)。

### 11.3 CBOW 中 Bag-of-Words 的含义

求和与平均不区分上下文词的排列：

$$
\frac{\mathbf e_a+\mathbf e_b}{2}
=\frac{\mathbf e_b+\mathbf e_a}{2}
$$

因此经典 CBOW 用窗口确定“哪些词属于上下文”，但聚合后不保留这些词的内部顺序。

这与前面“语言顺序重要”不矛盾：CBOW 是为了高效学习静态词向量而做出的简化，并没有解决全部语言理解问题。

### 11.4 Word2Vec 的静态性

```text
我吃了一个苹果
苹果发布了新手机
```

如果“苹果”是同一个词表条目，Word2Vec 查到的都是同一个向量。它不能仅凭查表就为两句话生成不同词义的表示。

经典的向量类比现象：

$$
\mathbf v_{\mathrm{king}}-\mathbf v_{\mathrm{man}}+\mathbf v_{\mathrm{woman}}
\approx\mathbf v_{\mathrm{queen}}
$$

这是特定训练条件下观察到的语义规律示例，不是每次训练都成立的恒等式。

### 11.5 课堂大纲中的两条实现路线

词向量部分的截图还列出了 Gensim 路线：安装工具、获取与加载公开词向量，以及用自己的语料训练词向量。可以把实现方式分成两类：

| 路线             | 复习时关注什么                         |
| -------------- | ------------------------------- |
| 工具库路线，如 Gensim | 词表覆盖、语料组织、加载或训练得到的向量如何使用        |
| PyTorch 手写路线   | 样本构造、张量维度、损失、反向传播和 Embedding 参数 |

本次对话详细展开的是 PyTorch 的 CBOW 原理；下节沿这条路线复原训练流程。

<a id="s12"></a>

## 12. CBOW 的完整训练流程

### 12.1 先确定 token 粒度与窗口含义

课堂截图以“自然语言处理是人工智能皇冠上的明珠”演示，按单字构造窗口：

```text
窗口半径 r = 2：左右各取 2 个位置，排除中心位置

[PAD, PAD, 然, 语] → 自
[PAD, 自,   语, 言] → 然
[自,   然,   言, 处] → 语
```

这里虽然沿用“中心词”的模型名称，实际训练单位是字。若先用 jieba 分词，训练单位就变成分词后的 token。**输入是什么粒度，学到的就是相应粒度的向量。**

为避免截图中示意 ID 前后不一致，本文统一约定一份演示映射：

```text
PAD=0，UNK=1，自=2，然=3，语=4，言=5，处=6

[0, 0, 3, 4] → 2
[0, 2, 4, 5] → 3
[2, 3, 5, 6] → 4
```

具体数字可以改变，但整份数据必须使用同一张词表。

### 12.2 生成训练样本

下面接第 9 节的 `tokenized`、`word2idx`，构建词级 CBOW 样本：

```python
radius = 2
pad_id = word2idx[PAD]
unk_id = word2idx[UNK]
samples = []

for words in tokenized:
    ids = [word2idx.get(w, unk_id) for w in words]
    for center in range(len(ids)):
        context = []
        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue  # 排除当前中心位置
            pos = center + offset
            context.append(ids[pos] if 0 <= pos < len(ids) else pad_id)
        if any(token_id != pad_id for token_id in context):
            samples.append((context, ids[center]))
```

边界也可采用缩短窗口或只使用完整窗口等策略；本例沿用课堂的 PAD 方案，并排除没有真实上下文的样本。不能跨句构造窗口。

### 12.3 前向传播：先查表，再平均，再预测

设：

- $V$：词表大小。
- $d$：词向量维度。
- $C$：上下文位置数，本例 $C=2r=4$。
- $N$：batch 大小。
- $E\in\mathbb R^{V\times d}$：输入 Embedding 表。
- $U\in\mathbb R^{V\times d}$：输出层权重。

```text
context_ids          [N, C]       整数索引
    ↓ Embedding
context_embeddings   [N, C, d]    每个上下文词的向量
    ↓ 排除 PAD 后求平均
h                    [N, d]       上下文表示
    ↓ 输出线性层
logits               [N, V]       每个候选词的未归一化分数
    ↓ Softmax（解释概率时使用）
P(中心词 | 上下文)    [N, V]
```

有效位掩码：

$$
m_i=\begin{cases}1,&c_i\ne\mathrm{PAD}\\0,&c_i=\mathrm{PAD}\end{cases}
$$

对单个样本：

$$
\mathbf h=\frac{\sum_{i=1}^{C}m_iE[c_i,:]}{\sum_{i=1}^{C}m_i}
$$

$$
\mathbf z=U\mathbf h+\mathbf b,\qquad
P(y=k\mid C)=\frac{\exp(z_k)}{\sum_{j=0}^{V-1}\exp(z_j)}
$$

这里把单个向量在公式中视作列向量。批量 PyTorch 计算相应写作 `h @ U.T + b`。教学实现可以带输出偏置，经典简化结构也常省略它。

### 12.4 交叉熵与参数更新

真实中心词 ID 为 $y$：

$$
L=-\log P(y\mid C)
$$

训练过程：

```text
上下文 ID
→ 前向计算 logits
→ 与真实中心词 ID 计算交叉熵
→ 反向传播
→ 优化器更新 E、U 和可选偏置
→ 重复读取大量窗口样本
```

PyTorch 的 `CrossEntropyLoss` 接收未归一化的 logits；对普通类别索引标签，内部完成相应的 LogSoftmax 与负对数似然计算。**不要在送入该损失前再手动 Softmax。** 标签通常是 `[N]` 形状的整数 ID。[PyTorch 交叉熵文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html)

### 12.5 与课堂数据衔接的最小模型

下面代码接第 9 节的预处理和第 12.2 节的样本构造；用于解释训练机制。整理时已检查 Python 语法，未进行实际训练验证；运行需要安装 `jieba` 与 `torch`。

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_id
        )
        self.out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_ids):
        emb = self.embedding(context_ids)       # [N, C, d]
        mask = (context_ids != self.pad_id).unsqueeze(-1)
        mask = mask.to(emb.dtype)               # [N, C, 1]
        count = mask.sum(dim=1).clamp_min(1)    # [N, 1]
        h = (emb * mask).sum(dim=1) / count    # [N, d]
        return self.out(h)                     # [N, V]，无 ReLU

if not samples:
    raise ValueError("没有有效的上下文训练样本")

contexts = torch.tensor([x for x, y in samples], dtype=torch.long)
targets = torch.tensor([y for x, y in samples], dtype=torch.long)
loader = DataLoader(TensorDataset(contexts, targets),
                    batch_size=16, shuffle=True)

torch.manual_seed(42)
model = CBOW(len(word2idx), embedding_dim=64, pad_id=pad_id)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(20):
    for context_ids, target_ids in loader:
        optimizer.zero_grad()
        logits = model(context_ids)
        loss = criterion(logits, target_ids)
        loss.backward()
        optimizer.step()

vectors = model.embedding.weight.detach().clone()  # [V, 64]
```

两条细节：

1. `padding_idx` 使默认初始化的 PAD 行为零，并使查表操作不向该行累积梯度；**它不会自动替你把平均分母从 4 改为真实词数**，所以仍要 mask。[Embedding 文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Embedding.html)
2. 只有两句语料时，代码能演示学习过程，但不能据此期待稳定的词义相似度或类比能力。实际质量要依靠更充分的语料和评估。

### 12.6 为什么会学出语义

随机初始化时，查表得到的是随机向量。大量窗口样本给参数提供了反复出现的共现约束：

```text
相似上下文与目标反复出现
→ 模型必须学习能支持这些预测的表示
→ 梯度持续调整参数
→ 向量逐渐编码语料中的共现、用法和部分语义关系
```

精确到一次 CBOW 更新：输入表中参与上下文的行得到梯度；输出表按损失得到更新。**不能说“猫和狗同时作为目标，所以这一条样本直接拉近了输入表的猫和狗”。** 输入词向量的结构来自它们在大量不同窗口中作为上下文参与训练的共同结果。

经典 Word2Vec 还常用负采样、层次 Softmax 降低大词表输出端成本。本文用全词表 Softmax 说明基础训练链路；负采样的若干二分类分数不直接等于完整词表上的归一化概率。

<a id="s13"></a>

## 13. Embedding 矩阵：词向量到底存在哪里

### 13.1 老师用 Linear 讲原理

设 One-Hot 列向量 $\mathbf x_i\in\mathbb R^V$，线性层：

$$
\mathbf e_i=W_1\mathbf x_i,\qquad W_1\in\mathbb R^{d\times V}
$$

One-Hot 只有 ID 对应的位置为 1，所以：

$$
W_1\mathbf x_i=W_1[:,i]
$$

**乘 One-Hot 在数学效果上等于取出一列。** 若真的构造大 One-Hot 并调用普通稠密线性层，实现仍可能执行多余计算。

```python
self.embedding = nn.Linear(vocab_size, embedding_dim, bias=False)
```

这里 `weight.shape == [embedding_dim, vocab_size]`，每一列对应一个词向量。PyTorch 的 `Linear` 权重顺序为 `[out_features, in_features]`。[Linear 文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Linear.html)

如果打开偏置，线性层输出是该列再加上偏置，不再只是原样查出该列。因此演示纯查表等价关系时写 `bias=False`。

### 13.2 工程中直接用 Embedding 查行

```python
self.embedding = nn.Embedding(vocab_size, embedding_dim)
```

$$
E\in\mathbb R^{V\times d},\qquad \mathbf e_i=E[i,:]
$$

每一行对应一个 token 的向量；与上面的无偏置线性层采用相同参数时，有 $E=W_1^\top$。[Embedding 文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Embedding.html)

| 写法                                   | 权重形状     | 取某个词向量                |
| ------------------------------------ | -------- | --------------------- |
| `Linear(V, d, bias=False)` + One-Hot | `[d, V]` | `weight[:, token_id]` |
| `Embedding(V, d)` + ID               | `[V, d]` | `weight[token_id]`    |

示例 $V=10000,d=300$：词表共 10000 行，每行 300 个可训练数值，共 $10000\times300=3000000$ 个参数。

### 13.3 训练结束后拿什么

```python
token = "人工智能"
token_id = word2idx[token]  # 先确认 token 在词表中
token_vector = model.embedding.weight[token_id].detach().clone()
```

CBOW 同时学习输入表与输出层权重；本课重点提取输入 Embedding 表。保存时也要保存 `word2idx` 或等价词表，否则不知道每一行属于哪个词。

**Embedding 是可训练查表机制；Word2Vec 是训练这张表的一类目标和方法。两者不是同义词。**

<a id="s14"></a>

## 14. 为什么经典 CBOW 的隐藏层不加 ReLU

### 14.1 课堂结论

**按经典 CBOW 结构实现时，Embedding / 投影与上下文平均之后，直接连接输出层，中间不加 ReLU。**

```text
ID → Embedding → 平均 / 求和 → 输出线性层 → 概率目标
```

原始 CBOW 的设计去掉了早期前馈神经语言模型的非线性隐藏层，采用共享投影和上下文平均，以降低训练成本。课堂口中的“隐藏层”主要对应这个投影 / 聚合表示。[Word2Vec 原始论文，第 3.1 节](https://arxiv.org/pdf/1301.3781)

### 14.2 加 ReLU 会改变什么

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

```text
[-0.8, 0.3, -0.2, 1.1]
           ↓ ReLU
[ 0.0, 0.3,  0.0, 1.1]
```

若加在平均之后：

$$
\mathbf z=U\operatorname{ReLU}(\mathbf h)+\mathbf b
$$

输出层读到的是截断后的表示；负坐标在这次前向传播中被置零，对应负输入处的局部梯度也为零。模型的计算和优化性质都变了。

而且放在平均之前与之后也不同：

$$
\operatorname{ReLU}\left(\frac{\mathbf e_1+\mathbf e_2}{2}\right)
\ne
\frac{\operatorname{ReLU}(\mathbf e_1)+\operatorname{ReLU}(\mathbf e_2)}2
\quad\text{（一般情况下）}
$$

### 14.3 不要把“不加”记成数学禁令

以下说法都过于绝对：

- “只要加了 ReLU 就不能训练词向量。”
- “词向量必须含负数才有语义。”
- “不加 ReLU 就一定能保证向量加减的语义关系。”

更准确的理解是：**经典 CBOW 选择简单的线性投影与聚合来高效学习表示；加 ReLU 会变成另一种结构，需要另行判断效果。**

ReLU 约束的是“经过激活的表示”非负，不会自动把原始 Embedding 权重矩阵全部变成非负。如果最终提取的是原始权重，不能把它与激活后的表示混为一谈。

此外，经典 CBOW 仍有非线性的概率目标，例如 Softmax；“中间不加 ReLU”不等于“损失和整个学习过程都是线性的”。

### 14.4 两层线性层为什么仍有价值

对平均 One-Hot 向量 $\bar{\mathbf x}$：

$$
\mathbf z=W_2W_1\bar{\mathbf x},\qquad
W_1\in\mathbb R^{d\times V},\quad W_2\in\mathbb R^{V\times d}
$$

虽然代数上能合成一个矩阵，但中间维度 $d$ 限制了乘积的秩：

$$
\operatorname{rank}(W_2W_1)\le d
$$

它相当于用两个较小矩阵学习一个共享的低维表示，同时把这个中间表示保留下来供其他任务使用。我们关心的不只是最终预测，也关心因预测任务而学到的参数结构。

<a id="s15"></a>

## 15. 电子信息工程类比与后续课程衔接

### 15.1 今天的数学主线

你觉得这一阶段像“概率论与数理统计”，原因很直接：

| 数学知识    | 今天用在哪里                  |
| ------- | ----------------------- |
| 条件概率    | 状态转移、发射、上下文预测           |
| 频率与参数估计 | 由标注语料估计 HMM 的 $\pi,A,B$ |
| 贝叶斯公式   | 把状态后验最大化转成联合概率最大化       |
| 对数      | 把路径概率连乘变成分数相加           |
| 向量与矩阵   | One-Hot、Embedding、线性映射  |
| 内积与模长   | 余弦相似度                   |
| 链式法则、梯度 | 反向传播更新词向量               |
| 动态规划    | Viterbi 最优路径            |

NLP 并非只有概率论，而是把概率统计、线性代数和优化接在一起。

### 15.2 状态系统的共同语言

```text
确定性状态机：
Q(t+1) = F(Q(t), 输入)

HMM：
P(s(t+1) | s(t))，以及 P(观测 | 状态)

RNN：
h(t) = f(Wx·x(t) + Wh·h(t-1) + b)
```

三者都可以从“状态如何传递”理解，但不要把状态类型混为一谈：FSM 是离散逻辑状态，HMM 有离散隐藏随机状态，RNN 的状态是连续高维向量。

Embedding 也可类比查表存储器：token ID 是地址，向量是存储内容。区别在于表中数值由训练更新，ID 编号自身不承载语义大小。

### 15.3 原对话中的后续预告

以下作为课程衔接，不与今天详细展开的 HMM、Tokenizer、Word2Vec 混成同一层重点。

| 路线          | 解决问题的直觉             | 关键式 / 概念                   |
| ----------- | ------------------- | -------------------------- |
| RNN         | 当前输入与上一时刻状态共同决定新状态  | $h_t=f(x_t,h_{t-1})$       |
| LSTM / GRU  | 用门控制保留、遗忘与更新        | 缓解长距离信息与梯度传播困难             |
| Seq2Seq     | 把一个序列映射成另一个序列       | Encoder → Decoder          |
| Attention   | 生成当前位置时，从已有状态中动态取信息 | $c_t=\sum_i\alpha_{ti}h_i$ |
| Transformer | 以注意力和前馈模块进行上下文建模    | 注意力 + 位置信息 + 前馈变换等         |

LSTM 的状态更新示意：

$$
C_t=f_t\odot C_{t-1}+i_t\odot\widetilde C_t
$$

普通 RNN 反向传播涉及沿时间连乘雅可比矩阵，可能梯度消失或爆炸；不能只看单个权重矩阵的某个特征值就判断所有情况。

Attention 的预告公式：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

从信号处理角度，可以把它理解为“根据当前输入计算权重，再对信息加权求和”。这里的注意力权重随输入变化，区别于训练结束后固定参数的普通卷积核。

典型架构例子：BERT 为 Encoder，GPT 为 Decoder，T5 为 Encoder–Decoder。自回归生成使用因果约束，预测：

$$
P(x_t\mid x_1,\ldots,x_{t-1})
$$

预测分布后既可以采样，也可以按最大概率等策略解码，不是所有生成都必须随机采样。

最关键的表示变化是：

```text
Word2Vec：同一词表条目 → 固定向量

Transformer：
同一 token 的初始查表向量可以相同
→ 结合位置信息和允许访问的上下文
→ 得到不同的上下文化隐藏表示
```

这才把前面“苹果的意思依赖语境”接回来。Transformer 的输入查表本身并不会自动根据语境变化，变化发生在后续上下文计算中。

<a id="s16"></a>

## 16. 复习速查与自测

### 16.1 十六个关键结论

| 问题                             | 应能直接说出的答案                        |
| ------------------------------ | -------------------------------- |
| 时序数据的三大差异？                     | 元素相关、长度可变、顺序有意义                  |
| 语言额外增加什么问题？                    | 离散符号需要映射为可计算的数值表示                |
| NLP 是一种网络吗？                    | 是处理自然语言的任务、方法与技术体系               |
| NLP 四次演进？                      | 规则 → 统计机器学习 → 神经表示学习 → 预训练模型与大模型 |
| 本课分词三代？                        | 规则 / 词典 → 统计 → 子词分割              |
| HMM 的隐藏与观测是什么？                 | 隐藏是 BMES 标签，观测是字符                |
| HMM 五元组？                       | 状态集、观测集、初始分布、转移概率、发射概率           |
| 转移与发射有什么区别？                    | 标签到标签；给定标签时看到某个字                 |
| Viterbi 保留什么？                  | 每个时刻、每个状态的最优历史及前驱                |
| Token 一定是词吗？                   | 不一定，可以是字、子词、标点、字节片段等             |
| BPE / WordPiece / Unigram 怎么记？ | 高频合并 / 关联收益与最长匹配 / 概率删减          |
| ID 是词向量吗？                      | ID 是索引；Embedding 查表后才得到向量        |
| One-Hot 缺点？                    | 高维稀疏，不同词正交，缺乏语义相似结构              |
| CBOW 与 Skip-gram？              | 周围猜中间；中间猜周围                      |
| CBOW 训练后重点提取什么？                | 输入 Embedding 矩阵，并保留词表映射          |
| 为什么经典 CBOW 不加 ReLU？            | 采用无非线性隐藏变换的共享投影与聚合，高效学习词表示       |

### 16.2 最容易混淆的地方

```text
计数表 ≠ 概率表：需要归一化
P(字 | 状态) ≠ P(状态 | 字)：条件方向不同
Viterbi ≠ 逐位置独立选最大：要考虑路径并回溯
Token ≠ 自然语言的完整词
Token ID ≠ Embedding
Tokenizer 训练 ≠ 词向量训练
Embedding ≠ Word2Vec
输入 Embedding 表 ≠ 输出层权重表
不加 ReLU ≠ 整个模型没有非线性概率目标
静态查表向量 ≠ 上下文化隐藏状态
```

### 16.3 合上笔记，试着回答

1. 为什么词表 ID 即使按词频排序，也不能拿 ID 差值计算语义距离？
2. 为什么 BMES 的 B 不能直接接 S？一句话最后为什么不能停在 M？
3. 在三句课堂语料中，B→E 的概率为何是 $8/9$？E 行的分母为何是 7？
4. 为什么 $P(今\mid B)=1/3$，而样本中的 $P(B\mid 今)=1$？
5. 两条路径到达同一时刻同一状态时，为什么 Viterbi 能丢掉分数较低的一条？
6. BPE 训练与编码新词时，分别使用什么规则？
7. Unigram 的“一条切分路径概率”与“文本概率”有什么差别？
8. `Embedding(V,d)` 的哪个维度对应词表？`Linear(V,d)` 的哪个维度对应词表？
9. CBOW 样本 `[PAD, PAD, 然, 语] → 自` 应如何平均上下文向量？
10. 为什么 CBOW 不保留窗口内部顺序，却仍能用于学习词向量？
11. 一次 CBOW 更新中，输入表哪些行有梯度？目标词一定直接更新输入表对应行吗？
12. 如果加了 ReLU，变成非负的是激活后的表示，还是整个原始参数矩阵？

### 16.4 一分钟口述版

> 今天先从普通全连接网络处理语言的困难出发：语言是变长、有序、内部相关的离散序列。我们先用分词或子词 Tokenizer 确定处理单位，再建立 token 与 ID 的映射。传统中文分词可以靠词典，也可以用 HMM 把词边界改写成 BMES 标签，通过初始、转移和发射概率给路径评分，再用 Viterbi 找最优路径。现代子词方法用有限片段组合文本，在词表大小、序列长度和未知词之间折中。ID 只是地址，One-Hot 也没有语义相似结构，所以需要可学习的稠密向量。Word2Vec 利用上下文提供自监督信号：CBOW 用周围预测中间，Skip-gram 用中间预测周围。CBOW 查 Embedding 表、平均有效上下文、预测中心词，再通过损失更新参数；经典结构中间不加 ReLU。训练后重点保留词向量表，后续再由序列模型和 Transformer 构造上下文化表示。

### 17 CBOW训练脚本

```python

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 1. 构造简单语料
sentence = [
    "我",
    "爱",
    "自然",
    "语言",
    "处理"
]

# 词表
word2idx = {
    word: idx
    for idx, word in enumerate(sentence)
}

idx2word = {
    idx: word
    for word, idx in word2idx.items()
}

vocab_size = len(word2idx)
# embedding维度
embedding_dim = 3

# 2. 构造CBOW训练样本
samples = []

for i in range(1, len(sentence)-1):
    context = [
        sentence[i-1],
        sentence[i+1]
    ]
    target = sentence[i]
    samples.append(
        (context, target)
    )
print("训练样本:")
for s in samples:
    print(s)

# 3. one-hot编码
def one_hot(index, size):

    vec = torch.zeros(size)

    vec[index] = 1

    return vec
# 4. CBOW模型

class CBOW(nn.Module):
    def __init__(self,vocab_size,embedding_dim):
        super().__init__()
        # 第一层：
        # one-hot -> word vector
        self.embedding = nn.Linear(
            vocab_size,
            embedding_dim,
            bias=False
        )
        # 输出层：
        # embedding -> vocab概率
        self.output = nn.Linear(
            embedding_dim,
            vocab_size
        )
    def forward(self, context):
        # 每个词映射成embedding
        emb = self.embedding(context)
        # CBOW:
        # 多个上下文向量求平均
        hidden = emb.mean(dim=0)
        # 分类预测中心词
        logits = self.output(hidden)
        return logits

# 5. 创建模型

model = CBOW(vocab_size, embedding_dim)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.1
)
# 6. 开始训练

epochs = 500
for epoch in range(epochs):
    total_loss = 0
    for context_words, target_word in samples:
        # 上下文one-hot
        context_vectors = torch.stack(
            [
                one_hot(
                    word2idx[w],
                    vocab_size
                )
                for w in context_words
            ]
        )
        # 标签
        target = torch.tensor(
            word2idx[target_word],
            dtype=torch.long
        )
        # 清梯度
        optimizer.zero_grad()
        # forward
        logits = model(
            context_vectors
        )
        # loss
        loss = criterion(
            logits.unsqueeze(0),
            target.unsqueeze(0)
        )
        # backward
        loss.backward()
        # 更新参数
        optimizer.step()
        total_loss += loss.item()

    if epoch % 50 == 0:
        print(
            f"epoch={epoch}, loss={total_loss:.4f}"
        )

# 7. 查看训练后的词向量

print("\n训练后的词向量:")

weight = model.embedding.weight.detach()
for word in sentence:
    idx = word2idx[word]
    # Linear权重:
    # shape=[embedding_dim,vocab_size]
    vector = weight[:, idx]
    print(
        word,
        vector.numpy()
    )

# 8. 计算词向量余弦相似度

def get_vector(word):
    idx = word2idx[word]
    return weight[:, idx]

v1 = get_vector("自然")
v2 = get_vector("语言")

similarity = F.cosine_similarity(
    v1.unsqueeze(0),
    v2.unsqueeze(0)
)

print(
    "\n自然 与 语言 相似度:",
    similarity.item()
)

```
