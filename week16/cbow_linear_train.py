
"""
CBOW (Continuous Bag of Words) 手搓训练示例
---------------------------------------
目标：
使用上下文词预测中心词，并训练第一层 Linear 权重作为词向量。

网络结构：
one-hot
  ↓
Linear(vocab_size -> embedding_dim)
  ↓
mean(context embeddings)
  ↓
Linear(embedding_dim -> vocab_size)
  ↓
CrossEntropyLoss

注意：
这里故意不用 nn.Embedding，而用 Linear 模拟 Embedding，
方便理解 Word2Vec 的本质。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# =========================
# 1. 构造简单语料
# =========================

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


# =========================
# 2. 构造CBOW训练样本
# =========================
# window=1:
# [左词,右词] -> 中心词

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


# =========================
# 3. one-hot编码
# =========================

def one_hot(index, size):

    vec = torch.zeros(size)

    vec[index] = 1

    return vec


# =========================
# 4. CBOW模型
# =========================

class CBOW(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim
    ):
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

        """
        context:
        [context_size, vocab_size]
        """

        # 每个词映射成embedding
        emb = self.embedding(context)

        # CBOW:
        # 多个上下文向量求平均
        hidden = emb.mean(dim=0)


        # 分类预测中心词
        logits = self.output(hidden)


        return logits



# =========================
# 5. 创建模型
# =========================

model = CBOW(
    vocab_size,
    embedding_dim
)


criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.1
)



# =========================
# 6. 开始训练
# =========================

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



# =========================
# 7. 查看训练后的词向量
# =========================

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



# =========================
# 8. 计算词向量余弦相似度
# =========================

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
