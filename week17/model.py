"""CBOW 模型：One-Hot -> 词向量 -> Masked Mean -> 中心词预测。"""

import torch
import torch.nn as nn


class CBOWModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()

        # One-Hot 乘权重矩阵等价于查找词向量。
        # bias=False 可保证全零 PAD 输入不会产生额外偏置向量。
        self.embedding = nn.Linear(
            in_features=vocab_size,
            out_features=embedding_dim,
            bias=False,
        )
        self.out = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size,
        )

    def forward(self, context_one_hot: torch.Tensor) -> torch.Tensor:
        """根据上下文预测中心词。

        context_one_hot: [batch_size, context_size, vocab_size]
        return: [batch_size, vocab_size]，为未经 Softmax 的 logits。
        """
        embedding = self.embedding(context_one_hot)

        # 真实词的 One-Hot 行之和为 1；被清零的 PAD 行之和为 0。
        valid_mask = (
            context_one_hot.sum(dim=-1, keepdim=True) > 0
        ).to(embedding.dtype)

        # PAD 既不进入分子，也不进入分母。
        embedding_sum = (embedding * valid_mask).sum(dim=1)
        no_pad_count = valid_mask.sum(dim=1).clamp(min=1)
        embedding_mean = embedding_sum / no_pad_count

        return self.out(embedding_mean)

    def get_token_embeddings(self) -> torch.Tensor:
        """返回形状为 [vocab_size, embedding_dim] 的词向量矩阵。"""
        # nn.Linear 的 weight 是 [embedding_dim, vocab_size]，所以需要转置。
        return self.embedding.weight.T
