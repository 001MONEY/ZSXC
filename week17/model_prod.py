"""工程版 CBOW：Embedding + Masked Mean + Negative Sampling。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOWNegativeSampling(nn.Module):
    """通过负采样训练词向量，避免对整个大词表计算 Softmax。"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        pad_index: int,
    ) -> None:
        super().__init__()
        self.pad_index = pad_index

        # 输入词向量用于编码上下文，输出词向量用于判断中心词。
        self.input_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,
        )
        self.output_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        bound = 0.5 / embedding_dim
        nn.init.uniform_(self.input_embeddings.weight, -bound, bound)
        nn.init.zeros_(self.output_embeddings.weight)
        with torch.no_grad():
            self.input_embeddings.weight[pad_index].zero_()

    def encode_context(self, context_ids: torch.Tensor) -> torch.Tensor:
        """对非 PAD 上下文做 Masked Mean。"""
        vectors = self.input_embeddings(context_ids)
        mask = (context_ids != self.pad_index).unsqueeze(-1)

        vector_sum = (vectors * mask).sum(dim=1)
        valid_count = mask.sum(dim=1).clamp(min=1)
        return vector_sum / valid_count

    def forward(
        self,
        context_ids: torch.Tensor,
        target_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> torch.Tensor:
        """返回一个 batch 的负采样损失。

        context_ids:  [batch_size, context_size]
        target_ids:   [batch_size]
        negative_ids: [batch_size, num_negative]
        """
        context_vectors = self.encode_context(context_ids)

        positive_vectors = self.output_embeddings(target_ids)
        positive_scores = (
            context_vectors * positive_vectors
        ).sum(dim=-1)

        negative_vectors = self.output_embeddings(negative_ids)
        negative_scores = (
            negative_vectors * context_vectors.unsqueeze(1)
        ).sum(dim=-1)

        positive_loss = -F.logsigmoid(positive_scores)
        negative_loss = -F.logsigmoid(-negative_scores).sum(dim=1)
        return (positive_loss + negative_loss).mean()

    def get_token_embeddings(self) -> torch.Tensor:
        """返回最终用于检索和相似度计算的输入词向量。"""
        return self.input_embeddings.weight
