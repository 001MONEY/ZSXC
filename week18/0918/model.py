"""
seq2seq 实现机器翻译：
   Encoder: 理解    双层双向的 GRU，输出各位置的编码和最终 hidden
   Decoder: 生成    双层单向的 GRU，每个时间步解码一个词
"""
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import Config
from utils import init_weights


class Encoder(nn.Module):
    """双向 GRU 编码器：把整句中文编码成各位置输出和最终 hidden。"""

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, pad_index=0, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size

        # 词嵌入：PAD 对应的向量恒为 0，且不参与梯度更新
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        # 双层双向 GRU：dropout 作用在层与层之间（num_layers > 1 时才生效）
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, source_tensors, src_lengths):
        """前向计算。

        source_tensors: [batch, seq_len]
        src_lengths:    [batch]，补 PAD 之前的真实长度（必须提供）
        output:         [batch, seq_len, hidden_size * 2]，PAD 位置为 0
        hidden:         [num_layers, batch, hidden_size * 2]
        """
        # embed: [batch, seq_len, embedding_dim]
        embed = self.embedding(source_tensors)

        # PAD 没有意义，打包后 GRU 会跳过 PAD，最终 hidden 对应真实句尾
        # enforce_sorted=False 允许一个 batch 内的句子长度乱序
        packed = pack_padded_sequence(
            embed,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, hidden = self.gru(packed)

        # hn 的排列是 [第0层正向, 第0层反向, 第1层正向, 第1层反向, ...]，
        # hn[0::2] 取出所有正向，hn[1::2] 取出所有反向，拼成 [num_layers, batch, 2 * hidden_size]
        hidden = torch.cat((hidden[0::2], hidden[1::2]), dim=2)

        # 把 PAD 填回输出，方便以后做 attention 时按位置取 output
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=source_tensors.size(1),
        )

        return output, hidden


class Decoder(nn.Module):
    """单向 GRU 解码器：每个时间步输入一个词，输出词表上的 logits。"""

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, pad_index=0, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        # 把隐状态映射成词表上每个词的分数
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, input_token, hidden):
        """单步解码。

        input_token: [batch]，当前时间步的词编号（第一步是 <SOS>）
        hidden:      [num_layers, batch, hidden_size]
        logits:      [batch, vocab_size]
        hidden:      [num_layers, batch, hidden_size]，传给下一步继续递推
        """
        # embed: [batch, 1, embedding_dim]
        embed = self.embedding(input_token).unsqueeze(1)

        # output: [batch, 1, hidden_size]
        output, hidden = self.gru(embed, hidden)

        # logits: [batch, vocab_size]
        logits = self.fc(output.squeeze(1))
        return logits, hidden



class Seq2Seq(nn.Module):
    """组合 Encoder 和 Decoder，训练阶段使用教师强制逐词解码。"""

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()

        # Decoder 的初始 hidden 直接来自 Encoder，双向拼接后维度翻倍
        if decoder.hidden_size != encoder.hidden_size * 2:
            raise ValueError(
                "Decoder 的 hidden_size 必须是 Encoder 的 2 倍"
                f"（应为 {encoder.hidden_size * 2}，当前是 {decoder.hidden_size}）"
            )

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source_tensors, target_tensor, src_lengths, teacher_forcing_ratio=0.5):
        """训练阶段前向：逐步解码，随机决定下一步输入真值还是模型预测。

        source_tensors:        [batch, src_len]
        target_tensor:         [batch, tgt_len]，第 0 位是 <SOS>
        src_lengths:           [batch]
        teacher_forcing_ratio: 下一步使用真值的概率
        logits:                [batch, tgt_len - 1, vocab_size]，与 target_tensor[:, 1:] 对齐
        """
        target_length = target_tensor.size(1)
        if target_length < 2:
            raise ValueError("target_tensor 至少需要 <SOS> 和一个目标词")

        # 编码整句中文，得到 Decoder 的初始 hidden
        _, hidden = self.encoder(source_tensors, src_lengths)

        # 第一步固定输入 <SOS>
        input_token = target_tensor[:, 0]
        step_logits = []

        # 循环解码：每次预测一个词，共 tgt_len - 1 步
        for step in range(target_length - 1):
            logits, hidden = self.decoder(input_token, hidden)
            step_logits.append(logits)

            # 教师强制：随机决定下一步喂真值，还是喂模型自己的预测
            if random.random() < teacher_forcing_ratio:
                input_token = target_tensor[:, step + 1]
            else:
                input_token = logits.argmax(dim=-1)

        return torch.stack(step_logits, dim=1)


def build_model(source_vocab_size, target_vocab_size, device):
    """按 Config 构建 Seq2Seq 模型（Encoder 双向 + Decoder 单向）并初始化权重。

    source_vocab_size: 中文词表大小
    target_vocab_size: 英文词表大小
    device:            训练设备（cuda / cpu）
    """
    encoder = Encoder(
        source_vocab_size, Config.EMBEDDING_DIM, Config.HIDDEN_SIZE,
        Config.ENCODER_LAYERS, Config.PADDING_IDX, Config.DROPOUT,
    )
    # 双向拼接后隐藏维度是 2 倍，所以 Decoder 的 hidden_size 也传 2 倍
    decoder = Decoder(
        target_vocab_size, Config.EMBEDDING_DIM, Config.HIDDEN_SIZE * 2,
        Config.DECODER_LAYERS, Config.PADDING_IDX, Config.DROPOUT,
    )
    model = Seq2Seq(encoder, decoder).to(device)

    # 注意顺序：init_weights 的 Xavier 会覆盖 Embedding 中 padding_idx 的零向量，
    # 所以初始化之后再把 PAD 位置的嵌入重新置零
    model.apply(init_weights)
    model.encoder.embedding.weight.data[Config.PADDING_IDX].zero_()
    model.decoder.embedding.weight.data[Config.PADDING_IDX].zero_()
    return model


if __name__ == '__main__':
    # 用假数据检查各层形状：中文词表 10，英文词表 12
    batch_size = 3
    zh_vocab_size, en_vocab_size = 10, 12
    embedding_dim, encoder_hidden_size, num_layers = 8, 16, 2

    encoder = Encoder(zh_vocab_size, embedding_dim, encoder_hidden_size, num_layers)
    # 双向拼接后是 2 倍，所以 Decoder 的 hidden_size 也要传 2 倍
    decoder = Decoder(en_vocab_size, embedding_dim, encoder_hidden_size * 2, num_layers)
    model = Seq2Seq(encoder, decoder)

    # 0 是 <PAD>，2 是 <EOS>，每行末尾的 2 表示句子真实结束位置
    source_tensors = torch.tensor([
        [5, 3, 7, 9, 2, 0, 0],
        [6, 8, 2, 0, 0, 0, 0],
        [4, 9, 5, 7, 3, 6, 2],
    ])
    src_lengths = torch.tensor([5, 3, 7])

    # 1 是 <SOS>，2 是 <EOS>
    target_tensor = torch.tensor([
        [1, 6, 8, 2, 0, 0],
        [1, 7, 2, 0, 0, 0],
        [1, 5, 9, 4, 2, 0],
    ])

    output, hidden = encoder(source_tensors, src_lengths)
    print("Encoder output:", tuple(output.shape))  # (3, 7, 32)
    print("Encoder hidden:", tuple(hidden.shape))  # (2, 3, 32)

    logits, _ = decoder(torch.full((batch_size,), 1), hidden)
    print("Decoder logits:", tuple(logits.shape))  # (3, 12)

    logits = model(source_tensors, target_tensor, src_lengths)
    print("Seq2Seq logits:", tuple(logits.shape))  # (3, 5, 12)

    # logits 与 target 错开一位对齐，<PAD> 不参与损失
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    loss = loss_function(
        logits.reshape(-1, en_vocab_size),
        target_tensor[:, 1:].reshape(-1),
    )
    print("loss:", loss.item())
