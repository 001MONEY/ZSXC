"""
seq2seq 实现机器翻译：
   Encoder:  理解    双层双向的 GRU，输出各位置的编码和最终 hidden
   Attention:对齐    Bahdanau 加性注意力，解码每一步查看编码器全部位置
   Decoder:  生成    双层单向的 GRU，输入 [词向量; 上下文向量]，每个时间步解码一个词
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

class Attention(nn.Module):
    """Bahdanau 加性注意力：计算解码器当前状态对编码器各位置的权重，并加权求和。

    encoder_dim: 编码器输出维度（双向拼接后 = 2 * hidden_size）
    decoder_dim: 解码器隐状态维度（本项目里与 encoder_dim 相等）
    attn_dim:    注意力中间层维度（对齐空间）
    """

    def __init__(self, encoder_dim, decoder_dim, attn_dim):
        super().__init__()
        # score(h_s, s_{t-1}) = v^T tanh(W1 @ h_s + W2 @ s_{t-1})
        self.W1 = nn.Linear(encoder_dim, attn_dim, bias=False)   # 投影编码器各位置输出
        self.W2 = nn.Linear(decoder_dim, attn_dim, bias=False)   # 投影解码器上一步隐状态
        self.v = nn.Linear(attn_dim, 1, bias=False)              # 打分 → 每个位置一个标量

    def forward(self, decoder_hidden, encoder_outputs, mask):
        """计算一步注意力。

        decoder_hidden:  [batch, decoder_dim]，上一步 decoder 顶层隐状态 s_{t-1}
        encoder_outputs: [batch, src_len, encoder_dim]，编码器各位置输出 h_s
        mask:            [batch, src_len]，True 表示真实词、False/0 表示 PAD

        context:         [batch, encoder_dim]，编码器输出的加权和（上下文向量）
        attn_weights:    [batch, src_len]，每个源位置的注意力权重，可用于可视化对齐
        """
        # W1(h_s) 每步都一样，可提到解码循环外预计算；这里为直观起见直接算
        # [batch, src_len, attn_dim] + [batch, 1, attn_dim] → tanh → [batch, src_len, attn_dim]
        energy = torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden).unsqueeze(1))

        # [batch, src_len, 1] → [batch, src_len]
        energy = self.v(energy).squeeze(-1)

        # PAD 位置不能分到注意力：填一个极小值，softmax 后权重约等于 0
        energy = energy.masked_fill(mask == 0, -1e9)

        # [batch, src_len]，每个源位置一个权重，且权重之和为 1
        attn_weights = torch.softmax(energy, dim=-1)

        # [batch, 1, src_len] @ [batch, src_len, encoder_dim] → [batch, 1, encoder_dim] → [batch, encoder_dim]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attn_weights

class Decoder(nn.Module):
    """单向 GRU 解码器：每个时间步先对编码器全部位置做注意力，再解码一个词。"""

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, pad_index=0, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        # 注意力：把解码器上一步隐状态对齐到 Encoder 各位置，得到上下文向量
        # （Encoder 输出与 Decoder 隐状态的维度都是 hidden_size，对齐空间维度也取 hidden_size）
        self.attention = Attention(hidden_size, hidden_size, hidden_size)
        # GRU 输入不再是单纯词向量，而是 [词向量; 上下文向量] 拼接，所以输入维度要加上 hidden_size
        self.gru = nn.GRU(
            input_size=embedding_dim + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        # 把隐状态映射成词表上每个词的分数
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, input_token, hidden, encoder_outputs, mask):
        """单步解码（带注意力）。

        input_token:     [batch]，当前时间步的词编号（第一步是 <SOS>）
        hidden:          [num_layers, batch, hidden_size]
        encoder_outputs: [batch, src_len, hidden_size]，Encoder 各位置输出
        mask:            [batch, src_len]，True=真实词、False=PAD

        logits:          [batch, vocab_size]
        hidden:          [num_layers, batch, hidden_size]，传给下一步继续递推
        attn_weights:    [batch, src_len]，本步对源句各位置的注意力权重（可可视化对齐）
        """
        # 注意力用解码器顶层隐状态 s_{t-1} 打分（hidden[-1]: [batch, hidden_size]）
        context, attn_weights = self.attention(hidden[-1], encoder_outputs, mask)

        # embed: [batch, 1, embedding_dim]
        embed = self.embedding(input_token).unsqueeze(1)

        # 词向量与上下文向量在最后一维拼接：[batch, 1, embedding_dim + hidden_size]
        gru_input = torch.cat([embed, context.unsqueeze(1)], dim=-1)

        # output: [batch, 1, hidden_size]
        output, hidden = self.gru(gru_input, hidden)

        # logits: [batch, vocab_size]
        logits = self.fc(output.squeeze(1))
        return logits, hidden, attn_weights



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

        # 编码整句中文，得到 Decoder 的初始 hidden 和全部位置的编码（注意力每步都要查询）
        encoder_outputs, hidden = self.encoder(source_tensors, src_lengths)

        # 注意力掩码：True 表示真实词；PAD 位置不允许分到注意力
        mask = source_tensors.ne(Config.PADDING_IDX)

        # 第一步固定输入 <SOS>
        input_token = target_tensor[:, 0]
        step_logits = []

        # 循环解码：每次预测一个词，共 tgt_len - 1 步
        for step in range(target_length - 1):
            logits, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, mask)
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

    encoder_outputs, hidden = encoder(source_tensors, src_lengths)
    print("Encoder output:", tuple(encoder_outputs.shape))  # (3, 7, 32)
    print("Encoder hidden:", tuple(hidden.shape))  # (2, 3, 32)

    # 0 是 <PAD>：第 0 行有 2 个 PAD、第 1 行有 4 个 PAD
    mask = source_tensors.ne(0)
    logits, _, attn = decoder(torch.full((batch_size,), 1), hidden, encoder_outputs, mask)
    print("Decoder logits:", tuple(logits.shape))  # (3, 12)
    print("Attention weights:", tuple(attn.shape))  # (3, 7)
    print("Attention 行和:", [round(x, 6) for x in attn.sum(-1).tolist()])
    print("PAD 位置权重（应全为 0）:", attn[:, 5:7].tolist())

    logits = model(source_tensors, target_tensor, src_lengths)
    print("Seq2Seq logits:", tuple(logits.shape))  # (3, 5, 12)

    # logits 与 target 错开一位对齐，<PAD> 不参与损失
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    loss = loss_function(
        logits.reshape(-1, en_vocab_size),
        target_tensor[:, 1:].reshape(-1),
    )
    print("loss:", loss.item())
