"""使用 RNN 实现基础 Seq2Seq 中英机器翻译模型。"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    """将中文编号序列编码成最终隐藏状态。"""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        num_layers=1,
        pad_index=0,
        dropout=0.2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_index,
        )
        # 当前只有一层 RNN，nn.RNN 自带的 dropout 不生效，因此显式作用于词向量。
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, src, src_lengths):
        # src: [batch_size, src_length]
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, src_length, embedding_dim]

        # 跳过每句话末尾补充的 PAD，使 hidden 对应真实句子末尾。
        packed = pack_padded_sequence(
            embedded,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.rnn(packed)
        # hidden: [num_layers, batch_size, hidden_size]
        return hidden


class Decoder(nn.Module):
    """根据 Encoder 隐藏状态生成英文词表上的预测。"""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        num_layers=1,
        pad_index=0,
        dropout=0.2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_index,
        )
        # 训练时随机屏蔽一部分词向量特征，验证和推理时会自动关闭。
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, decoder_input, hidden):
        # decoder_input: [batch_size, target_length]
        embedded = self.dropout(self.embedding(decoder_input))
        # embedded: [batch_size, target_length, embedding_dim]

        output, hidden = self.rnn(embedded, hidden)
        # output: [batch_size, target_length, hidden_size]

        logits = self.output_layer(output)
        # logits: [batch_size, target_length, target_vocab_size]
        return logits, hidden


class Seq2Seq(nn.Module):
    """组合 Encoder 和 Decoder。"""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lengths, decoder_input):
        """训练阶段：使用教师强制序列一次计算所有位置。"""
        encoder_hidden = self.encoder(src, src_lengths)
        logits, _ = self.decoder(decoder_input, encoder_hidden)
        return logits

    @torch.no_grad()
    def generate(
        self,
        src,
        src_lengths,
        sos_index,
        eos_index,
        max_length=30,
    ):
        """推理阶段：从 <SOS> 开始逐词生成，遇到 <EOS> 停止。"""
        if max_length < 1:
            raise ValueError("max_length 必须大于 0")

        hidden = self.encoder(src, src_lengths)
        batch_size = src.size(0)

        current_token = torch.full(
            (batch_size, 1),
            sos_index,
            dtype=torch.long,
            device=src.device,
        )
        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=src.device,
        )
        generated_tokens = []

        for _ in range(max_length):
            logits, hidden = self.decoder(current_token, hidden)
            next_token = logits[:, -1, :].argmax(dim=-1)
            generated_tokens.append(next_token)

            finished = finished | next_token.eq(eos_index)
            if finished.all():
                break

            # 已结束的句子继续送入 <EOS>，未结束的句子使用刚预测的词。
            next_token = torch.where(
                finished,
                torch.full_like(next_token, eos_index),
                next_token,
            )
            current_token = next_token.unsqueeze(1)

        return torch.stack(generated_tokens, dim=1)


if __name__ == "__main__":
    # 用假数据检查模型输入、输出形状。
    batch_size = 4
    src_length = 8
    target_length = 6
    zh_vocab_size = 100
    en_vocab_size = 120

    encoder = Encoder(
        vocab_size=zh_vocab_size,
        embedding_dim=64,
        hidden_size=128,
    )
    decoder = Decoder(
        vocab_size=en_vocab_size,
        embedding_dim=64,
        hidden_size=128,
    )
    model = Seq2Seq(encoder, decoder)

    src = torch.randint(4, zh_vocab_size, (batch_size, src_length))
    src_lengths = torch.tensor([8, 7, 6, 5])
    decoder_input = torch.randint(
        4,
        en_vocab_size,
        (batch_size, target_length),
    )

    logits = model(src, src_lengths, decoder_input)
    prediction = model.generate(
        src,
        src_lengths,
        sos_index=1,
        eos_index=2,
        max_length=10,
    )

    print("logits shape：", tuple(logits.shape))
    print("prediction shape：", tuple(prediction.shape))
