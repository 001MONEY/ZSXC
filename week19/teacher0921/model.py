"""
seq2seq实现机器翻译：
   Encoder: 理解    双层双向的循环神经网络
   Decoder: 生成    双层单向的循环神经网络
"""
import torch.nn as nn
import random
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import Config
from utils import init_weights


class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, padding_idx=0, dropout=0.5):
        super().__init__()
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        # 双层双向的循环神经网络
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

        # 这个线性层的作用是： 将hidden_size * 2 维度降维成 hidden_size
        self.fc = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)

    def forward(self, source_tensors, src_lengths):
        # source_tensors的shape: (batch, seq_len)  ==> (3, 5)
        # embed的shape: (batch, seq_len, embedding_dim)  ==> (3, 5, 10)
        # 交给词嵌入层
        embed = self.embedding(source_tensors)
        # 交给GRU训练的时候需要将PAD去除, 因为PAD没有意义
        # pack_padded_sequence: 压缩已经填充PAD的序列， 也就是压缩掉PAD
        # pad_packed_sequence: 填充被压缩了PAD的序列， 也就是填充PAD
        # lengths: 表示没有填充PAD时的真实长度
        if src_lengths is not None:
            pack_seq = pack_padded_sequence(embed, src_lengths, batch_first=True, enforce_sorted=False)
            # output的shape: (batch, seq_len, hidden_size)
            # hn的shape: (num_layers * 2, batch, hidden_size)
            # gru是双层双向的网络， 将第一层的正向和反向拼接； 第二层的正向和反向拼接
            # 可以直接取第二层的正向和反向拼接， 但是由于我们的Decoder也是2层， 所以Encoder获取第二层拼接后还需要做形状变换成2层
            output, hn = self.gru(pack_seq)
            # 将PAD填充回去
            output, _ = pad_packed_sequence(
                output,
                batch_first=True,
                total_length=source_tensors.size(1),
            )
        else:
            output, hn = self.gru(embed)

        # 每一层分别拼接双向 hidden：[2L, B, H] -> [L, B, 2H] -> [L, B, H]
        hn = torch.cat((hn[0::2], hn[1::2]), dim=2)
        hn = self.fc(hn)

        return output, hn

class Attention(nn.Module):
    """乘性注意力：e_tj = s_t^T W_a h_j。"""

    def forward(self, decoder_output, encoder_output, source_mask):
        # 使用解码器的隐状态和编码器的所有时间步的隐状态计算相关性
        # decoder_output的shape: (batch, 1, hidden_size)
        # encoder_output的shape: (batch, seq_len, hidden_size)

        # 将encoder_output转置成：(batch, hidden_size, seq_len)
        encoder_output_T = encoder_output.transpose(1, 2)
        # 和decoder_output计算相关性评分
        # scores的shape: (batch, 1, seq_len)
        scores = decoder_output.bmm(encoder_output_T)
        # PAD 位置不参与注意力归一化
        scores = scores.masked_fill(source_mask.unsqueeze(1) == 0, -1e9)
        # 计算权重
        atten_w = torch.softmax(scores, -1)
        # 跟 encoder_output 加权求和 - 直接使用矩阵乘法
        context = torch.matmul(atten_w, encoder_output)
        # 返回上下文向量和注意力权重
        # atten_w的shape: (batch, 1, seq_len) → squeeze成 (batch, seq_len)，方便按 [batch, src_len] 使用/可视化
        return context, atten_w.squeeze(1)

class Decoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, padding_idx=0, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 将循环神经网络的隐状态结果映射成词表的概率
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

        # W_a：把双向 Encoder 输出投影到 Decoder 隐状态维度，用于乘性打分
        self.encoder_project = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        # 上下文和当前状态融合后，使用独立参数投影，避免与 W_a 意外共享参数
        self.combine_project = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)

        # 初始化Attention对象
        self.attention = Attention()

    def forward(self, input_token, hidden, encoder_output, source_mask):
        # 单步解码
        # Decoder解码器需要接收： 当前时间步的输入（input_token）和Encoder输出的结果（hidden）
        # input_token的shape: (batch, )
        # embed的shape: (batch, embedding_dim) --> (batch, 1, embedding_dim)
        # hidden的shape: (num_layers, batch, hidden_size)
        embed = self.embedding(input_token).unsqueeze(1)
        # output的shape: (batch, 1, hidden_size)
        # hn的shape: (num_layers, batch, hidden_size)
        output, hn = self.gru(embed, hidden)

        # 使用注意力机制得到上下文
        # output的shape: (batch, 1, hidden_size)
        # encoder_output的shape: (batch, seq_len, 2 * hidden_size)
        # encoder_project_output的shape: (batch, seq_len, hidden_size)

        # context的shape: (batch, 1, hidden_size)
        # attn_weights的shape: (batch, seq_len)
        encoder_project_output = self.encoder_project(encoder_output)
        context, attn_weights = self.attention(output, encoder_project_output, source_mask)
        # 将上下文和当前隐状态信息进行cat融合
        # hn: (num_layers, batch, hideen_size)
        # hn[-1]: (batch, hideen_size)
        # hn[-1].unsqueeze(1): (batch, 1, hideen_size)
        # combine的shape: (batch, 1, hideen_size * 2)
        combine = torch.cat((context, hn[-1].unsqueeze(1)), dim=2) #
        # 线性层
        out = self.fc(self.combine_project(combine)).squeeze(1)
        # 额外返回注意力权重，供生成对齐热力图/分析使用
        return out, hn, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source_tensors, target_tensor, src_lengths, teacher_forcing_ratio=0.5):
        # 编码器
        # source_tensors的shape: (batch, src_len), 包含了 EOS, PAD
        # target_tensor的shape: (batch, tgt_len), 包含了 SOS，EOS, PAD
        # src_lengths： 真实的中文长度， 也就是没有PAD
        output, hn = self.encoder(source_tensors, src_lengths)
        source_mask = source_tensors.ne(Config.PADDING_IDX)
        batch_size = target_tensor.size(0)
        tgt_len = target_tensor.size(1)
        # 获取英文词表大小
        tgt_vocab_size = self.decoder.fc.out_features

        # outputs用来保存每个时间步预测结果的, 用来计算损失的
        # outputs保持和target_tensor同设备运行
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(target_tensor.device)
        # input_token第一次输入的是SOS
        input_token = target_tensor[:, 0]
        # 循环解码
        for t in range(1, tgt_len):
            # logits的shape: (batch, vocab_size)
            # 第三个返回值是注意力权重，训练损失只关心 logits，用 _ 丢弃
            logits, hn, _ = self.decoder(input_token, hn, output, source_mask)
            # logits得到的是当前时间步模型输出的分数，但是我们计算损失是要所有时间步的分数结果和真实值算损失
            outputs[:, t] = logits
            # logits最大的值对应的索引就是预测的token的id
            predict = logits.argmax(-1)

            # 教师强制
            is_teacher_force = random.random() < teacher_forcing_ratio
            input_token = target_tensor[:, t] if is_teacher_force else predict

        return outputs


def build_model(source_vocab_size, target_vocab_size, device):
    """
    构建Seq2Seq模型
    :param source_vocab_size: 中文词表
    :param target_vocab_size: 英文词表
    :return:
    """
    # 创建Encoder对象
    encoder = Encoder(source_vocab_size, Config.EMBEDDING_DIM, Config.HIDDEN_SIZE,
                      Config.ENCODER_LAYERS, Config.PADDING_IDX, Config.DROPOUT_RATE)

    # 创建Decoder对象
    decoder = Decoder(target_vocab_size, Config.EMBEDDING_DIM, Config.HIDDEN_SIZE,
                      Config.DECODER_LAYERS, Config.PADDING_IDX, Config.DROPOUT_RATE)
    # 创建Seq2Seq对象
    model = Seq2Seq(encoder, decoder).to(device)
    model.apply(init_weights)
    # init_weights 的 Xavier 初始化会覆盖 Embedding 中 padding_idx 的零向量，
    # 这里重新置零，保证 PAD 位置的嵌入始终为零向量
    model.encoder.embedding.weight.data[Config.PADDING_IDX].zero_()
    model.decoder.embedding.weight.data[Config.PADDING_IDX].zero_()

    return model


if __name__ == '__main__':
    # decoder_output的shape: (batch, 1, hidden_size)
    # encoder_output的shape: (batch, seq_len, hidden_size)
    decoder_output = torch.rand((64, 1, 128))
    encoder_output = torch.rand((64, 23, 128))
    attention = Attention()
    source_mask = torch.ones((64, 23), dtype=torch.bool)
    context, attn_weights = attention(decoder_output, encoder_output, source_mask)
    print(context.shape)
    print(attn_weights.shape)
