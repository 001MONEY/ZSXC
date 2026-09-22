"""
工具函数模块

包含：随机种子设置、模型权重初始化、贪心解码。
（BLEU 评估统一使用 evaluate.py 中基于 NLTK corpus_bleu 的实现）
"""

import random

import numpy as np
import torch

from data import Vocabulary


# ============================================================
#  随机种子
# ============================================================

def set_seed(seed: int) -> None:
    """
    统一设置所有随机种子，确保实验可复现。

    涵盖 Python random、NumPy、PyTorch CPU/GPU 的随机源。

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
#  权重初始化
# ============================================================

def init_weights(m: torch.nn.Module) -> None:
    """
    模型权重初始化策略。

    - RNN 的隐藏权重（weight_hh）：正交初始化，有助于缓解梯度消失/爆炸
    - RNN 的输入权重（weight_ih）：Xavier 均匀初始化
    - RNN 的偏置（bias_ih / bias_hh）：保持默认，不做额外初始化
    - 嵌入层 / 线性层的权重：Xavier 均匀初始化
    - 嵌入层 / 线性层的偏置：零初始化

    Args:
        m: PyTorch 模块
    """
    for name, param in m.named_parameters():
        if "weight_hh" in name:
            # RNN 隐藏权重：正交初始化
            torch.nn.init.orthogonal_(param)
        elif "weight_ih" in name:
            # RNN 输入权重：Xavier 均匀初始化
            torch.nn.init.xavier_uniform_(param)
        elif "bias" in name:
            # 区分 RNN bias 和普通层 bias：RNN bias 名称格式为 bias_ih_lN / bias_hh_lN
            if "_ih_" not in name and "_hh_" not in name:
                torch.nn.init.zeros_(param)
        elif param.dim() >= 2:
            # 权重矩阵（嵌入层、线性层）：Xavier 均匀初始化
            torch.nn.init.xavier_uniform_(param)


# ============================================================
#  贪心解码
# ============================================================

def greedy_decode(
    model: torch.nn.Module,
    src_indices: list[int],
    tgt_vocab: Vocabulary,
    device: torch.device,
    max_len: int = 50,
    return_attention: bool = False,
) -> list[int] | tuple[list[int], torch.Tensor]:
    """
    贪心解码（Greedy Decoding）：逐步选择概率最大的词作为下一步输入。

    这是 Seq2Seq 最简单的推理方式，每步取 argmax，不涉及束搜索。
    训练时的验证翻译和推理阶段的交互翻译均使用此函数。

    Args:
        model:       Seq2Seq 模型
        src_indices: 源序列的索引列表（已编码，含 EOS）
        tgt_vocab:   目标语言词表
        device:      计算设备
        max_len:     最大解码步数，防止无限循环
        return_attention: True 时额外返回注意力矩阵
    Returns:
        默认返回预测的 token 索引列表（不含 SOS/EOS/PAD）。
        return_attention=True 时额外返回 [目标词数, 源句长度] 的注意力矩阵；
        每一行与一个实际输出词对应，不包含 EOS/PAD/SOS 的注意力行。
    """
    # 记录原模式，函数返回前恢复，避免把 eval 状态泄漏给调用方（如训练循环）
    was_training = model.training
    model.eval()
    src_tensor = torch.tensor([src_indices], dtype=torch.long).to(device)

    # Decoder 首步输入：<SOS>
    input_token = torch.tensor([tgt_vocab.SOS_INDEX], dtype=torch.long).to(device)
    output_tokens = []
    attention_steps = []

    with torch.no_grad():
        # Encoder 编码源序列
        output, hidden = model.encoder(src_tensor, [len(src_indices)])
        source_mask = src_tensor.ne(Vocabulary.PAD_INDEX)

        for _ in range(max_len):
            prediction, hidden, attn_weights = model.decoder(input_token, hidden, output, source_mask)
            # 索引保持在 device 上，避免每步 GPU→CPU→GPU 往返；仅在判断和记录时取标量
            top1 = prediction.argmax(1)
            top1_item = top1.item()
            # 遇到 <EOS> 停止解码
            if top1_item == tgt_vocab.EOS_INDEX:
                break
            # 跳过特殊标记，只记录实际内容词
            if top1_item not in (tgt_vocab.PAD_INDEX, tgt_vocab.SOS_INDEX):
                output_tokens.append(top1_item)
                # 只保留与输出词一一对应的注意力行（EOS/PAD/SOS 的行不记录）
                attention_steps.append(attn_weights[0].detach().cpu())
            input_token = top1

    # 恢复模型原有的训练/评估模式
    if was_training:
        model.train()
    if return_attention:
        if attention_steps:
            attention_matrix = torch.stack(attention_steps, dim=0)
        else:
            attention_matrix = torch.empty((0, src_tensor.size(1)))
        return output_tokens, attention_matrix
    return output_tokens


def translate_sentence(
    model: torch.nn.Module,
    zh_text: str,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    src_tokenize,
    device: torch.device,
    max_len: int = 50,
) -> str:
    """
    完整的中→英翻译流程：中文文本 → 英文文本。

    将分词、编码、贪心解码、还原四个步骤串起来。

    Args:
        model:        Seq2Seq 模型
        zh_text:      输入的中文字符串
        src_vocab:    源语言词表
        tgt_vocab:    目标语言词表
        src_tokenize: 源语言分词函数（jieba_tokenize）
        device:       计算设备
        max_len:      最大解码长度
    Returns:
        英文翻译结果字符串
    """
    tokens = src_tokenize(zh_text)
    src_indices = src_vocab.tokens2index(tokens, add_eos=True)
    pred_indices = greedy_decode(model, src_indices, tgt_vocab, device, max_len)
    return " ".join(tgt_vocab.index2tokens(pred_indices))


def translate_sentence_with_attention(
    model: torch.nn.Module,
    zh_text: str,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    src_tokenize,
    device: torch.device,
    max_len: int = 50,
) -> tuple[str, list[str], list[str], torch.Tensor]:
    """翻译一句话，并返回与输出词逐行对齐的注意力矩阵。

    返回值：(翻译文本, 源句token标签（末尾补<EOS>，与注意力列对齐）, 目标token标签, 注意力矩阵)
    """
    source_tokens = src_tokenize(zh_text)
    src_indices = src_vocab.tokens2index(source_tokens, add_eos=True)
    pred_indices, attention_matrix = greedy_decode(
        model,
        src_indices,
        tgt_vocab,
        device,
        max_len,
        return_attention=True,
    )
    target_tokens = tgt_vocab.index2tokens(pred_indices)
    # 源句标签要和 src_indices 对齐：tokens2index(add_eos=True) 会在末尾补 <EOS>
    source_labels = source_tokens + [src_vocab.EOS_TOKEN]
    return " ".join(target_tokens), source_labels, target_tokens, attention_matrix
