"""
评估脚本


功能：
    1. 加载训练好的模型和词表
    2. 在验证集上计算 BLEU-4 分数
    3. 展示翻译示例
"""

import random

import torch

from config import Config, get_device
from data import zh_tokenizer, en_tokenizer, create_dataloaders
from model import build_model
from utils import set_seed, translate_sentence
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction



def evaluate_bleu(model, val_pairs, src_vocab, tgt_vocab, device, sample_size=None, seed=42):
    """
    在验证集上计算 BLEU-4 分数。

    Args:
        model:       Seq2Seq 模型
        val_pairs:   验证集句对列表
        src_vocab:   源语言词表
        tgt_vocab:   目标语言词表
        device:      计算设备
        sample_size: 采样评估的句对数量，None 表示全量评估
        seed:        随机种子

    Returns:
        BLEU-4 分数
    """
    random.seed(seed)
    if sample_size is None:
        sample = val_pairs
    else:
        sample = random.sample(val_pairs, min(sample_size, len(val_pairs)))

    preds, refs = [], []
    for zh, en in sample:
        pred_en = translate_sentence(model, zh, src_vocab, tgt_vocab, zh_tokenizer, device)
        preds.append(pred_en.split())
        refs.append(en_tokenizer(en))

    # method1 平滑：Tatoeba 短句语料上预测与参考若缺少 4-gram 重合，
    # 未平滑的 BLEU 会直接为 0 并打印警告
    bleu4 = corpus_bleu([[ref] for ref in refs], preds,
                        smoothing_function=SmoothingFunction().method1)
    return bleu4


def show_examples(model, val_pairs, src_vocab, tgt_vocab, device, num=8):
    """
    展示翻译示例，对比参考翻译与模型输出。

    Args:
        model:     Seq2Seq 模型
        val_pairs: 验证集句对列表
        src_vocab: 源语言词表
        tgt_vocab: 目标语言词表
        device:    计算设备
        num:       展示的句对数量
    """
    print("\n翻译示例：")
    for zh, en in val_pairs[:num]:
        pred = translate_sentence(model, zh, src_vocab, tgt_vocab, zh_tokenizer, device)
        print(f"  中文: {zh}")
        print(f"  参考: {en}")
        print(f"  翻译: {pred}")
        print()


# ============================================================
#  PyCharm 右键运行入口
# ============================================================

if __name__ == "__main__":
    cfg = Config()
    device = get_device()
    set_seed(cfg.SEED)

    # 获取验证集和词表（词表文件已存在会直接加载，确保与训练时一致）
    print("加载数据和词表...")
    _, _, _, val_pairs, src_vocab, tgt_vocab = create_dataloaders()
    print(f"验证集: {len(val_pairs)} 条")
    print(f"词表: 中文 {len(src_vocab)} | 英文 {len(tgt_vocab)}")

    # 加载模型（用返回的词表大小重新构建）
    print("加载模型...")
    model = build_model(len(src_vocab), len(tgt_vocab), device)
    ckpt_path = cfg.CHECKPOINT_PATH
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print("模型加载完成")

    # 计算 BLEU
    bleu4 = evaluate_bleu(model, val_pairs, src_vocab, tgt_vocab, device)
    print(f"\n验证集 BLEU-4: {bleu4:.4f}（百分制 {bleu4 * 100:.2f}）")
    print("（该分数来自 Attention 模型的无 Teacher Forcing 贪心解码）")

    # 展示翻译示例
    show_examples(model, val_pairs, src_vocab, tgt_vocab, device)
