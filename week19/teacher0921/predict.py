"""

推理预测

运行后进入交互模式：
    输入中文 → 输出英文翻译
    输入 q  → 退出
"""

import os
import torch
from config import Config, get_device
from data import Vocabulary, zh_tokenizer, generate_dataloader
from model import build_model
from utils import translate_sentence


def interactive_translate(model, src_vocab, tgt_vocab, device):
    """
    控制台交互式翻译循环。

    用户输入中文句子，模型实时输出英文翻译。
    输入 'q' 退出。

    Args:
        model:     Seq2Seq 模型
        src_vocab: 源语言词表
        tgt_vocab: 目标语言词表
        device:    计算设备
    """
    print("=" * 50)
    print("中英翻译交互式推理（输入中文，输出英文；输入 q 退出）")
    print("=" * 50)

    while True:
        try:
            text = input("\n中文: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if text.lower() == "q":
            print("退出")
            break
        if not text:
            continue

        result = translate_sentence(model, text, src_vocab, tgt_vocab, zh_tokenizer, device)
        print(f"English: {result}")


# ============================================================
#  PyCharm 右键运行入口
# ============================================================

if __name__ == "__main__":
    cfg = Config()
    device = get_device()

    # 加载词表（使用缓存，确保与训练时一致）
    print("加载数据和词表...")
    _, _, _, _, src_vocab, tgt_vocab = generate_dataloader()
    print(f"词表: 中文 {len(src_vocab)} | 英文 {len(tgt_vocab)}")

    # 构建模型并加载权重
    print("加载模型...")
    model = build_model(len(src_vocab), len(tgt_vocab), device)
    ckpt_path = os.path.join(cfg.SAVE_DIR, cfg.CHECKPOINT_NAME)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print("模型加载完成")

    interactive_translate(model, src_vocab, tgt_vocab, device)
