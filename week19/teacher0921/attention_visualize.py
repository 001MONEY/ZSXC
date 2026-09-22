"""生成中英翻译的 Attention 对齐热力图（乘性注意力版）。"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from config import Config, get_device
from data import Vocabulary, zh_tokenizer
from model import build_model
from utils import translate_sentence_with_attention


def plot_attention_heatmap(source_tokens, target_tokens, attention, output_path):
    """把 [目标词数, 源词数] 的注意力矩阵保存为热力图。"""
    if attention.numel() == 0 or not target_tokens:
        raise ValueError("模型没有生成可绘制的内容词，无法创建 Attention 热力图")
    if attention.shape != (len(target_tokens), len(source_tokens)):
        raise ValueError(
            "Attention 形状与词标签不匹配："
            f"attention={tuple(attention.shape)}, "
            f"target={len(target_tokens)}, source={len(source_tokens)}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 中文字体：按可用性依次回退，避免热力图标签出现方块
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    width = max(7.0, 0.75 * len(source_tokens) + 2.5)
    height = max(4.5, 0.55 * len(target_tokens) + 2.0)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(attention.numpy(), aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(source_tokens)), labels=source_tokens, rotation=35, ha="right")
    ax.set_yticks(range(len(target_tokens)), labels=target_tokens)
    ax.set_xlabel("源句（中文）")
    ax.set_ylabel("模型输出（英文）")
    ax.set_title("Seq2Seq 乘性 Attention 对齐")

    # 词数不多时把数值直接标在格子里（颜色深的地方用白字）
    if len(source_tokens) * len(target_tokens) <= 120:
        threshold = float(attention.max()) * 0.55
        for row in range(attention.size(0)):
            for col in range(attention.size(1)):
                value = float(attention[row, col])
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value >= threshold else "black",
                    fontsize=8,
                )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Attention 权重")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="生成 Attention 对齐热力图")
    parser.add_argument("--sentence", default="我爱你。", help="待翻译的中文句子")
    parser.add_argument(
        "--output",
        type=Path,
        default=Config.ATTENTION_DIR / "attention_heatmap.png",
        help="热力图保存位置",
    )
    args = parser.parse_args()

    device = get_device()
    # 词表文件为 raw/zh.json、raw/en.json（Vocabulary.load 内部会拼 Config.DATA_PATH）
    src_vocab = Vocabulary.load("zh.json")
    tgt_vocab = Vocabulary.load("en.json")
    model = build_model(len(src_vocab), len(tgt_vocab), device)
    ckpt_path = os.path.join(Config.SAVE_DIR, Config.CHECKPOINT_NAME)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    translation, source_tokens, target_tokens, attention = translate_sentence_with_attention(
        model,
        args.sentence,
        src_vocab,
        tgt_vocab,
        zh_tokenizer,
        device,
    )
    output_path = plot_attention_heatmap(
        source_tokens,
        target_tokens,
        attention,
        args.output,
    )
    print(f"中文: {args.sentence}")
    print(f"英文: {translation}")
    print(f"Attention: {tuple(attention.shape)}")
    print(f"热力图: {output_path}")


if __name__ == "__main__":
    main()
