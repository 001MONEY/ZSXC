"""
参数模块：全局超参数与设备选择。

合并了老师新版 config 的参数（模型结构 / 训练策略 / 日志目录），
并把老师重复定义的项（SEED、DROPOUT_RATE）合并为单一命名。
"""
from pathlib import Path

import torch

path = Path(__file__).resolve().parent


class Config:
    # ── 数据 ──
    FILE_PATH = path / "raw" / "cmn.txt"
    DATA_PATH = path / "raw"
    # 划分训练集和验证集的比例
    TRAIN_VAL_RATE = 0.9
    # 随机种子（词表 raw/zh.json、raw/en.json 基于该种子划分的训练集构建，勿随意改动）
    SEED = 10

    # ── 模型结构 ──
    BATCH_SIZE = 64          # 老师课堂演示值为 3，实际训练用 64（可按显存调整）
    EMBEDDING_DIM = 128
    HIDDEN_SIZE = 64
    ENCODER_LAYERS = 2
    DECODER_LAYERS = 2
    PADDING_IDX = 0
    DROPOUT = 0.3

    # ── 训练策略 ──
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    # Teacher Forcing 线性衰减：前期全用真实词快速收敛，后期保留 50% 真实词学习纠错
    TF_RATIO_START = 1.0
    TF_RATIO_END = 0.5
    # 梯度裁剪阈值，防止梯度爆炸
    CLIP = 1.0
    # Early Stopping 容忍轮数（验证集 loss 连续 N 轮不降则停止）
    PATIENCE = 10
    # 学习率衰减（ReduceLROnPlateau）：val 连续 N 轮不降 → 学习率 × factor
    LR_DECAY_FACTOR = 0.5
    LR_DECAY_PATIENCE = 3

    # ── 日志与权重 ──
    # Attention 版与 0918 无 Attention 对照组分开保存，避免误加载或覆盖旧权重。
    TENSORBOARD_DIR = path / "runs_attention"
    SAVE_DIR = path / "weights"
    CHECKPOINT_NAME = "best_attention.pt"
    CHECKPOINT_PATH = SAVE_DIR / CHECKPOINT_NAME
    ATTENTION_DIR = path / "attention_maps"


def get_device():
    """优先使用 GPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

