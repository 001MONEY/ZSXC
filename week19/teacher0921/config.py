"""
参数模块
"""
from pathlib import Path
import torch
path = Path(__file__).resolve().parent


class Config:
    FILE_PATH = path / "raw" / "cmn.txt"
    DATA_PATH = path / "raw"
    # 划分训练集和验证集的比例
    TRAIN_VAL_RATE = 0.9
    SEED = 10
    BATCH_SIZE = 64
    EMBEDDING_DIM = 128
    HIDDEN_SIZE = 64
    ENCODER_LAYERS = 2
    DECODER_LAYERS = 2
    PADDING_IDX = 0
    DROPOUT_RATE = 0.3
    # 学习率
    LEARNING_RATE = 0.001
    # TensorBoard 日志目录
    TENSORBOARD_DIR = path / "runs_multiplicative"
    # 模型权重保存目录
    SAVE_DIR = path / "weights"
    CHECKPOINT_NAME = "best_multiplicative.pt"
    # 注意力热力图保存目录
    ATTENTION_DIR = path / "attention_maps"
    # 最大训练轮数
    NUM_EPOCHS: int = 50
    # Teacher Forcing 起始比例（训练前期全用真实词，快速收敛）
    TF_RATIO_START: float = 1.0
    # Teacher Forcing 终止比例（训练后期保留50%真实词，学习纠错能力）
    TF_RATIO_END: float = 0.5
    # 梯度裁剪阈值，防止梯度爆炸
    CLIP: float = 1.0
    # Early Stopping 容忍轮数（验证集 loss 连续 N 轮不降则停止）
    PATIENCE: int = 10
    # 学习率衰减（ReduceLROnPlateau）：val 连续 N 轮不降 → 学习率 × factor
    LR_DECAY_FACTOR: float = 0.5
    LR_DECAY_PATIENCE: int = 3


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

