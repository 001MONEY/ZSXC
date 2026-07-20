"""
VisA 以图搜图 — 配置文件
"""
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VISA_ROOT = r'D:\project\step1\week8\VisA\data\VisA_20220922'

CATEGORIES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum',
]

SUBSETS = ['Normal', 'Anomaly']       # 正常 / 异常

# ============================================================
# 特征库参数
# ============================================================
MAX_PER_CLASS = 50                     # 每类每子集最多取多少张
FEAT_DIM = 1280                        # 特征维度 (MobileNetV2)
FEAT_FILE = os.path.join(BASE_DIR, 'feats_visa.txt')   # 绝对路径，不受运行目录影响

# ============================================================
# 模型参数
# ============================================================
MODEL_NAME = 'mobilenet_v2'            # 讲义示例用的是 MobileNetV2

# ============================================================
# 设备
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
