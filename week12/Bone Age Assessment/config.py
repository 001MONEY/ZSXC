# -*- coding: utf-8 -*-
"""
全局配置：路径与类别常量
所有数据准备 / 训练 / 推理脚本统一从这里读取路径，保证一致性。
"""
from pathlib import Path

# ---------------- 目录定义 ----------------
BAA_DIR = Path(__file__).resolve().parent          # Bone Age Assessment/
WORKSPACE = BAA_DIR.parent                          # week12/（原始数据所在）
DATASETS = BAA_DIR / "datasets"                     # 处理后数据集统一放这里

# 原始数据位置
HANDBONE = WORKSPACE / "handbone"                   # 检测数据（VOC 格式）
ARTHROSIS = WORKSPACE / "arthrosis"                 # 分类数据（关节类型/等级）

# 处理后数据位置
DETECTION_DIR = DATASETS / "detection"              # YOLO 格式检测数据集
CLASSIFICATION_DIR = DATASETS / "classification"    # ImageFolder 格式分类数据集

# ---------------- 检测模型 7 类（对应 VOC XML 中的 <name>）----------------
DET_CLASSES = [
    "Radius",            # 桡骨
    "Ulna",              # 尺骨
    "MCPFirst",          # 第一掌骨（拇指）
    "MCP",               # 掌骨
    "ProximalPhalanx",   # 近节指骨
    "MiddlePhalanx",     # 中节指骨
    "DistalPhalanx",     # 远节指骨
]
DET_CLASS2ID = {name: i for i, name in enumerate(DET_CLASSES)}

# ---------------- 分类模型 9 个关节类型（对应 arthrosis/ 下的目录）----------------
JOINT_TYPES = [
    "DIP", "DIPFirst", "MCP", "MCPFirst",
    "MIP", "PIP", "PIPFirst", "Radius", "Ulna",
]

# ---------------- 划分参数 ----------------
TRAIN_RATIO = 0.8        # 训练集比例（val 占 20%）
SEED = 42                # 随机种子（保证可复现）
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
