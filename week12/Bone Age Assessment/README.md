# Bone Age Assessment（骨龄评估系统）

基于"检测 + 分类 + 计分"两阶段方案（见 `../骨龄评估系统研发方案v2.md`）的骨龄评估系统。

## 目录结构

```
Bone Age Assessment/
├── config.py                      # 全局路径与类别常量（所有脚本共用）
├── voc_to_yolo.py                 # 检测数据：VOC → YOLO 格式 + train/val 划分
├── prepare_classification.py      # 分类数据：arthrosis/ → ImageFolder 结构划分
├── preprocess.py                  # 预处理基线：灰度化 + 中值滤波 + CLAHE
└── datasets/                      # 处理后的数据集（脚本自动生成）
    ├── detection/                 # YOLO 检测数据集（含 data.yaml）
    ├── detection_pre/             # 预处理版检测数据集（训练用）
    ├── classification/{关节}/     # 9 个关节类型各自的分类数据集
    └── classification_pre/{关节}/ # 预处理版分类数据集
```

## 数据准备

```bash
# 检测数据（handbone/ VOC → datasets/detection/ YOLO）
python voc_to_yolo.py
python voc_to_yolo.py --dry-run    # 只看统计不落盘

# 分类数据（arthrosis/ → datasets/classification/）
python prepare_classification.py
python prepare_classification.py --dry-run
python prepare_classification.py --joints DIP Radius   # 只处理指定关节

# 预处理基线（灰度化 + 中值滤波 + CLAHE，生成 *_pre 数据集）
python preprocess.py                 # 全量处理检测 + 分类
python preprocess.py --detection     # 只处理检测
python preprocess.py --preview       # 只看前后对比图（output/preview/）
```

## 数据说明

- **检测数据**：`handbone/` 共 881 张左手腕 X 光片，7 类（Radius / Ulna / MCPFirst / MCP / ProximalPhalanx / MiddlePhalanx / DistalPhalanx）
- **分类数据**：`arthrosis/` 9 个关节类型（DIP / DIPFirst / MCP / MCPFirst / MIP / PIP / PIPFirst / Radius / Ulna），每类按发育等级分文件夹
- 划分比例 train:val = 8:2，随机种子 42，可复现

## 下一步

1. 检测模型训练：`datasets/detection/data.yaml` + YOLOv8 迁移学习
2. 分类模型训练：`datasets/classification/{关节}/` 用 ImageFolder 训练 9 个分类模型
