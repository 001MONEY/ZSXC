# 真术相成学习笔记 — Python 与计算机视觉实训

> **学员：** 钱富森  
> **周期：** 2026.05.20 — 2026.08.21（14 周）  
> **内容：** Python 基础 → 数据结构与 GUI → 文件与数据处理 → 计算机视觉 → 深度学习基础 → 深度学习入门与小测验 → CNN 实战 → 经典架构与工业异常检测 → 小黄人目标检测 → 金鱼目标检测实战 → YOLOv5 目标检测 → 骨龄评估系统与 YOLOv8-pose 关键点检测 → 人体动作识别与 YOLOv8 分割/ONNX 部署 → 模型压缩与 TensorRT 部署 / 度量学习损失

---

## 📚 课程周历

| 周次         | 日期            | 主题              | 核心内容                                           |
| ---------- | ------------- | --------------- | ---------------------------------------------- |
| **Week 1** | 05.20 - 05.22 | Python 基础语法     | 变量与类型、类型转换、ASCII 编码、循环结构                       |
| **Week 2** | 05.25 - 05.29 | 数据结构与 UI        | 列表/字典、CRUD 操作、Gradio 图形界面开发                    |
| **Week 3** | 06.01 - 06.05 | 文件与数据格式         | 文件 I/O、CSV/JSON/YAML/XML 解析、数据集管理              |
| **Week 4** | 06.08 - 06.12 | 计算机视觉应用         | OpenCV 图像处理、特征匹配、IoU 计算、考试项目                   |
| **Week 5** | 06.15 - 06.18 | 深度学习入门          | 神经网络基础、前向传播、损失函数、PyTorch 张量运算、模型保存与加载          |
| **Week 6** | 06.22 - 06.25 | 深度学习入门与小测验      | Pandas 深入、KNN 算法、MySQL 数据库、PyTorch 神经网络、深度学习理论 |
| **Week 7** | 06.29 - 07.03 | 深度学习进阶 - 卷积神经网络 | CNN 原理、卷积/池化、图像分类实战、PyTorch 模型训练调优             |
| **Week 8** | 07.06 - 07.10 | 经典 CNN 架构与实战 | VGG、ResNet、空洞卷积、CIFAR-10 训练、Oxford Pet 二分类、小黄人目标检测（全连接 vs 全卷积对比） |
| **Week 9** | 07.13 - 07.17 | YOLOv3 目标检测 | Darknet-53 骨干、FPN 特征金字塔、YOLOv3 三部分损失函数、解码与 NMS、完整训练循环、little_data 检测实战 |
| **Week 10** | 07.20 - 07.26 | 金鱼目标检测实战 | XML 标注解析、K-Means Anchor 聚类、YOLOv3 完整训练与推理、视频目标检测、损失函数对比优化 |
| **Week 11** | 07.27 - 08.02 | YOLOv5 目标检测 | YOLOv5 架构（CSPDarknet53/C3/SPPF）、训练核心机制、PCB 六类缺陷实战训练、P/R/mAP 评估、detect.py 推理部署 |
| **Week 12** | 08.03 - 08.09 | 骨龄评估系统 & YOLOv8-pose | 两阶段骨龄评估（7类骨检测+9关节分类+RUS计分/数据驱动校准）、RSNA 真实骨龄验证、YOLOv8-pose 解读与 luosi-keypoint 关键点训练 |
| **Week 13** | 08.10 - 08.16 | 动作识别 & 分割与 ONNX 部署 | 人体动作识别（YOLOv8-pose 关键点特征）、Br35H 脑肿瘤分割（YOLOv8-seg）、YOLO 系列 ONNX 部署推理、IoU 多目标跟踪 |
| **Week 14** | 08.17 - 08.21 | 模型压缩 & TensorRT 部署 & 度量学习 | 模型剪枝/蒸馏/量化三件套、TensorRT FP16/INT8 引擎推理对比、CenterLoss/ArcFace 训练 MNIST 二维特征可视化 |
---

## 📂 项目结构

```
step1/
├── week1/          # Python 基础语法
│   ├── 0520.py/.ipynb    — 变量、输入输出
│   ├── 0521.py/.ipynb    — 编码、循环
│   └── 0522.py/.ipynb    — 类型综合练习
│
├── week2/          # 数据结构与 GUI
│   ├── 0525.ipynb        — 字典通讯录 CRUD
│   ├── 0526.ipynb        — Gradio UI 通讯录系统
│   └── 0527~0529.ipynb   — 综合练习
│
├── week3/          # 文件处理与数据格式
│   ├── 0601.ipynb ~ 0605.ipynb  — 逐日课程
│   ├── data/                  — 数据文件
│   ├── dataset/               — 训练/验证集
│   └── MNIST_IMG_TOP9/        — MNIST 手写数字样例
│
├── week4/          # 计算机视觉
│   ├── 0608.ipynb ~ 0612.ipynb  — 逐日课程
│   ├── exam.ipynb              — 小测验：人脸特征检测
│   ├── data/                   — 背景图与素材
│   ├── file/                   — 特征向量文件
│   ├── images/ & images_resized/ — 图片素材
│   └── figures/                — 可视化图表
│
├── week5/          # 深度学习入门
│   ├── 0615.ipynb ~ 0618.ipynb  — 逐日课程
│   ├── MNIST_IMG_TOP9/          — MNIST 手写数字图片
│   ├── model_weights.pth        — 训练好的模型权重
│   └── img_data.txt             — 图片数据文本
│
├── week6/          # 深度学习入门与小测验
│   ├── 0622.ipynb              — Pandas 数据预处理与 KNN 算法详解
│   ├── 0623.ipynb              — MySQL 数据库基础语法
│   ├── 0624.ipynb              — Python 连接 MySQL 实战（PyMySQL）
│   ├── 0625.ipynb              — PyTorch 神经网络拟合正弦函数
│   ├── exam.ipynb              — 小测验（深度学习理论 + 人脸特征检测）
│   ├── zs03-学生管理系统.py     — 学生管理系统（MySQL 版）
│   ├── mysql_db.py             — 数据库连接工具模块
│   ├── titanic_train.csv        — Titanic 数据集
│   └── MNIST_IMG_TOP9/         — MNIST 手写数字图片（测试/训练）
│
├── week7/          # 深度学习进阶 - 卷积神经网络
│   ├── 0629.ipynb              — 卷积神经网络基础（卷积、池化、LeNet）
│   ├── 0701.ipynb              — PyTorch 数据读取与图像分类实战
│   ├── 0702.py                 — CNN 分类脚本（CIFAR-10）
│   ├── 0703.ipynb              — 模型训练调优（MSE vs 交叉熵、SGD、数据增强）
│   ├── 练习题.ipynb            — CNN 练习题
│   ├── cnn_demo.html           — CNN 演示页面
│   ├── 0629.html / 0701.html / 0703.html — 导出 HTML
│   ├── 20260629钱富森.pdf      — 6月29日笔记
│   ├── 20260701钱富森.pdf      — 7月1日笔记
│   ├── 20260703钱富森.pdf      — 7月3日笔记
│   ├── checkpoints/            — 训练好的模型权重
│   │   ├── best_model_aug.pth  — 数据增强模型
│   │   ├── best_model_ce.pth   — 交叉熵损失模型
│   │   ├── best_model_mse.pth  — MSE 损失模型
│   │   └── best_model_sgd.pth  — SGD 优化模型
│   ├── templates/              — HTML 模板
│   └── net.py                  — 网络结构定义
│
├── week8/          # 经典 CNN 架构与实战 + VisA 工业异常检测 + 小黄人目标检测
│   ├── 0706.ipynb               — VGG、ResNet 理论 + 空洞卷积演示
│   ├── 0707.ipynb               — VGG16/ResNet18/ResNet34 训练 CIFAR-10 & Oxford Pet
│   ├── 0708.ipynb               — VisA 数据集分析与以图搜图系统
│   ├── 0710.ipynb               — 小黄人目标检测：数据集生成 → Dataset → 全连接版训练 → 推理 → 全卷积版对比
│   ├── vgg16_cifar10_best.pth   — VGG16 CIFAR-10 最佳模型
│   ├── resnet18_cifar10_best.pth — ResNet18 CIFAR-10 最佳模型
│   ├── resnet34_pet_best.pth    — ResNet34 Oxford Pet 最佳模型
│   ├── checkpoints/             — 训练检查点（5 epoch + best + latest）
│   ├── dataset/                 — 小黄人检测数据集
│   │   ├── bg_pic/              — 背景图（20 张）
│   │   ├── yellow/              — 小黄人 PNG（20 个，含透明通道）
│   │   └── sample/              — 自动生成的训练/测试集（正负样本，YOLO 格式标注）
│   ├── minion_detector_best.pth — 小黄人检测最佳模型权重
│   ├── VisA/                   — VisA 工业异常检测数据集（12 类）
│   └── visa_search/            — 基于内容的图像检索系统（Flask + PyTorch）
│
├── week9/          # YOLOv3 目标检测
│   ├── 0713.ipynb               — YOLOv3 概述：骨干网络 Darknet-53、FPN、锚框、损失函数
│   ├── 0715.ipynb               — YOLOv3 详解：训练与预测流程、数据集标注格式
│   ├── 0716.ipynb               — 损失函数详解：坐标损失 + 置信度损失 + 分类损失
│   ├── 0717.ipynb               — 完整实现：网络结构 + Train 类 + NMS 详解 + little_data 训练 + 推理可视化
│   ├── net.py                   — YOLOv3 练习框架（含 TODO 注释）
│   ├── net_full.py              — YOLOv3 完整实现（Darknet53 → YOLOHead → 解码 → NMS → 检测）
│   ├── yolov3.weights           — Darknet 预训练权重（需下载，已 .gitignore）
│   ├── little_data/             — 小样本训练数据集（18 张图片，4 类：人/猫/狗/马）
│   │   ├── images/              — 图片文件（01.jpg ~ 18.jpg）
│   │   ├── Parse_label.txt      — YOLO 格式标签（像素坐标）
│   │   ├── outputs_voc/         — PascalVOC 格式 XML 标注
│   │   └── 图片缩放.py / 标签绘制测试.py / Parse_xml_pascalvoc.py — 数据处理脚本
│   └── checkpoints_little/      — little_data 训练检查点（已 .gitignore）
│
├── week10/         # 金鱼目标检测实战
│   ├── 0720.ipynb / 0724.ipynb  — 课程笔记
│   ├── train_yolo_xml.py        — YOLOv3 训练（XML标注 + K-Means Anchor + Ignore区域 + BCE分类）
│   ├── detect_mp4.py            — 视频推理（向量化解码 + torchvision NMS + sigmoid分类）
│   ├── train.py / trainv2.py    — 参考训练脚本
│   ├── checkpoints/             — 训练权重（yolov3_best.pth，已 .gitignore）
│   ├── 图片文件/                — 85张训练图片（已 .gitignore）
│   ├── 标注文件/                — 85个 Pascal VOC XML 标注（已 .gitignore）
│   ├── 1.mp4                   — 测试视频（已 .gitignore）
│   ├── output_detected.mp4      — 输出检测视频（已 .gitignore）
│   ├── 参考代码/                — YOLOv3 教学参考实现（7个模块）
│   └── yolov3_bug/             — 原始有Bug版本代码（已 .gitignore）
│
├── week11/         # YOLOv5 目标检测
│   ├── 0727.ipynb               — Day1：项目结构与模型架构（CSPDarknet53、C3、SPPF）
│   ├── 0728.ipynb / 0728.html   — Day2：训练核心机制（parse_model、损失、Mosaic、训练循环）
│   ├── 0729.ipynb / 0729.html   — Day3：PCB 六类缺陷实战训练 + P/R/mAP 评估
│   ├── 0731.ipynb / 0731.html   — Day4：detect.py 推理部署 + 评估指标详解
│   ├── 20260728~31钱富森.pdf    — 逐日 PDF 笔记
│   ├── yolov5-7.0/              — YOLOv5 源码框架（train/val/detect/export + models/utils/data）
│   │   └── runs/                — 训练结果（exp4: best.pt、results.png、PR_curve.png 等）
│   ├── PCBYOLODataset/          — PCB 缺陷数据集（YOLO 格式，images + labels + pcb.yaml）
│   ├── PCB-VOC/                 — PCB VOC 标注 + 转换脚本（labelimg2yolo / read_xml / 切图）
│   ├── datasets/                — 配套数据集（coco128 等）
│   └── ultralytics-8.4.113/     — ultralytics 库（已 .gitignore）
│
├── week12/         # 骨龄评估系统 + YOLOv8-pose 关键点检测
│   ├── 0803.ipynb / 0805.ipynb / 0807.ipynb (+ .html) — 逐日课程笔记
│   ├── 骨龄评估系统研发方案v2.md     — 两阶段骨龄评估方案文档
│   ├── 蔬菜分类训练.md / qianyixuexi.py — 蔬菜分类 & 迁移学习
│   ├── Bone Age Assessment/      — 骨龄评估代码（检测/分类/计分/校准/pipeline + Qt 界面）
│   ├── handbone/                 — 手骨检测数据集（VOC 格式, 881张, 7类）
│   ├── arthrosis/                — 9 关节等级分类数据集
│   ├── rsna_tmp/                 — RSNA 骨龄挑战赛数据（真实骨龄标签）
│   ├── luosi-keypoint/           — 螺丝 6 关键点数据集（LabelMe 标注, 已 .gitignore）
│   └── ultralytics-8.4.113/      — ultralytics 源码（已 .gitignore）
│
├── week13/         # 人体动作识别 + 脑肿瘤分割 + YOLO ONNX 部署
│   ├── 0810.ipynb / 0810.html        — 基于 YOLOv8-pose 的人体动作识别（9 类动作）
│   ├── 0812_onnx.ipynb               — YOLOv5 ONNX 部署推理教程
│   ├── 0812.ipynb / 0812.html        — YOLOv5/YOLOv8/YOLOv8-pose ONNX 推理
│   ├── 0813.ipynb / 0813.html        — YOLOv8 视频检测 + IoU 多目标跟踪
│   ├── seg_train.py                  — YOLOv8s-seg 脑肿瘤分割训练
│   ├── seg_detect.py                 — 肿瘤掩码黑底抠图部署
│   ├── smoke_test.py                 — 6GB 显存冒烟测试（1 epoch）
│   ├── 20260810/12/13钱富森.pdf      — 逐日 PDF 笔记
│   ├── action_train/                 — 9 类动作数据集（已 .gitignore）
│   ├── action_train_pose_out/        — 关键点检测可视化输出（已 .gitignore）
│   ├── feats.txt                     — 34 维关键点特征库（已 .gitignore）
│   ├── Br35HDet/                     — 脑肿瘤 Br35H 数据集与转换脚本
│   │   ├── change_json_key_1.py      — 标注 JSON key 规范化
│   │   ├── cv_lablme_2.py            — VIA regions → LabelMe 标准格式
│   │   ├── cv_yolov8seg_3.py         — LabelMe → YOLOv8-seg 格式
│   │   ├── split_train_val_4.py      — train/val 划分 + dataset.yaml
│   │   ├── Br35HDet/                 — 原始图片与标注（已 .gitignore）
│   │   ├── yolo_all/                 — YOLO-seg 中间产物（已 .gitignore）
│   │   └── yolodataset/              — 划分后的训练/验证集（已 .gitignore）
│   ├── runs/                         — 分割训练输出 seg_train/smoke_test（已 .gitignore）
│   ├── onnx_models/                  — ONNX 模型 yolov5su/yolov8n/yolov8n-pose（已 .gitignore）
│   ├── tumor_crop_out/               — 肿瘤抠图输出（已 .gitignore）
│   └── car.mp4 / car2.mp4            — 检测/跟踪测试视频（已 .gitignore）
│
├── week14/         # 模型压缩 + TensorRT 部署 + 度量学习损失
│   ├── 0817.ipynb / 0817.html        — 模型压缩三件套：剪枝 / 蒸馏 / 量化 + TensorRT/OpenVINO 导出
│   ├── 0819.ipynb / 0819.html        — TensorRT engine 推理：FP16 vs INT8 速度/精度对比
│   ├── 0821.ipynb / 0821.html        — CenterLoss / ArcFace 训练 MNIST 二维特征可视化 + YOLOv8 ONNX 推理
│   ├── 20260817/19/21钱富森.pdf      — 逐日 PDF 笔记
│   ├── trt_utils.py                  — TensorRT engine 加载 / IO 打印工具
│   ├── params/                       — MobileNet / 剪枝 / 重训模型权重与 ONNX
│   ├── data/                         — MNIST 数据集
│   └── yolo_engine_out/              — YOLO TensorRT 引擎输出图（已 .gitignore）
│
├── env/            # Python 虚拟环境（已忽略）
├── .gitignore
└── README.md
```

---

## 🔑 各周重点

### Week 1 — Python 基础

- `print` / `input` 基本输入输出
- 数据类型：`int`、`float`、`bool`、`str`
- 类型转换与 ASCII 编码（`ord` / `chr`）
- `while` 循环结构

### Week 2 — 数据结构与界面

- 列表、字典的增删改查
- 通讯录系统（纯数据层）
- Gradio 框架快速构建 Web UI（MVC 分层）
- DataFrame 表格展示

### Week 3 — 文件与数据处理

- 文件读写模式（`r` / `w` / `a`）
- CSV、JSON、YAML、XML 格式解析
- 文件存在性检查与异常处理
- 用户注册数据管理系统
- Matplotlib 基础可视化

### Week 4 — 计算机视觉

- OpenCV 图像读写、缩放、色彩通道操作
- 九宫格分割与拼接还原
- 图片合成（小黄人贴图、渐变图拼接）
- 验证码生成
- **IoU（交并比）计算与可视化**
- **人脸特征检测** — `FaceFeatDetect` 类（欧氏距离相似度匹配）
- 视频处理

### Week 5 — 深度学习入门

- 神经网络基本概念（输入层、隐藏层、输出层）
- **前向传播与反向传播** 原理
- 损失函数（MSE、交叉熵）
- **PyTorch** 框架入门：张量运算、自动求导
- 模型定义、训练与保存（`model_weights.pth`）
- 加载预训练模型进行推理

### Week 6 — 深度学习入门与小测验（06.22 - 06.25）

#### 6月22日 — Pandas 深入与 KNN 算法

- **Pandas 数据预处理**：`isnull()` / `dropna()` / `fillna()` / `drop_duplicates()` / `apply()` / `map()` / `sort_values()` / `value_counts()`
- **KNN（K-近邻）算法**：核心思想（近朱者赤）、k 值选择、距离度量、优缺点
- sklearn 实现 KNN 分类（Iris 数据集）

#### 6月23日 — MySQL 数据库基础

- MySQL CRUD（`CREATE` / `DROP` / `ALTER` / `INSERT` / `SELECT` / `UPDATE` / `DELETE`）
- 数据类型、约束（`PRIMARY KEY` / `FOREIGN KEY` / `NOT NULL` / `UNIQUE` / `DEFAULT`）
- 查询进阶：`WHERE` / `LIKE` / `ORDER BY` / `LIMIT` / `GROUP BY` / `HAVING` / 聚合函数
- 表关联：`JOIN` / 子查询（`IN` / `EXISTS`）

#### 6月24日 — Python 连接 MySQL 实战

- **PyMySQL** 库操作 MySQL：连接配置、游标、事务提交
- 学生管理系统（MySQL 版）：`zs03-学生管理系统.py`
- 超市营业额 Excel 数据分析

#### 6月25日 — PyTorch 神经网络实践

- `nn.Sequential` 搭建全连接网络拟合正弦曲线
- GPU 训练检测（`torch.cuda`）
- 训练流程：前向传播 → 损失计算 → 反向传播 → 参数更新
- 损失曲线可视化

### Week 7 — 深度学习进阶：卷积神经网络（06.29 - 07.03）

#### 6月29日 — 卷积神经网络基础

- **CNN 核心概念**：卷积核（Filter）、特征图（Feature Map）、步长（Stride）、填充（Padding）
- **池化层**：最大池化（MaxPooling）、平均池化（AvgPooling）
- **LeNet-5 架构**：经典 CNN 结构剖析
- **PyTorch 实现**：`nn.Conv2d` / `nn.MaxPool2d` / `nn.Linear`

#### 7月1日 — PyTorch 数据读取与图像分类

- `torchvision.datasets` 加载内置数据集（FashionMNIST、CIFAR-10）
- `DataLoader` 批量加载与数据打乱
- **CNN 图像分类完整流程**：数据加载 → 模型定义 → 训练 → 评估
- 测试集准确率评估与可视化

#### 7月2日 — CNN 分类脚本

- `net.py` — 可复用的网络结构定义
- `0702.py` — CIFAR-10 分类训练脚本

#### 7月3日 — 模型训练调优对比实验

- **损失函数对比**：MSE vs 交叉熵（CrossEntropyLoss）
- **优化器对比**：SGD vs Adam
- **数据增强效果**：有无数据增强对准确率的影响
- 四种模型分别保存：`best_model_mse.pth` / `best_model_ce.pth` / `best_model_sgd.pth` / `best_model_aug.pth`
- 训练曲线对比可视化

### Week 8 — 经典 CNN 架构与实战（07.06 - 07.10）

#### 7月6日 — VGG 与 ResNet 理论

- **空洞卷积（Dilated/Atrous Convolution）**：原理、dilation rate、等效感受野计算、参数量不变性
- **VGG 网络**：小卷积核堆叠思想、VGG11/13/16/19 配置对比、参数量分析
- **ResNet 核心思想**：退化问题、残差连接 $F(x)+x$、BasicBlock vs Bottleneck
- **ResNet 系列**：ResNet-18/34/50/101/152 参数量对比、梯度流通可视化

#### 7月6日（实践） — VGG16 训练 CIFAR-10

- 自定义 `VGG_CIFAR` 类（适配 32×32 小图）
- 数据增强：RandomCrop + RandomHorizontalFlip
- SGD + CosineAnnealingLR 学习率调度
- 训练曲线可视化、预测结果展示

#### 7月7日 — ResNet18/34 训练 CIFAR-10 & Oxford Pet

- **ResNet18 适配 CIFAR-10**：修改首层卷积（7×7 → 3×3）、移除 MaxPool
- **ResNet34 二分类（Oxford Pet）**：自定义 `OxfordPetDataset`、ImageNet 预训练微调
- **Oxford Pet 数据集**：37 种猫狗品种，二分类（猫 vs 狗），99.78% 测试准确率

#### 7月8日 — VisA 工业异常检测与以图搜图

- **VisA 数据集分析**：12 类工业产品（candle、capsules、pcb 等），含 Anomaly/Normal 图片及 Mask
- **以图搜图系统**：基于 MobileNetV2 特征提取 + 余弦相似度检索
- **图像检索全流程**：特征提取 → 特征库构建 → 相似度搜索 → Top-k 评估 → 可视化
- **检索性能**：Top-1 ~75%+、Top-5 ~92%+（MobileNetV2 1280 维特征）
- **Flask Web 应用**：`visa_search/app.py` — 上传图片检索相似样本，支持 Web 界面交互

#### 7月10日 — 小黄人单目标检测

- **数据集自动生成**：20 张背景图 + 20 个小黄人 PNG（含透明通道）→ 合成正负样本，YOLO 格式标注
- **数据增强**：随机位置/尺寸、水平翻转
- **全连接版网络 (Net)**：8 层 Conv + MaxPool → Flatten → Linear(512,5)，输出 [x1,y1,x2,y2,conf]
- **训练策略**：BCELoss（分类）+ MSELoss（回归，仅正样本），CosineAnnealingLR，~9.5M 参数
- **推理回原图**：网络输出归一化坐标 → 反归一化 → 原图画框，完整流程演示
- **全卷积版网络 (FullyConvNet V2)** ⭐：保留 9×9 Grid 空间，每个 Grid Cell 独立预测 → 取最高置信度 Cell 解码，~3.4M 参数，更轻量、更符合检测器本质
- **拓展对比**：全连接版（FC head）vs 全卷积版（Conv head），参数量、收敛速度对比

---

### Week 9 — YOLOv3 目标检测（07.13 - 07.17）

#### 7月13日 — YOLOv3 概述与网络结构

- **YOLO 系列发展**：YOLOv1（回归思想）→ YOLOv2（BatchNorm、高分辨率、Anchor）→ YOLOv3（多尺度+FPN）
- **Darknet-53 骨干网络**：53 层卷积（含残差连接），无池化层（stride=2 降采样），相比 ResNet-152 速度更快、精度相当
- **特征金字塔网络（FPN）**：3 个尺度输出（13×13、26×26、52×52），上采样 + 横向连接融合高低层语义
- **Anchor 机制**：9 个锚框（大/中/小各 3 个），K-Means 聚类得到
- **$S \times S$ 网格思想**：每格预测 3 个 anchor，每个 anchor 预测 $(5+C)$ 个值

#### 7月15日 — YOLOv3 训练与预测流程

- **标签编码（Training）**：归一化坐标 → 网格映射 → 计算 Anchor 偏移 → one-hot 类别 → 填充三个尺度的 label 张量
- **解码（Inference）**：$\sigma(t_x)+c_x$、$\sigma(t_y)+c_y$、$p_w e^{t_w}$、$p_h e^{t_h}$ → 边界框坐标
- **数据集标注格式**：`img_path cls cx cy w h` 归一化坐标

#### 7月16日 — YOLOv3 损失函数详解

- **三部分损失**：$L = \lambda_{\text{coord}} L_{\text{coord}} + L_{\text{obj}} + L_{\text{cls}}$
- **坐标损失**：MSE 回归 $(t_x,t_y,t_w,t_h)$，仅正样本参与，$\lambda_{\text{coord}}=5$ 强调定位
- **置信度损失**：BCE 判断有无目标，负样本 $\lambda_{\text{noobj}}=0.5$ 降权（正负样本比 1:100+）
- **分类损失**：BCE 而非 Softmax（多标签非互斥），每个类别独立二分类
- **Sigmoid vs Softmax**：一个目标可同时属于"狗"和"动物"

#### 7月17日 — YOLOv3 完整实现

- **网络结构**：`Darknet53` + `YOLOHead` + `YOLOv3`（FPN 组装）
- **`Train` 类**：完整训练循环含三部分损失、梯度裁剪、MultiStepLR 调度、Checkpoint 保存/加载
- **`decode_scale`**：全向量化解码（3 尺度 × 每个 anchor 独立解码）
- **`post_process` / `nms`**：非极大值抑制（按置信度排序 → 循环保留最高分框 → 删除 IoU > 0.45 同类框）
- **`detect_image`**：预处理 → 推理 → 解码 → NMS → 坐标还原 → 原图画框
- **little_data 实战**：18 张图片、4 类（人/猫/狗/马），训练 100 epoch → 推理可视化

---

### Week 10 — 金鱼目标检测实战（07.20 - 07.26）

基于 **YOLOv3** 对鱼缸视频中三种金鱼（红/黑/白）进行检测与分类，85 张图片训练，427 帧视频推理。

**核心技术**：K-Means Anchor聚类、Ignore区域、BCE分类(训推一致)、Sigmoid on xy、向量化解码、torchvision NMS

**损失函数演变**：v1(双mean,CE)→v2(正mean+负sum/n,CE)→v3(sum+Ignore,BCE+sigmoid(xy), 15:1权重)

**文件**：`train_yolo_xml.py`(训练)、`detect_mp4.py`(推理)、`参考代码/`(教学参考)、`yolov3_bug/`(原Bug版)

**局限**：仅85张数据过拟合、13/26尺度检出少、需增加数据量和数据增强

---

### Week 11 — YOLOv5 目标检测（07.27 - 08.02）

基于 **YOLOv5** 对 PCB 电路板六类缺陷进行检测，完整走通「理论 → 源码 → 训练 → 评估 → 推理部署」全流程，最终达到 **mAP@0.5=92.1%、P=98.4%、83 FPS** 的工业可用水平。

#### 7月27日 — 项目结构与模型架构

- **项目结构**：顶层脚本（`detect.py` / `train.py` / `val.py` / `export.py`）+ `models/` + `utils/` + `data/`
- **YOLOv5 vs YOLOv3 六大改进**：CSPDarknet53（计算量 -20%）、FPN+PAN 双向融合、SiLU 激活、3 anchor × 3 grid 正样本、CIoU 回归、Mosaic+MixUp+Multi-Scale 增强
- **C3 模块**：双通道（深层提取 + 原始信息保留）拼接融合
- **SPPF**：1 个 MaxPool 串行 3 次 → 感受野 5/9/13，比 SPP 快 2~3 倍
- **n/s/m/l/x 五档模型**：改 `gd`（深度）/`gw`（宽度）两个系数即可切换

#### 7月28日 — 训练核心机制

- **`parse_model()`**：从 YAML 配置生成 `nn.Sequential` 模型
- **三部分损失**：$L = \lambda_{box}\text{CIoU} + \lambda_{obj}\text{BCE} + \lambda_{cls}\text{BCE}$（权重 0.05 / 1.0 / 0.5）
- **`build_targets` 正样本匹配**：每个 GT 匹配 3 anchor × 3 相邻 grid = 9 个正样本
- **数据加载**：Mosaic（4 图拼接，变相 batch×4）、MixUp、HSV 增强、Rect 矩形训练（提速 30%）
- **训练技巧**：Warmup、Multi-Scale（320~960）、梯度累积、AMP 混合精度、EMA、EarlyStopping

#### 7月29日 — PCB 实战训练 + 评估

- **数据集**：PCB 六类缺陷（`missing_hole` 缺孔 / `mouse_bite` 缺口 / `open_circuit` 断路 / `short` 短路 / `spur` 毛刺 / `spurious_copper` 多余铜）
- **配置**：YOLOv5s 预训练、640×640、batch=16、100 epochs（RTX 3060 约 3.1h）
- **结果**：mAP@0.5=**92.1%**、mAP@0.5:0.95=**63.5%**、P=**98.4%**、R=**90.6%**
- **六类表现**：open_circuit 95.9%、spur 94.1%、missing_hole 93.2%、short 92.5%、spurious_copper 88.9%、mouse_bite 88.2%
- **评估产物**：`results.png` / `PR_curve.png` / `confusion_matrix.png` / `F1_curve.png` / `best.pt`
- **推理速度**：~12ms/张（83 FPS），达实时检测水平

#### 7月31日 — 推理部署（detect.py）

- **流程**：`DetectMultiBackend` 加载模型 → `LoadImages` 数据加载 → 前向推理 + NMS → `scale_boxes` 缩放回原图 → 标注保存
- **关键参数**：`--conf-thres`（置信度阈值，调高 → P↑ R↓）、`--iou-thres`（NMS 阈值）、`--save-txt`、`--device`
- **评估指标总结**：IoU、TP/FP/FN、P（管误检）、R（管漏检）、AP（PR 曲线下面积）、mAP@0.5（宽松标准）vs mAP@0.5:0.95（严格标准）

**学习主线**：`yaml 配置 → parse_model 建模型 → Mosaic 数据加载 → ComputeLoss 三损失 → train 训练（Warmup/AMP/EMA）→ val 评估（P/R/mAP）→ detect 推理部署`

### Week 12 — 骨龄评估系统 + YOLOv8-pose 关键点检测（08.03 - 08.09）

#### 🦴 骨龄评估系统（两阶段方案）

- **方案**：`7类骨检测(YOLOv8) → 9关节等级分类 → RUS计分/数据驱动校准 → 骨龄`
- **检测**：`handbone/` 881 张手骨图（VOC→YOLO），bone7_ft 模型 **mAP50=0.991 / mAP50-95=0.599**，两段式迁移学习（先 freeze 主干再全量微调）
- **分类**：`arthrosis/` 9 关节（DIP/PIP/MCP/MIP/Radius/Ulna…）等级 1-14，ResNet18 分类 + CORN 序数回归（解决 Ulna 弱区分度问题，等级 MAE 2.48→1.67）
- **计分/校准**：13 根 RUS 骨得分特征 → GradientBoosting 回归骨龄，全量 1278 张训练 **MAE=12.83 月（1.07 岁）、相关 0.884**
- **端到端**：`pipeline.py`（检测→`filter_bones.py` 过滤 13 骨→分类→计分→骨龄）+ `gradio_app.py` / `qt_server.py` 界面
- **关键发现**：`arthrosis` 部分骨头等级标注与真实成熟度脱节（桡骨权重 210/1000 却是噪声）→ 用 RSNA 真实骨龄标签做**数据驱动校准**解决，端到端精度提升 4 倍

#### 🔩 YOLOv8-pose 关键点检测（0807.ipynb）

- **架构解读**：逐层分析 `yolov8-pose.yaml` —— Backbone（C2f/SPPF）+ Neck（FPN+PAN）+ Pose 头（框 cv2 / 类 cv3 / 关键点 cv4），关键点解码公式 $(2p + a - 0.5)\times s$
- **实战训练** `luosi-keypoint`：螺丝 6 顶点关键点检测，639 张（LabelMe→YOLO-pose），`yolov8n-pose` 迁移学习（kpt_shape 17→6 自动重建 Pose 头），100 epochs 仅 8.6 分钟
- **结果**：**box mAP50=0.995 / mAP50-95=0.897，pose mAP50=0.823 / mAP50-95=0.657**，单张推理 2.8ms

#### 🥦 其他

- 蔬菜分类训练（迁移学习实战）与 `qianyixuexi.py` 迁移学习脚本

---

### Week 13 — 人体动作识别 + 脑肿瘤分割 + ONNX 部署（08.10 - 08.16）

#### 🕺 人体动作识别（0810.ipynb）

- **9 类动作数据集** `action_train/`：left / right / pause / shut / up / 0k / right_fly / left_fly / open
- **YOLOv8n-pose 批量关键点提取**：COCO 预训练权重直接推理，输出 17 个人体关键点，可视化保存到 `action_train_pose_out/`
- **`PoseDetect` 特征提取**：34 维归一化特征（以双肩中点为原点、肩宽为尺度归一化 → 抗平移/缩放；不可见关键点置 0）
- **`PoseFeatureExtractor` + `FeatureLibrary`**：按动作类别构建模板特征库，生成 `feats.txt` 供动作匹配识别

#### 🚗 YOLO 系列 ONNX 部署推理（0812_onnx.ipynb / 0812.ipynb）

- **YOLOv5 ONNX**：输入 1×3×640×640、输出 1×84×8400，拆分「前处理 → 推理 → 后处理」，OpenCV NMS 去重（conf=0.5, IoU=0.45）
- **YOLOv8 ONNX**：`yolov8n.onnx` 推理，COCO 80 类，逐类着色标注
- **YOLOv8-pose ONNX**：`yolov8n-pose.onnx` 关键点推理，`filter_pose` 后处理（置信度过滤 + 坐标还原）+ 17 关键点骨架可视化

#### 🎥 视频检测与多目标跟踪（0813.ipynb）

- **`yolov8_detector`**：YOLOv8 逐帧检测 `car2.mp4`（3411 帧）
- **`IOU_tracker`**：IoU 贪心匹配多目标跟踪器 —— 检测 → IoU 匹配 → 更新/新建/删除轨迹，为每个目标分配稳定 ID；`lost` 计数容忍短暂遮挡/漏检
- 输出带跟踪标注视频 `track_out2.mp4`（画框 + ID）

#### 🧠 Br35H 脑肿瘤分割（YOLOv8-seg）

- **数据转换流水线**：`change_json_key_1.py`（key 规范化）→ `cv_lablme_2.py`（VIA regions → LabelMe）→ `cv_yolov8seg_3.py`（LabelMe → YOLOv8-seg）→ `split_train_val_4.py`（train/val 划分 + `dataset.yaml`）
- **冒烟测试** `smoke_test.py`：验证 imgsz=512、batch=16 在 6GB 显存可正常训练（1 epoch）
- **训练** `seg_train.py`：YOLOv8s-seg 预训练迁移学习，100 epochs（关 AMP 自动检查）
- **结果**：Box **mAP50=0.945 / mAP50-95=0.813**（P=0.894, R=0.898）；Mask **mAP50=0.919 / mAP50-95=0.763**
- **部署抠图** `seg_detect.py`：验证集肿瘤掩码黑底抠图 → `tumor_crop_out/`（每张肿瘤独立保存）

---

### Week 14 — 模型压缩 + TensorRT 部署 + 度量学习损失（08.17 - 08.21）

#### ✂️ 模型压缩三件套（0817.ipynb）

- **剪枝（Pruning）**：`torch.nn.utils.prune` 对 MobileNetV2 剪枝——把数值接近 0 的不重要权重清零，模型更省内存、推理更快、精度基本不掉；剪枝后重训（retrain）恢复精度
- **蒸馏（Distillation）**：大模型（教师）软标签指导小模型（学生）学习，用更小的模型逼近大模型精度
- **量化（Quantization）**：FP16 / INT8 低精度表示，模型体积大幅缩小
- **部署导出**：TensorRT FP16/INT8 engine 与 OpenVINO INT8 导出
- **产物**：`params/` 下 `model_mobilenet.pt` / `model_prune.pt` / `model_retrain.pt` / `mobilenet.onnx` / `prune.onnx` / `retrain.onnx`

#### ⚡ TensorRT 部署（0819.ipynb）

- **`trt_utils.py`**：TensorRT engine 加载工具（剥离 ultralytics 导出的元数据头）+ IO 信息打印
- **FP16 vs INT8 引擎对比**：`yolov8n_fp16.engine` / `yolov8n_int8.engine`（COCO 80 类）
- **性能**：FP16 ~25ms/张、INT8 ~20ms/张（含前后处理）；`mobilenet.onnx` 转 FP16 引擎约 55s、6MB

#### 📐 度量学习损失（0821.ipynb）

- **CenterLoss**：让同类特征向类中心聚拢，类内更紧凑（$L = L_{cls} + \lambda L_{center}$）
- **ArcFace（加性角度间隔）**：特征/权重 L2 归一化后对真实类别角度加 margin：$L=-\log\dfrac{e^{s\cos(\theta_{y_i}+m)}}{e^{s\cos(\theta_{y_i}+m)}+\sum_{j\neq y_i}e^{s\cos\theta_j}}$
- **踩坑修复**：`cos_theta` 不能 clamp 到 ±1（`sin_theta=0` 导致梯度除以 0 → NaN、图片空白），应 clamp 到 $\pm(1-10^{-7})$
- **结果**：2 维特征可视化，ArcFace 200 epochs 训练集 acc 最高 **99.88%**，loss 收敛到由 margin 决定的理论下限 ~0.70
- **其他**：Python 装饰器示例（计时/工厂/叠加/类装饰器）、YOLOv8 ONNX 推理

---

## 🧪 VisA 以图搜图项目

`visa_search/` 是一个基于内容的图像检索（CBIR）系统，在 VisA 工业异常检测数据集上实现相似图片搜索。

### 项目结构

```
visa_search/
├── main.py                  # 主入口：特征库构建 + 评估 + 演示
├── model.py                 # 特征提取模型（MobileNetV2 / ResNet18）
├── dataset.py               # 数据集样本收集与统计
├── feature_lib.py           # 特征库构建、加载、搜索、评估
├── config.py                # 配置文件（路径、模型、参数）
├── app.py                   # Flask Web 应用（交互式检索）
├── import_to_db.py          # 特征入库脚本
├── visualize.py             # 可视化工具（检索结果展示）
├── feats_visa.txt           # 预提取特征库文件
├── static/                  # 静态资源
└── templates/
    └── index.html           # Web 前端页面
```

### 技术要点

| 模块     | 技术                                           |
| ------ | -------------------------------------------- |
| 特征提取   | MobileNetV2（ImageNet 预训练），AdaptiveAvgPool 降维 |
| 特征维度   | 1280 维                                       |
| 相似度度量  | 余弦相似度                                        |
| 检索策略   | 遍历特征库，按相似度降序排列取 Top-k                        |
| 评估指标   | Top-1 / Top-5 准确率                            |
| Web 框架 | Flask + HTML/CSS/JS                          |

### 运行方式

```bash
# 激活环境后，进入 visa_search 目录
cd week8/visa_search

# 完整流程：构建特征库 + 评估 + 演示
python main.py

# 启动 Web 交互界面
python app.py
```
