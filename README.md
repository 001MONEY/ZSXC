# 真术相成学习笔记 — Python 与计算机视觉实训

> **学员：** 钱富森  
> **周期：** 2026.05.20 — 2026.07.10（8 周）  
> **内容：** Python 基础 → 数据结构与 GUI → 文件与数据处理 → 计算机视觉 → 深度学习基础 → 深度学习入门与小测验 → CNN 实战 → 经典架构与工业异常检测

---

## 📚 课程周历

| 周次 | 日期 | 主题 | 核心内容 |
|------|------|------|----------|
| **Week 1** | 05.20 - 05.22 | Python 基础语法 | 变量与类型、类型转换、ASCII 编码、循环结构 |
| **Week 2** | 05.25 - 05.29 | 数据结构与 UI | 列表/字典、CRUD 操作、Gradio 图形界面开发 |
| **Week 3** | 06.01 - 06.05 | 文件与数据格式 | 文件 I/O、CSV/JSON/YAML/XML 解析、数据集管理 |
| **Week 4** | 06.08 - 06.12 | 计算机视觉应用 | OpenCV 图像处理、特征匹配、IoU 计算、考试项目 |
| **Week 5** | 06.15 - 06.18 | 深度学习入门 | 神经网络基础、前向传播、损失函数、PyTorch 张量运算、模型保存与加载 |
| **Week 6** | 06.22 - 06.25 | 深度学习入门与小测验 | Pandas 深入、KNN 算法、MySQL 数据库、PyTorch 神经网络、深度学习理论 |
| **Week 7** | 06.29 - 07.03 | 深度学习进阶 - 卷积神经网络 | CNN 原理、卷积/池化、图像分类实战、PyTorch 模型训练调优 |
| **Week 8** | 07.06 - 07.10 | 经典 CNN 架构与实战 | VGG、ResNet、空洞卷积、CIFAR-10 训练、Oxford Pet 二分类 |

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
├── week8/          # 经典 CNN 架构与实战 + VisA 工业异常检测
│   ├── 0706.ipynb              — VGG、ResNet 理论 + 空洞卷积演示
│   ├── 0707.ipynb              — VGG16/ResNet18/ResNet34 训练 CIFAR-10 & Oxford Pet
│   ├── 0708.ipynb              — VisA 数据集分析与以图搜图系统
│   ├── vgg16_cifar10_best.pth  — VGG16 CIFAR-10 最佳模型
│   ├── resnet18_cifar10_best.pth — ResNet18 CIFAR-10 最佳模型
│   ├── resnet34_pet_best.pth   — ResNet34 Oxford Pet 最佳模型
│   ├── checkpoints/            — 训练检查点（5 epoch + best + latest）
│   ├── VisA/                   — VisA 工业异常检测数据集（12 类）
│   └── visa_search/            — 基于内容的图像检索系统（Flask + PyTorch）
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

#### 小测验 `exam.ipynb`（每两周一次）
- **深度学习理论**：激活函数（Sigmoid/Tanh/ReLU/Leaky ReLU/Softmax/Swish）、欠拟合/拟合/过拟合、可微分张量与计算图、期望/方差/标准差、交叉熵损失与 MSE、SQL 练习题
- **人脸特征检测**：`FaceFeatDetect` 类（欧氏距离相似度匹配，top-k 检索）

---

## 🚀 运行环境

- Python 3.x
- 主要依赖：`numpy`, `opencv-python`, `pillow`, `matplotlib`, `gradio`, `pandas`, `pyyaml`, `pymysql`, `torch`, `torchvision`, `scikit-learn`, `flask`

```bash
# 激活虚拟环境（conda）
conda activate D:\project\step1\env
```

---

## 📝 说明

- 每日课程涵盖 **`.ipynb`（Jupyter Notebook）**、**`.py`（纯 Python 脚本）** 和 **`.html`（导出页面）** 三种格式
- 每日配套 **PDF 笔记** 以 `日期姓名.pdf` 命名
- 小测验 `exam.ipynb`（每两周一次）包含深度学习理论（激活函数、过拟合、交叉熵等）与人脸特征检测（欧氏距离匹配）等综合题目

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

| 模块 | 技术 |
|------|------|
| 特征提取 | MobileNetV2（ImageNet 预训练），AdaptiveAvgPool 降维 |
| 特征维度 | 1280 维 |
| 相似度度量 | 余弦相似度 |
| 检索策略 | 遍历特征库，按相似度降序排列取 Top-k |
| 评估指标 | Top-1 / Top-5 准确率 |
| Web 框架 | Flask + HTML/CSS/JS |

### 运行方式

```bash
# 激活环境后，进入 visa_search 目录
cd week8/visa_search

# 完整流程：构建特征库 + 评估 + 演示
python main.py

# 启动 Web 交互界面
python app.py
```
