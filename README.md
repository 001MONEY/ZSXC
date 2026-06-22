# 真术相成学习笔记 — Python 与计算机视觉实训

> **学员：** 钱富森  
> **周期：** 2026.05.20 — 2026.06.22（6 周）  
> **内容：** Python 基础 → 数据结构与 GUI → 文件与数据处理 → 计算机视觉 → 深度学习基础 → 数据分析与预处理

---

## 📚 课程周历

| 周次 | 日期 | 主题 | 核心内容 |
|------|------|------|----------|
| **Week 1** | 05.20 - 05.22 | Python 基础语法 | 变量与类型、类型转换、ASCII 编码、循环结构 |
| **Week 2** | 05.25 - 05.29 | 数据结构与 UI | 列表/字典、CRUD 操作、Gradio 图形界面开发 |
| **Week 3** | 06.01 - 06.05 | 文件与数据格式 | 文件 I/O、CSV/JSON/YAML/XML 解析、数据集管理 |
| **Week 4** | 06.08 - 06.12 | 计算机视觉应用 | OpenCV 图像处理、特征匹配、IoU 计算、考试项目 |
| **Week 5** | 06.15 - 06.18 | 深度学习入门 | 神经网络基础、前向传播、损失函数、PyTorch 张量运算、模型保存与加载 |
| **Week 6** | 06.22 | 数据分析与预处理 | Pandas 高级功能：缺失值处理、去重、排序、映射、聚合统计 |

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
│   ├── exam.ipynb              — 期末考试：人脸特征检测
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
├── week6/          # 数据分析与预处理
│   └── 0622.ipynb              — Pandas 数据预处理综合演示
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

### Week 6 — 数据分析与预处理
- **Pandas 高级功能**
- 缺失值处理：`isnull()` / `dropna()` / `fillna()`
- 重复数据处理：`drop_duplicates()`
- 数据变换：`apply()` / `map()` / NumPy 函数
- 排序：`sort_values()` / `sort_index()`
- 数据探索：`unique()` / `value_counts()` / `isin()`

---

## 🚀 运行环境

- Python 3.x
- 主要依赖：`numpy`, `opencv-python`, `pillow`, `matplotlib`, `gradio`, `pandas`, `pyyaml`

```bash
# 激活虚拟环境（conda）
conda activate D:\project\step1\env
```

---

## 📝 说明

- 每日课程涵盖 **`.ipynb`（Jupyter Notebook）**、**`.py`（纯 Python 脚本）** 和 **`.html`（导出页面）** 三种格式
- 每日配套 **PDF 笔记** 以 `日期姓名.pdf` 命名
- 期末考试 `exam.ipynb` 包含人脸特征检测、VOC 标注解析、IoU 计算、图像分割与合并等综合题目
