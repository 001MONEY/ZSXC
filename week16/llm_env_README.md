# 第三阶段 LLM 学习环境说明

本分支（`llm`）用于承载**第三阶段 · 大模型与 NLP 课程**的代码与笔记。

## 运行环境

- 环境位置：`D:\project\step3\llm`（conda 虚拟环境，Python 3.11，**不纳入 Git**）
- 依赖清单：`requirements-llm-env.txt`（由 `pip freeze` 导出，149 个包）
- 核心包：torch 2.10.0+cu128（CUDA 12.8）、transformers 5.16.1、tokenizers、datasets、
  accelerate、sentencepiece、gensim、jieba、nltk、pandas、scikit-learn、matplotlib 等

## 复现环境（可选）

```bash
# 若需在别处重建同等环境
pip install -r requirements-llm-env.txt
```

> torch 的 CUDA 版本如需匹配本机驱动，可单独安装：`pip install torch --index-url https://download.pytorch.org/whl/cu128`

## 分支约定

- `main`：第一、二阶段（week01–week15）内容
- `llm`：第三阶段（大模型 NLP，week16 起）内容
