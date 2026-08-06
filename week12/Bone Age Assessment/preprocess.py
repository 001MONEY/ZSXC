# -*- coding: utf-8 -*-
"""
图像预处理基线：灰度化 + 中值滤波去噪 + CLAHE 自适应直方图均衡化

对应方案第一阶段任务2：预处理基线。

流程：BGR原图 → 灰度化(加权平均) → 中值滤波去噪 → CLAHE → 转回3通道
说明：
  - 灰度化用加权平均法 0.299R + 0.587G + 0.114B（cv2 默认，符合医学图像惯例）
  - 中值滤波适合去 X 光片噪声且保留边缘（核选奇数，默认3）
  - CLAHE 分块均衡化，提升局部对比度、保留骨密度局部特征
  - 输出保持 3 通道，与 YOLO / ImageFolder 训练直接兼容
  - 检测框坐标不变，labels / data.yaml 直接复用

输出（不覆盖原始数据集）：
    datasets/detection_pre/            （检测预处理版：images + labels + data.yaml）
    datasets/classification_pre/{关节}/ （分类预处理版：train/val 同结构）
    output/preview/                     （--preview 生成的前后对比图）

用法：
    python preprocess.py                 # 全量处理检测 + 分类
    python preprocess.py --detection     # 只处理检测
    python preprocess.py --classification
    python preprocess.py --preview       # 只生成几张前后对比图
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import config


# ---------------- 预处理参数（可自行调整） ----------------
GRAY_WEIGHTS = (0.299, 0.587, 0.114)   # 加权平均法灰度化权重
MEDIAN_KSIZE = 3                       # 中值滤波核大小（奇数）
CLAHE_CLIP = 2.0                       # CLAHE 对比度限制（越大对比度越强）
CLAHE_GRID = (8, 8)                    # CLAHE 分块网格

DETECTION_PRE = config.DETECTION_PRE      # 预处理版检测数据集
CLASSIFICATION_PRE = config.CLASSIFICATION_PRE  # 预处理版分类数据集
PREVIEW_DIR = config.BAA_DIR / "output" / "preview"


# ---------------------------------------------------------------- 核心函数
def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """灰度化 → 中值滤波 → CLAHE。输入输出均为 3 通道 BGR。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)        # 加权平均法灰度化
    gray = cv2.medianBlur(gray, MEDIAN_KSIZE)               # 中值滤波去噪
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    eq = clahe.apply(gray)                                  # CLAHE 均衡化
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)             # 转回 3 通道


def load_and_process(path: Path):
    """读取并预处理，失败返回 None。"""
    img = cv2.imread(str(path))
    if img is None:
        print(f"[警告] 读取失败: {path}")
        return None
    return preprocess_image(img)


# ---------------------------------------------------------------- 检测数据
def process_detection():
    src, dst = config.DETECTION_DIR, DETECTION_PRE
    for split in ("train", "val"):
        src_img, dst_img = src / "images" / split, dst / "images" / split
        dst_lab = dst / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lab.mkdir(parents=True, exist_ok=True)

        files = sorted(src_img.iterdir())
        count = 0
        for f in tqdm(files, desc=f"detection/{split}", ncols=80):
            proc = load_and_process(f)
            if proc is None:
                continue
            cv2.imwrite(str(dst_img / f.name), proc)
            count += 1
        # 标签与坐标无关，直接复制
        for lab in (src / "labels" / split).iterdir():
            shutil.copy2(lab, dst_lab / lab.name)
        print(f"  {split}: {count} 张")

    # 重新生成 data.yaml（指向预处理后的路径）
    content = [
        "# 检测数据集配置（预处理版，自动生成）",
        f"path: {str(dst).replace(chr(92), '/')}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(config.DET_CLASSES)}",
        "names:",
    ]
    content += [f"  {i}: {name}" for i, name in enumerate(config.DET_CLASSES)]
    (dst / "data.yaml").write_text("\n".join(content) + "\n", encoding="utf-8")
    print(f"[OK] 检测预处理完成 -> {dst}")


# ---------------------------------------------------------------- 分类数据
def process_classification():
    src, dst = config.CLASSIFICATION_DIR, CLASSIFICATION_PRE
    tasks = []  # (源文件, 目标文件)
    for joint in config.JOINT_TYPES:
        j_src = src / joint
        if not j_src.exists():
            print(f"[警告] 缺少关节目录: {j_src}")
            continue
        for split in ("train", "val"):
            sp = j_src / split
            if not sp.exists():
                continue
            for grade_dir in sp.iterdir():
                if not grade_dir.is_dir():
                    continue
                out_dir = dst / joint / split / grade_dir.name
                for f in grade_dir.iterdir():
                    if f.is_file():
                        tasks.append((f, out_dir / f.name))

    for f, out in tqdm(tasks, desc="classification", ncols=80):
        proc = load_and_process(f)
        if proc is None:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), proc)

    # 复制每个关节的 classes.txt / stats.csv
    for joint in config.JOINT_TYPES:
        for name in ("classes.txt", "stats.csv"):
            p = src / joint / name
            if p.exists():
                shutil.copy2(p, dst / joint / name)
    print(f"[OK] 分类预处理完成: {len(tasks)} 张 -> {dst}")


# ---------------------------------------------------------------- 预览对比
def preview(n: int = 6):
    """生成 n 张 前后对比图（左: 原图，右: CLAHE预处理后）到 output/preview/。"""
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    imgs = sorted((config.DETECTION_DIR / "images" / "train").glob("*"))[:n]
    if not imgs:
        print("[警告] 检测训练集为空，无法生成预览")
        return

    for i, f in enumerate(imgs):
        img = cv2.imread(str(f))
        if img is None:
            continue
        proc = preprocess_image(img)
        # 限制高度方便查看，保持比例
        target_h = 800
        img = cv2.resize(img, (int(img.shape[1] * target_h / img.shape[0]), target_h))
        proc = cv2.resize(proc, (int(proc.shape[1] * target_h / proc.shape[0]), target_h))
        # 加标注
        cv2.putText(img, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(proc, "CLAHE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        combo = np.hstack([img, proc])
        out = PREVIEW_DIR / f"preview_{i:02d}_{f.stem}.png"
        cv2.imwrite(str(out), combo)
        print(f"[OK] {out}")
    print(f"共生成 {len(imgs)} 张对比图 -> {PREVIEW_DIR}")


# ---------------------------------------------------------------- 主流程
def main():
    parser = argparse.ArgumentParser(description="预处理基线：灰度化 + 中值滤波 + CLAHE")
    parser.add_argument("--detection", action="store_true", help="只处理检测数据")
    parser.add_argument("--classification", action="store_true", help="只处理分类数据")
    parser.add_argument("--preview", action="store_true", help="只生成前后对比图")
    parser.add_argument("--n", type=int, default=6, help="预览图片数量（默认6）")
    args = parser.parse_args()

    print("预处理参数: 灰度化(加权平均) + 中值滤波 k=%d + CLAHE clip=%.1f grid=%s"
          % (MEDIAN_KSIZE, CLAHE_CLIP, CLAHE_GRID))

    if args.preview:
        preview(args.n)
        return

    do_det = args.detection or not (args.detection or args.classification)
    do_cls = args.classification or not (args.detection or args.classification)

    if do_det:
        process_detection()
    if do_cls:
        process_classification()
    print("\n[OK] 全部完成！")


if __name__ == "__main__":
    main()
