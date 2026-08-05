# -*- coding: utf-8 -*-
"""
分类数据准备：arthrosis/{关节}/{等级}/*.png → ImageFolder 结构

输入：
    arthrosis/DIP/1/xxx.png, arthrosis/DIP/2/xxx.png, ...   （9 个关节类型）

输出（每个关节类型独立一个数据集，供 9 个分类模型分别训练）：
    datasets/classification/{关节}/
        ├── train/{等级}/*.png
        ├── val/{等级}/*.png
        ├── classes.txt       （等级顺序，即类别标签）
        └── stats.csv         （每等级样本数统计）

用法：
    python prepare_classification.py                 # 默认全量处理
    python prepare_classification.py --dry-run       # 只统计，不落盘
    python prepare_classification.py --joints DIP Radius   # 只处理指定关节
"""
import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import config


def collect_images(joint: str):
    """收集某关节类型的所有图片，返回 [(grade:int, img_path), ...]"""
    joint_dir = config.ARTHROSIS / joint
    items = []
    for grade_dir in joint_dir.iterdir():
        if not grade_dir.is_dir():
            continue
        try:
            grade = int(grade_dir.name)
        except ValueError:
            print(f"[警告] 忽略非数字等级目录: {grade_dir}")
            continue
        for img in grade_dir.iterdir():
            if img.is_file() and img.suffix.lower() in config.IMG_EXTS:
                items.append((grade, img))
    return items


def split_items(items, ratio, seed):
    """
    按等级分层划分 train/val：每级至少 1 张进 val、1 张进 train。
    返回 (train_items, val_items)，元素为 (grade, img_path)。
    """
    random.seed(seed)
    by_grade = defaultdict(list)
    for grade, img in items:
        by_grade[grade].append(img)

    train, val = [], []
    for grade in sorted(by_grade):
        imgs = by_grade[grade]
        random.shuffle(imgs)
        if len(imgs) == 1:
            print(f"[警告] {grade} 级只有 1 张图，全部放入 train")
            train.extend((grade, p) for p in imgs)
            continue
        n_val = max(1, min(len(imgs) - 1, round(len(imgs) * (1 - ratio))))
        val.extend((grade, p) for p in imgs[:n_val])
        train.extend((grade, p) for p in imgs[n_val:])
    return train, val


def write_items(items, split_name, dst_joint: Path):
    """把一批 (grade, img) 复制到 dst_joint/{split_name}/{grade}/ 下。"""
    for grade, img in items:
        dst_dir = dst_joint / split_name / str(grade)
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst_dir / img.name)


def process_joint(joint: str, dry_run: bool):
    print("\n" + "=" * 60)
    print(f"处理关节类型: {joint}")

    items = collect_images(joint)
    if not items:
        print("[警告] 该关节无有效图片，跳过")
        return None

    grades = sorted({g for g, _ in items})
    print(f"  等级数: {len(grades)} ({grades[0]} ~ {grades[-1]})   图片总数: {len(items)}")

    train, val = split_items(items, config.TRAIN_RATIO, config.SEED)
    by_grade_tr = defaultdict(int)
    by_grade_va = defaultdict(int)
    for g, _ in train:
        by_grade_tr[g] += 1
    for g, _ in val:
        by_grade_va[g] += 1

    if dry_run:
        return {"joint": joint, "grades": grades, "train": dict(by_grade_tr), "val": dict(by_grade_va)}

    dst_joint = config.CLASSIFICATION_DIR / joint
    if dst_joint.exists():
        shutil.rmtree(dst_joint)  # 重新生成，避免残留旧文件
    write_items(train, "train", dst_joint)
    write_items(val, "val", dst_joint)

    # 写等级类别文件
    (dst_joint / "classes.txt").write_text("\n".join(str(g) for g in grades) + "\n", encoding="utf-8")
    # 写统计 csv
    with open(dst_joint / "stats.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["grade", "train", "val", "total"])
        for g in grades:
            writer.writerow([g, by_grade_tr[g], by_grade_va[g], by_grade_tr[g] + by_grade_va[g]])

    print(f"  train: {len(train)} 张 / val: {len(val)} 张  ->  {dst_joint}")
    return {"joint": joint, "grades": grades, "train": dict(by_grade_tr), "val": dict(by_grade_va)}


def main():
    parser = argparse.ArgumentParser(description="分类数据准备（9 个关节类型）")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    parser.add_argument("--joints", nargs="+", default=None, help="只处理指定关节，如 DIP Radius")
    args = parser.parse_args()

    joints = args.joints or config.JOINT_TYPES
    for j in joints:
        if j not in config.JOINT_TYPES:
            print(f"[警告] 未知关节类型: {j}，跳过")

    summary = [r for r in (process_joint(j, args.dry_run) for j in joints) if r]
    if summary:
        print("\n" + "=" * 60)
        print("汇总：")
        for r in summary:
            print(f"  {r['joint']:<10} 等级 {len(r['grades']):>2} 个  train {sum(r['train'].values()):>4}  val {sum(r['val'].values()):>4}")
    print("\n[✓] 完成！")


if __name__ == "__main__":
    main()
