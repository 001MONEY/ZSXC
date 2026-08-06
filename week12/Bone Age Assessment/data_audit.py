# -*- coding: utf-8 -*-
"""
数据质量审计：检查关节分类数据（arthrosis/）的完整性与标注质量

功能：
1. 基础审计：各等级数量、图片亮度统计、完全重复图片（跨等级=标注冲突）
2. 距离检验：类内 vs 类间相似度（判断等级标注是否有区分度）
3. 蒙太奇图：按等级排列样本，人工直观检查发育进展

用法：
    python data_audit.py                    # 全部关节基础审计 + 距离检验
    python data_audit.py --joints Ulna Radius   # 只查指定关节
    python data_audit.py --montage          # 额外生成蒙太奇图到 output/data_check/
"""
import argparse
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config

OUT_DIR = config.BAA_DIR / "output" / "data_check"


# ---------------------------------------------------------------- 1. 基础审计
def audit_basic(joint):
    base = config.ARTHROSIS / joint
    grades = sorted([int(p.name) for p in base.iterdir() if p.is_dir()])
    cnt, bright, dup = {}, {}, defaultdict(list)
    total = 0
    for g in grades:
        imgs = [p for p in base.glob(f"{g}/*") if p.is_file()]
        cnt[g] = len(imgs)
        total += len(imgs)
        stats = []
        for p in imgs:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            stats.append(img.mean())
            dup[hashlib.md5(img.tobytes()).hexdigest()[:12]].append(f"g{g}/{p.name}")
        bright[g] = (round(float(np.mean(stats)), 1) if stats else 0,
                     round(float(np.std(stats)), 1) if stats else 0)
    print(f"--- {joint}  合计{total}张")
    for g in grades:
        print(f"  等级{g:>2}: {cnt[g]:>3}张  亮度{bright[g]}")
    dups = {h: v for h, v in dup.items() if len(v) > 1}
    cross = {h: v for h, v in dups.items() if len({x.split('/')[0] for x in v}) > 1}
    print(f"  重复组:{len(dups)}  其中跨等级(标注冲突):{len(cross)}")
    for h, v in list(cross.items())[:10]:
        print("     [冲突]", v)
    print()
    return len(cross)


# ---------------------------------------------------------------- 2. 距离检验
def _feats(joint, max_per_grade=60):
    base = config.ARTHROSIS / joint
    grades = sorted([int(p.name) for p in base.iterdir() if p.is_dir()])
    X, Y = [], []
    for g in grades:
        for p in sorted(base.glob(f"{g}/*.png"))[:max_per_grade]:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (32, 32)).astype(np.float32)
            img = (img - img.mean()) / (img.std() + 1e-6)
            X.append(img.ravel())
            Y.append(g)
    return np.array(X), np.array(Y)


def audit_distance(joint):
    X, Y = _feats(joint)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
    sim = X @ X.T
    n = len(Y)
    within, adj, between = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            s = sim[i, j]
            if Y[i] == Y[j]:
                within.append(s)
            elif abs(Y[i] - Y[j]) == 1:
                adj.append(s)
            else:
                between.append(s)
    within, adj = np.array(within), np.array(adj)
    between = np.array(between)
    np.fill_diagonal(sim, -1)
    nn_same = np.mean([Y[j] == Y[i] for i in range(n) for j in [sim[i].argmax()]])
    nn_adj = np.mean([abs(Y[j] - Y[i]) <= 1 for i in range(n) for j in [sim[i].argmax()]])
    sep = within.mean() - between.mean()
    print(f"--- {joint}  样本{len(Y)}  等级{len(set(Y))}")
    print(f"  类内相似度   : {within.mean():.4f}   相邻等级相似度: {adj.mean():.4f}")
    print(f"  类内-类间间隔: {sep:.4f}   最近邻同等级率: {nn_same:.3f}   ±1级率: {nn_adj:.3f}")
    print()


# ---------------------------------------------------------------- 3. 蒙太奇
def make_montage(joint, per_grade=3):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = config.ARTHROSIS / joint
    grades = sorted([int(p.name) for p in base.iterdir() if p.is_dir()])
    S = 96
    n = len(grades)
    canvas = np.full((per_grade * S + (n + 1) * 20, n * S + 60, 3), 255, np.uint8)
    for gi, g in enumerate(grades):
        imgs = sorted(base.glob(f"{g}/*.png"))
        picks = random.sample(imgs, min(per_grade, len(imgs)))
        for ri, p in enumerate(picks):
            img = cv2.imread(str(p))
            img = cv2.resize(img, (S, S))
            canvas[20 + ri * S:20 + ri * S + S, 40 + gi * S:40 + gi * S + S] = img
        cv2.putText(canvas, str(g), (40 + gi * S + S // 2 - 10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(canvas, joint, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    out = OUT_DIR / f"montage_{joint}.png"
    cv2.imwrite(str(out), canvas)
    print(f"[OK] {out}")


def main():
    parser = argparse.ArgumentParser(description="关节数据质量审计")
    parser.add_argument("--joints", nargs="+", default=None, help="只审计指定关节")
    parser.add_argument("--montage", action="store_true", help="生成蒙太奇图")
    args = parser.parse_args()

    joints = args.joints or config.JOINT_TYPES
    for j in joints:
        if j not in config.JOINT_TYPES:
            print(f"[警告] 未知关节: {j}，跳过")
            continue
        audit_basic(j)
        audit_distance(j)
        if args.montage:
            make_montage(j)
    print("[OK] 审计完成！")


if __name__ == "__main__":
    main()
