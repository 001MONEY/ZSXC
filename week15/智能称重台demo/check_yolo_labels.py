# -*- coding: utf-8 -*-
"""
校验 YOLO 标注文件（labels/*.txt）是否存在问题。

检查项：
1. 文件对应：labels 与 images 是否一一对应
2. 行格式：每行必须为 class x_center y_center width height 共 5 个数值
3. 数值范围：x/y/w/h 应在 [0,1] 内（归一化），w/h 必须 > 0
4. 类别编号：必须在 [0, num_classes) 内
5. 重复框：完全相同（或高度相似）的检测框
6. 空标注文件
7. 坐标贴边（等于 0 或 1）提示
"""
import os
import sys
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "yolo_dataset_raw")
SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_classes(split):
    cls_file = os.path.join(DATA, "labels", split, "classes.txt")
    if os.path.exists(cls_file):
        with open(cls_file, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return None


def check_split(split):
    print("=" * 70)
    print(f"[{split}]")
    img_dir = os.path.join(DATA, "images", split)
    lbl_dir = os.path.join(DATA, "labels", split)

    imgs = {}
    if os.path.isdir(img_dir):
        for name in os.listdir(img_dir):
            stem, ext = os.path.splitext(name)
            if ext.lower() in IMG_EXTS:
                imgs[stem] = name
    lb_files = {}
    if os.path.isdir(lbl_dir):
        for name in os.listdir(lbl_dir):
            if name.lower().endswith(".txt") and name != "classes.txt":
                lb_files[os.path.splitext(name)[0]] = name

    num_classes = len(load_classes(split)) if load_classes(split) else None
    if num_classes:
        print(f"类别数: {num_classes} ({load_classes(split)})")

    # 1. 文件对应关系
    img_only = sorted(set(imgs) - set(lb_files))
    lbl_only = sorted(set(lb_files) - set(imgs))
    print(f"图片数: {len(imgs)}  标注数: {len(lb_files)}")
    if img_only:
        print(f"  [缺失标注] 有图片无标注 ({len(img_only)}): {img_only[:10]}")
    if lbl_only:
        print(f"  [多余标注] 有标注无图片 ({len(lbl_only)}): {lbl_only[:10]}")

    # 2-7. 逐行校验
    stats = Counter()
    issue_examples = []  # (level, msg)

    for stem in sorted(lb_files):
        lbl_path = os.path.join(lbl_dir, lb_files[stem])
        lines = []
        try:
            with open(lbl_path, encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except Exception as e:
            issue_examples.append(("ERROR", f"{lb_files[stem]}: 读取失败 {e}"))
            continue

        if not lines:
            stats["empty"] += 1
            issue_examples.append(("WARN", f"{lb_files[stem]}: 空标注文件（无任何框）"))
            continue

        boxes = []
        for i, ln in enumerate(lines, 1):
            parts = ln.split()
            if len(parts) != 5:
                stats["bad_format"] += 1
                issue_examples.append(
                    ("ERROR", f"{lb_files[stem]} 第{i}行: 应有5个值,实际{len(parts)}个 -> {ln}"))
                continue
            try:
                vals = [float(v) for v in parts]
            except ValueError:
                stats["bad_format"] += 1
                issue_examples.append(
                    ("ERROR", f"{lb_files[stem]} 第{i}行: 含非数值 -> {ln}"))
                continue
            c, x, y, w, h = vals
            if c != int(c) or int(c) < 0 or (num_classes and int(c) >= num_classes):
                stats["bad_class"] += 1
                issue_examples.append(
                    ("ERROR", f"{lb_files[stem]} 第{i}行: 类别越界 c={c} -> {ln}"))
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                stats["coord_out"] += 1
                issue_examples.append(
                    ("ERROR", f"{lb_files[stem]} 第{i}行: 中心坐标越界 ({x},{y}) -> {ln}"))
            if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                stats["bad_size"] += 1
                issue_examples.append(
                    ("ERROR", f"{lb_files[stem]} 第{i}行: 宽高非法 w={w} h={h} -> {ln}"))
            # 贴边提示
            for nm, v in (("x", x), ("y", y), ("w", w), ("h", h)):
                if v <= 0.0 or v >= 1.0:
                    stats["edge"] += 1
                    issue_examples.append(
                        ("HINT", f"{lb_files[stem]} 第{i}行: {nm}={v} 贴边/越界，请人工确认 -> {ln}"))
                    break
            boxes.append((c, round(x, 4), round(y, 4), round(w, 4), round(h, 4)))

        # 重复框
        dup = Counter(boxes)
        for box, cnt in dup.items():
            if cnt > 1:
                stats["dup"] += 1
                issue_examples.append(
                    ("WARN", f"{lb_files[stem]}: 重复框 x{cnt} {box}"))

    # 汇总
    summary = {
        "bad_format": "格式错误(非5个数值)",
        "bad_class": "类别越界",
        "coord_out": "中心坐标越界",
        "bad_size": "宽高非法(<=0或>1)",
        "empty": "空标注文件",
        "dup": "重复框文件",
        "edge": "贴边框(需人工确认)",
    }
    has_issue = False
    for key, desc in summary.items():
        if stats[key]:
            has_issue = True
            print(f"  [发现] {desc}: {stats[key]} 处")
    if not has_issue:
        print("  ✓ 格式、类别、坐标、尺寸均正常")
    if stats["edge"]:
        print(f"  (提示) 贴边/边界框 {stats['edge']} 处，可能正常（商品贴边）但建议抽查")

    return issue_examples, stats


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None
    all_issues = []
    for split in SPLITS:
        if target and split != target:
            continue
        issues, _ = check_split(split)
        all_issues.extend((split, lv, msg) for lv, msg in issues)

    print("=" * 70)
    print("问题明细（ERROR=必须处理 / WARN=建议处理 / HINT=人工确认）")
    if not all_issues:
        print("无任何问题")
        return
    for split, lv, msg in all_issues:
        print(f"  [{split}] [{lv}] {msg}")


if __name__ == "__main__":
    main()
