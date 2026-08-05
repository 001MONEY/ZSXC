# -*- coding: utf-8 -*-
"""
检测数据准备：VOC 格式 → YOLO 格式

输入：
    handbone/Annotations/*.xml       （VOC 标注）
    handbone/JPEGImages/*.png        （原始 X 光片）

输出：
    datasets/detection/
        ├── images/train|val/*.png        （按划分复制的图片）
        ├── labels/train|val/*.txt        （YOLO 归一化标注: class xc yc w h）
        ├── data.yaml                     （YOLOv8 训练用配置文件）
        └── stats.json                    （各类别统计）

用法：
    python voc_to_yolo.py             # 默认全量转换 + 划分
    python voc_to_yolo.py --dry-run   # 只统计，不落盘
"""
import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import config


# ---------------------------------------------------------------- 解析 VOC
def parse_voc_xml(xml_path: Path):
    """解析单个 VOC XML，返回 (filename, (w, h), objects)。"""
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename")
    size = root.find("size")
    width, height = int(size.findtext("width")), int(size.findtext("height"))
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bnd = obj.find("bndbox")
        xmin, ymin = float(bnd.findtext("xmin")), float(bnd.findtext("ymin"))
        xmax, ymax = float(bnd.findtext("xmax")), float(bnd.findtext("ymax"))
        objects.append({"name": name, "box": (xmin, ymin, xmax, ymax)})
    return filename, (width, height), objects


def to_yolo_line(name: str, box, size):
    """VOC 框 → YOLO 归一化标注行（越界自动截断，宽高非法则跳过）。"""
    w_img, h_img = size
    xmin, ymin, xmax, ymax = box
    xmin, xmax = max(xmin, 0.0), min(xmax, float(w_img))
    ymin, ymax = max(ymin, 0.0), min(ymax, float(h_img))
    if name not in config.DET_CLASS2ID:
        return None
    cls_id = config.DET_CLASS2ID[name]
    xc = (xmin + xmax) / 2.0 / w_img
    yc = (ymin + ymax) / 2.0 / h_img
    w = (xmax - xmin) / w_img
    h = (ymax - ymin) / h_img
    if w <= 0 or h <= 0:
        return None
    return f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


# ---------------------------------------------------------------- 收集样本
def collect_samples():
    """收集所有 xml + 对应图片，返回样本列表。"""
    samples = []
    xml_files = sorted(config.HANDBONE.glob("Annotations/*.xml"))
    for xml_path in xml_files:
        filename, size, objects = parse_voc_xml(xml_path)
        img_path = config.HANDBONE / "JPEGImages" / filename
        if not img_path.exists():
            print(f"[警告] 图片不存在，跳过: {img_path}")
            continue
        samples.append({
            "id": xml_path.stem,
            "img": img_path,
            "size": size,
            "objects": objects,
        })
    return samples


# ---------------------------------------------------------------- 划分
def split_train_val(samples, ratio, seed):
    """
    随机打乱后划分 train/val，并保证 val 覆盖全部 7 类（不漏类）。
    返回 (train, val)。
    """
    random.seed(seed)
    shuffled = samples[:]
    random.shuffle(shuffled)

    # 1) 优先保证每类至少一张进 val
    val_list, seen_classes, remaining = [], set(), []
    for s in shuffled:
        cls_set = {o["name"] for o in s["objects"]}
        new_cls = cls_set - seen_classes
        if new_cls:
            val_list.append(s)
            seen_classes |= cls_set
        else:
            remaining.append(s)

    # 2) 从剩余样本补足 val 数量
    n_val = max(len(val_list), round(len(shuffled) * (1 - ratio)))
    for s in remaining:
        if len(val_list) >= n_val:
            break
        val_list.append(s)

    val_ids = {s["id"] for s in val_list}
    train = [s for s in shuffled if s["id"] not in val_ids]
    val = [s for s in shuffled if s["id"] in val_ids]
    return train, val


# ---------------------------------------------------------------- 落盘
def write_split(samples, split_name, dst_dir):
    """把一批样本写成 YOLO 的 images + labels 目录，返回目标框数。"""
    img_dir = dst_dir / "images" / split_name
    lab_dir = dst_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    n_boxes, n_written = 0, 0
    for s in samples:
        lines = []
        for o in s["objects"]:
            line = to_yolo_line(o["name"], o["box"], s["size"])
            if line:
                lines.append(line)
        if not lines:
            continue  # 无有效目标则跳过（不生成空标注）
        shutil.copy2(s["img"], img_dir / s["img"].name)
        (lab_dir / f"{s['id']}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_boxes += len(lines)
        n_written += 1
    return n_written, n_boxes


def write_data_yaml(dst_dir):
    """生成 YOLOv8 训练用的 data.yaml。"""
    content = [
        "# 检测数据集配置（自动生成）",
        f"path: {str(dst_dir).replace(chr(92), '/')}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(config.DET_CLASSES)}",
        "names:",
    ]
    for i, name in enumerate(config.DET_CLASSES):
        content.append(f"  {i}: {name}")
    (dst_dir / "data.yaml").write_text("\n".join(content) + "\n", encoding="utf-8")
    print(f"[✓] 已生成 data.yaml -> {dst_dir / 'data.yaml'}")


def count_classes(samples):
    c = Counter()
    for s in samples:
        for o in s["objects"]:
            if o["name"] in config.DET_CLASS2ID:
                c[o["name"]] += 1
    return c


def print_stats(samples, tag):
    c = count_classes(samples)
    print(f"\n[{tag}] 图片数: {len(samples)}   目标框总数: {sum(c.values())}")
    for name in config.DET_CLASSES:
        print(f"    {name:<16} {c[name]}")


# ---------------------------------------------------------------- 主流程
def main():
    parser = argparse.ArgumentParser(description="VOC → YOLO 检测数据准备")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = parser.parse_args()

    print("=" * 60)
    print("步骤1/4  收集并解析 VOC 标注...")
    samples = collect_samples()
    print(f"共找到 {len(samples)} 张有效样本")
    print_stats(samples, "全量")

    print("\n步骤2/4  划分 train/val ...")
    train, val = split_train_val(samples, config.TRAIN_RATIO, config.SEED)
    print_stats(train, "train")
    print_stats(val, "val")

    if args.dry_run:
        print("\n[dry-run] 未写任何文件。")
        return

    dst = config.DETECTION_DIR
    print(f"\n步骤3/4  写入 YOLO 数据集 -> {dst}")
    n_tr, box_tr = write_split(train, "train", dst)
    n_va, box_va = write_split(val, "val", dst)
    print(f"train: {n_tr} 张 / {box_tr} 框")
    print(f"val  : {n_va} 张 / {box_va} 框")

    print("\n步骤4/4  生成 data.yaml ...")
    write_data_yaml(dst)

    # 统计存档
    stats = {
        "num_samples": len(samples),
        "split": {
            "train_images": n_tr, "train_boxes": box_tr,
            "val_images": n_va, "val_boxes": box_va,
        },
        "classes": dict(count_classes(samples)),
        "seed": config.SEED,
        "train_ratio": config.TRAIN_RATIO,
    }
    (dst / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[✓] 完成！统计已保存 -> {dst / 'stats.json'}")


if __name__ == "__main__":
    main()
