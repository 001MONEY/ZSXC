r"""阈值标定脚本：用已注册与未注册商品共同标定检索阈值。

用法：
    1) 已注册商品：默认用 classification_dataset_from_videos 的 val 图（商品特写）。
    2) 未注册商品：把冰红茶/阿萨姆等照片放到一个目录（任意背景），
       脚本会先用 YOLO 检测裁切，再提取特征检索。

    D:\project\step1\env\python.exe calibrate_threshold.py --unknown-dir D:/unknown_samples

输出：已注册/未注册商品的 Top1相似度 与 Top1-Top2间隔 分布，
并给出推荐的 similarity_threshold 与 margin_threshold。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_demo import (  # noqa: E402
    CLASS_NAMES,
    PROJECT_ROOT,
    expand_box,
    load_feature_library,
    load_yolo,
    retrieval_match,
    YOLO_MODEL,
)


DATA_ROOT = PROJECT_ROOT / "classification_dataset_from_videos"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="标定特征检索阈值。")
    parser.add_argument("--unknown-dir", type=Path, default=None, help="未注册商品图片目录（必填以完整标定）。")
    parser.add_argument("--device", default="auto", help="设备：auto/0/cpu。")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO置信度阈值。")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{normalized}" if normalized.isdigit() else normalized)


def collect_registered_images() -> list[tuple[Path, str]]:
    """收集已注册商品的 val 图，返回 (图片路径, 期望SKU)。"""
    items: list[tuple[Path, str]] = []
    for group in ("bag", "bottle", "box", "cylinder"):
        val_root = DATA_ROOT / group / "val"
        if not val_root.is_dir():
            continue
        for class_dir in sorted(val_root.iterdir()):
            if not class_dir.is_dir():
                continue
            images = sorted(
                path for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            for image_path in images:
                items.append((image_path, class_dir.name))
    return items


def collect_unknown_images(directory: Path) -> list[Path]:
    """收集未注册商品图片。"""
    if not directory.is_dir():
        raise FileNotFoundError(f"未注册图片目录不存在：{directory}")
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def crop_objects(frame_bgr: np.ndarray, yolo, yolo_device: str, args, width: int, height: int):
    """用YOLO检测并裁切商品区域，返回 [(裁切图, 包装类型), ...]。"""
    results = yolo.predict(
        source=frame_bgr, conf=args.conf, imgsz=640, device=yolo_device, verbose=False
    )[0]
    crops: list[tuple[np.ndarray, str]] = []
    if results.boxes is not None:
        for box in results.boxes:
            yolo_class = int(box.cls.item())
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue
            left, top, right, bottom = expand_box(box.xyxy[0], width, height, 0.05)
            crop = frame_bgr[top:bottom, left:right]
            if crop.size > 0:
                crops.append((crop, CLASS_NAMES[yolo_class]))
    return crops


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    # Ultralytics 不接受 auto，需转换为 '0' 或 'cpu'。
    yolo_device = f"{device.index}" if device.type == "cuda" else "cpu"
    print(f"设备：{device}")

    yolo = load_yolo(YOLO_MODEL, yolo_device)
    feature_library = load_feature_library(device)

    records: dict[str, list] = {"registered": [], "registered_wrong": [], "unknown": []}

    # 已注册商品：val特写图直接检索（图片已是商品裁切），并校验是否正确识别。
    registered_images = collect_registered_images()
    print(f"已注册商品 val 图：{len(registered_images)}张")
    for image_path, expected in registered_images:
        with Image.open(image_path) as image:
            crop = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        group = expected.split("_")[0].lower()
        if group not in feature_library:
            continue
        model_class, top1, top2_class, top2_sim, margin = retrieval_match(
            crop, feature_library[group], device
        )
        if model_class == expected:
            records["registered"].append((top1, margin, expected))
        else:
            records["registered_wrong"].append((top1, margin, expected, model_class))
    print(
        f"  正确识别 {len(records['registered'])}，识别错误 {len(records['registered_wrong'])}"
    )

    # 未注册商品：YOLO检测裁切后，按包装类型只进入对应特征库检索。
    if args.unknown_dir is not None:
        unknown_images = collect_unknown_images(args.unknown_dir)
        print(f"未注册商品图片：{len(unknown_images)}张")
        for image_path in unknown_images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            height, width = frame.shape[:2]
            crops = crop_objects(frame, yolo, yolo_device, args, width, height)
            for crop, group in crops:
                if group not in feature_library:
                    continue
                model_class, top1, top2_class, top2_sim, margin = retrieval_match(
                    crop, feature_library[group], device
                )
                records["unknown"].append((top1, margin, f"{group}/{model_class}"))
        print(f"  完成 {len(records['unknown'])} 条")

    # 输出分布。
    def summarize(name: str, rows: list[tuple[float, float, str]]) -> None:
        if not rows:
            print(f"\n[{name}] 无数据")
            return
        sims = [row[0] for row in rows]
        margins = [row[1] for row in rows]
        print(f"\n[{name}] {len(rows)}条")
        print(f"  Top1相似度: min={min(sims):.4f} mean={sum(sims)/len(sims):.4f} max={max(sims):.4f}")
        print(f"  Top1-Top2间隔: min={min(margins):.4f} mean={sum(margins)/len(margins):.4f} max={max(margins):.4f}")

    summarize("已注册", records["registered"])
    summarize("未注册", records["unknown"])

    # 已注册识别错误明细（这些不能作为正样本，需单独排查）。
    if records["registered_wrong"]:
        print(f"\n[已注册识别错误] {len(records['registered_wrong'])}条：")
        for top1, margin, expected, predicted in records["registered_wrong"][:20]:
            print(f"  期望={expected} -> 检索到={predicted} (sim={top1:.3f}, margin={margin:.3f})")

    # 建议阈值：已注册min 与 未注册max 的中点（若可分）。
    if records["registered"] and records["unknown"]:
        reg_sims = [r[0] for r in records["registered"]]
        unk_sims = [r[0] for r in records["unknown"]]
        reg_min = min(reg_sims)
        unk_max = max(unk_sims)
        if reg_min > unk_max:
            suggested = round((reg_min + unk_max) / 2, 3)
            print(f"\n建议 similarity_threshold ≈ {suggested}（已注册min={reg_min:.3f} > 未注册max={unk_max:.3f}）")
        else:
            print(f"\n注意：已注册min={reg_min:.3f} 与 未注册max={unk_max:.3f} 有重叠，"
                  f"需要 margin 间隔辅助或提升特征区分度（ArcFace）。")
        reg_margins = [r[1] for r in records["registered"]]
        unk_margins = [r[1] for r in records["unknown"]]
        print(f"  margin参考：已注册min={min(reg_margins):.4f}，未注册max={max(unk_margins):.4f}")

    # 阈值组合网格评估：找 (sim, margin) 最佳组合。
    if records["registered"] and records["unknown"]:
        print("\n=== 阈值组合评估（已注册保留率 / 未注册误判率） ===")
        reg = records["registered"]
        unk = records["unknown"]
        best = None
        for sim_t in (0.80, 0.82, 0.85, 0.88):
            for margin_t in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20):
                reg_pass = sum(1 for s, m, _ in reg if s >= sim_t and m >= margin_t) / len(reg)
                unk_wrong = sum(1 for s, m, _ in unk if s >= sim_t and m >= margin_t) / len(unk)
                print(f"  sim>={sim_t:.2f} margin>={margin_t:.2f}: 已注册保留={reg_pass:.1%} 未注册误判={unk_wrong:.1%}")
                if unk_wrong == 0.0 and best is None:
                    best = (sim_t, margin_t, reg_pass)
                elif unk_wrong == 0.0 and best is not None and reg_pass > best[2]:
                    best = (sim_t, margin_t, reg_pass)
        if best:
            print(f"\n推荐：sim>={best[0]} margin>={best[1]}（已注册保留 {best[2]:.1%}，未注册0误判）")


if __name__ == "__main__":
    main()
