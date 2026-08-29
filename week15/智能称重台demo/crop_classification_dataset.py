r"""使用训练好的YOLOv8检测模型裁切分类数据集。

默认命令：

    D:\project\step1\env\python.exe crop_classification_dataset.py

脚本不会修改原始图片。裁切结果默认写入：

    classification_dataset_cropped/train

输出目录会保留原来的24个SKU子目录，可直接供ResNet的ImageFolder读取。
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"
DEFAULT_SOURCE = PROJECT_ROOT / "classification_dataset_raw" / "train"
DEFAULT_OUTPUT = PROJECT_ROOT / "classification_dataset_cropped" / "train"
DEFAULT_MODEL = PROJECT_ROOT / "runs" / "detect" / "smart_checkout_yolov8n" / "weights" / "best.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# SKU目录前缀与YOLO四类检测标签的对应关系。
PREFIX_TO_CLASS = {
    "BAG_": (0, "bag"),
    "BOTTLE_": (1, "bottle"),
    "BOX_": (2, "box"),
    "CYLINDER_": (3, "cylinder"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用YOLOv8检测框裁切ResNet分类数据集。")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="原始分类训练集目录。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="裁切结果目录。")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="训练好的YOLO权重。")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO推理尺寸，默认640。")
    parser.add_argument("--conf", type=float, default=0.25, help="最低检测置信度，默认0.25。")
    parser.add_argument("--padding", type=float, default=0.05, help="检测框四周扩展比例，默认5%%。")
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.03,
        help="检测框面积至少占原图的比例，默认0.03，用于过滤背景小框。",
    )
    parser.add_argument(
        "--min-aspect-ratio",
        type=float,
        default=0.18,
        help="检测框最小宽高比，默认0.18，用于过滤商品边缘形成的极窄框。",
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=5.5,
        help="检测框最大宽高比，默认5.5，用于过滤商品边缘形成的极宽框。",
    )
    parser.add_argument("--batch", type=int, default=16, help="批量推理图片数，默认16。")
    parser.add_argument("--device", default="0", help="推理设备，例如0或cpu，默认第0张GPU。")
    parser.add_argument(
        "--allow-any-class",
        action="store_true",
        help="未检出期望大类时，允许采用其他大类中置信度最高的商品框。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已经存在的裁切图片。")
    return parser.parse_args()


def expected_class_for_folder(folder_name: str) -> tuple[int, str]:
    """根据SKU目录名确定该目录应当检测的商品大类。"""
    for prefix, class_info in PREFIX_TO_CLASS.items():
        if folder_name.upper().startswith(prefix):
            return class_info
    raise ValueError(f"无法根据目录名判断YOLO类别：{folder_name}")


def collect_images(source: Path) -> list[tuple[Path, Path, int, str]]:
    """收集图片，并记录相对路径和期望的大类标签。"""
    items: list[tuple[Path, Path, int, str]] = []
    class_dirs = sorted(path for path in source.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"没有找到SKU子目录：{source}")

    for class_dir in class_dirs:
        expected_id, expected_name = expected_class_for_folder(class_dir.name)
        images = sorted(
            path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for image_path in images:
            items.append((image_path, image_path.relative_to(source), expected_id, expected_name))
    return items


def expand_box(
    box: tuple[float, float, float, float], image_width: int, image_height: int, padding: float
) -> tuple[int, int, int, int]:
    """按检测框宽高比例向四周扩展，并限制在图片边界内。"""
    x1, y1, x2, y2 = box
    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)
    pad_x = box_width * padding
    pad_y = box_height * padding

    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(image_width, int(x2 + pad_x + 0.999))
    bottom = min(image_height, int(y2 + pad_y + 0.999))
    return left, top, right, bottom


def load_model(model_path: Path, device: str):
    """加载项目内固定版本的Ultralytics和训练权重。"""
    if not THIRD_PARTY_ROOT.is_dir():
        raise FileNotFoundError(f"项目内Ultralytics源码不存在：{THIRD_PARTY_ROOT}")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO权重不存在：{model_path}")

    sys.path.insert(0, str(THIRD_PARTY_ROOT))
    import torch
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != "8.4.113":
        raise RuntimeError(f"Ultralytics版本不正确：{ultralytics.__version__}，预期8.4.113")
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("指定了GPU推理，但当前PyTorch无法使用CUDA。可添加 --device cpu。")

    print(f"Ultralytics：{ultralytics.__version__}")
    print(f"模型权重：{model_path}")
    if device != "cpu":
        print(f"推理设备：{torch.cuda.get_device_name(0)}")
    else:
        print("推理设备：CPU")
    return YOLO(str(model_path))


def save_crop(
    source_path: Path,
    output_path: Path,
    box: tuple[float, float, float, float],
    padding: float,
) -> tuple[int, int, int, int]:
    """读取原图、扩展检测框并保存裁切结果。"""
    with Image.open(source_path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        crop_box = expand_box(box, image.width, image.height, padding)
        cropped = image.crop(crop_box)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_kwargs: dict[str, object] = {}
        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs = {"quality": 95, "subsampling": 0}
        cropped.save(output_path, **save_kwargs)
        return crop_box


def main() -> None:
    args = parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.model = args.model.resolve()

    if not args.source.is_dir():
        raise FileNotFoundError(f"原始分类数据集不存在：{args.source}")
    if not 0.0 <= args.padding <= 0.5:
        raise ValueError("--padding必须在0到0.5之间。")
    if not 0.0 < args.conf <= 1.0:
        raise ValueError("--conf必须在0到1之间。")
    if not 0.0 <= args.min_area_ratio <= 1.0:
        raise ValueError("--min-area-ratio必须在0到1之间。")
    if not 0.0 < args.min_aspect_ratio <= args.max_aspect_ratio:
        raise ValueError("检测框宽高比范围不正确。")

    # 将Ultralytics配置放到项目内，避免修改用户的全局配置。
    config_dir = PROJECT_ROOT / "work" / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    items = collect_images(args.source)
    print(f"原始图片：{len(items)}张，SKU目录：{len({item[1].parts[0] for item in items})}个")
    print(f"输出目录：{args.output}")
    model = load_model(args.model, args.device)

    dataset_root = args.output.parent
    review_dir = dataset_root / "review_failed_detection"
    manifest_path = dataset_root / "crop_manifest.csv"
    dataset_root.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source",
        "output",
        "sku",
        "expected_class_id",
        "expected_class_name",
        "detected_class_id",
        "detected_class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "status",
    ]
    previous_rows: dict[str, dict[str, object]] = {}
    if manifest_path.is_file() and not args.overwrite:
        with manifest_path.open("r", newline="", encoding="utf-8-sig") as file:
            previous_rows = {str(row["source"]): dict(row) for row in csv.DictReader(file)}

    rows_by_source: dict[str, dict[str, object]] = {}
    status_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()

    # 已经生成的裁切图不重复推理，同时保留上次清单中的检测信息。
    pending_items: list[tuple[Path, Path, int, str]] = []
    for item in items:
        source_path, relative_path, expected_id, expected_name = item
        output_path = args.output / relative_path
        if output_path.exists() and not args.overwrite:
            previous = previous_rows.get(str(source_path))
            if previous is not None:
                rows_by_source[str(source_path)] = previous
            else:
                rows_by_source[str(source_path)] = {
                    "source": str(source_path),
                    "output": str(output_path),
                    "sku": relative_path.parts[0],
                    "expected_class_id": expected_id,
                    "expected_class_name": expected_name,
                    "status": "skipped_existing",
                }
            status_counts["skipped_existing"] += 1
        else:
            pending_items.append(item)

    print(f"需要推理：{len(pending_items)}张，已有裁切：{status_counts['skipped_existing']}张")

    for batch_start in range(0, len(pending_items), args.batch):
        batch_items = pending_items[batch_start : batch_start + args.batch]
        batch_sources = [str(item[0]) for item in batch_items]
        results = model.predict(
            source=batch_sources,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )

        for (source_path, relative_path, expected_id, expected_name), result in zip(batch_items, results, strict=True):
            output_path = args.output / relative_path
            row: dict[str, object] = {
                "source": str(source_path),
                "output": str(output_path),
                "sku": relative_path.parts[0],
                "expected_class_id": expected_id,
                "expected_class_name": expected_name,
                "detected_class_id": "",
                "detected_class_name": "",
                "confidence": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "status": "",
            }

            expected_candidates: list[tuple[float, int, tuple[float, float, float, float]]] = []
            all_candidates: list[tuple[float, int, tuple[float, float, float, float]]] = []
            image_height, image_width = result.orig_shape
            image_area = max(float(image_width * image_height), 1.0)
            if result.boxes is not None:
                for class_id, confidence, xyxy in zip(
                    result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist(), strict=True
                ):
                    x1, y1, x2, y2 = (float(value) for value in xyxy)
                    box_width = max(x2 - x1, 0.0)
                    box_height = max(y2 - y1, 0.0)
                    box_area_ratio = box_width * box_height / image_area
                    if box_area_ratio < args.min_area_ratio:
                        continue
                    box_aspect_ratio = box_width / max(box_height, 1.0)
                    if not args.min_aspect_ratio <= box_aspect_ratio <= args.max_aspect_ratio:
                        continue
                    candidate = (float(confidence), int(class_id), tuple(float(value) for value in xyxy))
                    all_candidates.append(candidate)
                    if int(class_id) == expected_id:
                        expected_candidates.append(candidate)

            candidates = expected_candidates
            if not candidates and args.allow_any_class:
                candidates = all_candidates

            if not candidates:
                review_path = review_dir / relative_path
                review_path.parent.mkdir(parents=True, exist_ok=True)
                if source_path.resolve() != review_path.resolve():
                    shutil.copy2(source_path, review_path)
                row["status"] = "failed_no_expected_class"
                status_counts["failed_no_expected_class"] += 1
                rows_by_source[str(source_path)] = row
                continue

            confidence, detected_id, detected_box = max(candidates, key=lambda item: item[0])
            crop_box = save_crop(source_path, output_path, detected_box, args.padding)
            detected_name = str(result.names.get(detected_id, detected_id))
            status = "cropped_expected_class" if detected_id == expected_id else "cropped_other_class"
            row.update(
                {
                    "detected_class_id": detected_id,
                    "detected_class_name": detected_name,
                    "confidence": f"{confidence:.6f}",
                    "x1": crop_box[0],
                    "y1": crop_box[1],
                    "x2": crop_box[2],
                    "y2": crop_box[3],
                    "status": status,
                }
            )
            status_counts[status] += 1
            class_counts[expected_name] += 1
            rows_by_source[str(source_path)] = row

            # 如果该图上次被列为漏检，成功补裁后移除脚本生成的复核副本。
            stale_review_path = review_dir / relative_path
            if stale_review_path.is_file() and stale_review_path.resolve() != source_path.resolve():
                stale_review_path.unlink()

        processed = min(batch_start + len(batch_items), len(pending_items))
        print(f"推理进度：{processed}/{len(pending_items)}", end="\r", flush=True)

    rows = [rows_by_source[str(item[0])] for item in items]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(" " * 60, end="\r")
    print(f"按期望大类裁切：{status_counts['cropped_expected_class']}张")
    print(f"采用其他大类框补裁：{status_counts['cropped_other_class']}张")
    print(f"已存在并跳过：{status_counts['skipped_existing']}张")
    print(f"未检测到期望类别：{status_counts['failed_no_expected_class']}张")
    if class_counts:
        print("分类统计：" + "，".join(f"{name}={class_counts[name]}" for name in sorted(class_counts)))
    print(f"裁切清单：{manifest_path}")
    if status_counts["failed_no_expected_class"]:
        print(f"待复核原图：{review_dir}")


if __name__ == "__main__":
    main()
