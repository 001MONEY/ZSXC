r"""训练智能称重台四类商品检测模型。

默认使用项目内的 YOLOv8n 预训练权重和 Ultralytics 8.4.113 源码：

    D:\project\step1\env\python.exe train_yolov8.py --check-only
    D:\project\step1\env\python.exe train_yolov8.py
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from collections import Counter
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"
DEFAULT_DATA = PROJECT_ROOT / "smart_checkout_data.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolov8n.pt"
DEFAULT_RUNS = PROJECT_ROOT / "runs" / "detect"
EXPECTED_CLASSES = {0: "bag", 1: "bottle", 2: "box", 3: "cylinder"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练智能称重台YOLOv8四类目标检测模型。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="数据集YAML配置文件。")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="初始模型权重或模型YAML。")
    parser.add_argument("--epochs", type=int, default=100, help="最大训练轮数，默认100。")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸，默认640。")
    parser.add_argument("--batch", type=int, default=8, help="批大小，6GB显存默认8。")
    parser.add_argument("--device", default="0", help="训练设备，例如0或cpu，默认使用第0张GPU。")
    parser.add_argument("--workers", type=int, default=4, help="数据加载进程数，默认4。")
    parser.add_argument("--patience", type=int, default=30, help="早停等待轮数，默认30。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认42。")
    parser.add_argument("--project", type=Path, default=DEFAULT_RUNS, help="训练结果根目录。")
    parser.add_argument("--name", default="smart_checkout_yolov8n", help="本次训练名称。")
    parser.add_argument("--cache", action="store_true", help="将图片缓存到内存或磁盘以加速训练。")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="LAST_PT",
        help="从指定的last.pt继续训练，例如 --resume runs/detect/实验名/weights/last.pt。",
    )
    parser.add_argument("--exist-ok", action="store_true", help="允许复用同名结果目录。")
    parser.add_argument("--check-only", action="store_true", help="只检查数据、环境和权重，不开始训练。")
    return parser.parse_args()


def load_dataset_config(data_file: Path) -> tuple[Path, dict[int, str]]:
    data_file = data_file.resolve()
    if not data_file.is_file():
        raise FileNotFoundError(f"数据集配置不存在：{data_file}")

    with data_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset_root = Path(config["path"])
    if not dataset_root.is_absolute():
        dataset_root = (data_file.parent / dataset_root).resolve()
    else:
        dataset_root = dataset_root.resolve()

    raw_names = config.get("names", {})
    if isinstance(raw_names, list):
        class_names = dict(enumerate(raw_names))
    else:
        class_names = {int(class_id): str(name) for class_id, name in raw_names.items()}

    if class_names != EXPECTED_CLASSES:
        raise ValueError(f"类别映射不正确：{class_names}，预期：{EXPECTED_CLASSES}")
    return dataset_root, class_names


def validate_split(dataset_root: Path, split: str, class_names: dict[int, str]) -> dict[str, object]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"{split}图片目录不存在：{image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"{split}标签目录不存在：{label_dir}")

    images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    labels = {path.stem: path for path in label_dir.glob(f"yolo_{split}_*.txt") if path.is_file()}
    image_stems = {path.stem for path in images}

    missing_labels = sorted(image_stems - labels.keys())
    extra_labels = sorted(labels.keys() - image_stems)
    if missing_labels:
        raise ValueError(f"{split}缺少{len(missing_labels)}个标签：{missing_labels[:10]}")
    if extra_labels:
        raise ValueError(f"{split}存在{len(extra_labels)}个无对应图片的标签：{extra_labels[:10]}")

    class_counts: Counter[int] = Counter()
    empty_labels = 0
    box_count = 0
    errors: list[str] = []

    for image in images:
        label = labels[image.stem]
        lines = label.read_text(encoding="utf-8-sig").splitlines()
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        if not non_empty_lines:
            empty_labels += 1
            continue

        for line_number, line in enumerate(non_empty_lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{label.name}:{line_number} 应为5列，实际{len(parts)}列")
                continue
            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
            except ValueError:
                errors.append(f"{label.name}:{line_number} 存在无法解析的数字")
                continue

            if class_id not in class_names:
                errors.append(f"{label.name}:{line_number} 类别ID越界：{class_id}")
            coordinates = (x_center, y_center, width, height)
            if not all(0.0 <= value <= 1.0 for value in coordinates):
                errors.append(f"{label.name}:{line_number} 坐标不在0到1之间：{coordinates}")
            if width <= 0.0 or height <= 0.0:
                errors.append(f"{label.name}:{line_number} 框宽高必须大于0：{width}, {height}")

            class_counts[class_id] += 1
            box_count += 1

    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"{split}标签校验失败，共{len(errors)}处：\n{preview}")

    return {
        "images": len(images),
        "labels": len(labels),
        "empty_labels": empty_labels,
        "boxes": box_count,
        "class_counts": class_counts,
    }


def validate_dataset(data_file: Path) -> None:
    dataset_root, class_names = load_dataset_config(data_file)
    print(f"数据集目录：{dataset_root}")
    for split in ("train", "val"):
        summary = validate_split(dataset_root, split, class_names)
        counts = summary["class_counts"]
        class_text = ", ".join(f"{class_names[class_id]}={counts[class_id]}" for class_id in class_names)
        print(
            f"[{split}] 图片={summary['images']}，标签={summary['labels']}，"
            f"空样本={summary['empty_labels']}，目标框={summary['boxes']}（{class_text}）"
        )


def load_training_runtime(model_file: Path, device: str):
    if not THIRD_PARTY_ROOT.is_dir():
        raise FileNotFoundError(f"项目内Ultralytics源码不存在：{THIRD_PARTY_ROOT}")
    sys.path.insert(0, str(THIRD_PARTY_ROOT))

    import torch
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != "8.4.113":
        raise RuntimeError(f"Ultralytics版本不正确：{ultralytics.__version__}，预期8.4.113")
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("指定了GPU训练，但当前PyTorch无法使用CUDA。可添加 --device cpu 改用CPU。")
    if not model_file.is_file():
        raise FileNotFoundError(f"模型权重不存在：{model_file}")

    print(f"Python：{sys.version.split()[0]}")
    print(f"PyTorch：{torch.__version__}")
    print(f"Ultralytics：{ultralytics.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA：{torch.version.cuda}，GPU：{torch.cuda.get_device_name(0)}")
    else:
        print("训练设备：CPU")
    print(f"初始权重：{model_file}")
    return YOLO(str(model_file))


def main() -> None:
    args = parse_args()
    args.data = args.data.resolve()
    args.model = args.model.resolve()
    args.project = args.project.resolve()
    if args.resume is not None:
        args.resume = args.resume.resolve()

    # 将Ultralytics运行配置放在项目目录内，避免污染用户全局配置。
    config_dir = PROJECT_ROOT / "work" / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    validate_dataset(args.data)
    initial_model = args.resume if args.resume is not None else args.model
    model = load_training_runtime(initial_model, args.device)
    if args.check_only:
        print("数据集、运行环境和YOLOv8n权重检查通过，未开始训练。")
        return

    print(f"训练输出：{args.project / args.name}")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        project=str(args.project),
        name=args.name,
        cache=args.cache,
        resume=str(args.resume) if args.resume is not None else False,
        exist_ok=args.exist_ok,
        pretrained=True,
        val=True,
        plots=True,
        close_mosaic=10,
    )

    run_dir = Path(model.trainer.save_dir)
    print("训练完成。")
    print(f"最佳权重：{run_dir / 'weights' / 'best.pt'}")
    print(f"最后权重：{run_dir / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
