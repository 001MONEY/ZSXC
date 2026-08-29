r"""构建商品特征向量库（余弦相似度检索用）。

对四个包装大类的训练图片提取 ResNet18 分类头之前的 512 维特征，
入库前统一做 L2 归一化。每种包装单独建库，只与自己的 6 个 SKU 比较。

每个 SKU 保存多张图的特征样本，并额外计算类中心（均值后 L2 归一化）。

默认命令：

    D:\project\step1\env\python.exe build_feature_library.py
    D:\project\step1\env\python.exe build_feature_library.py --update-db   # 顺带把类中心标识写入 products.feature_index

输出（runs/features/）：
    <group>_embeddings.npy   所有样本特征 (N, 512)，L2归一化
    <group>_labels.json      每个样本对应的SKU名
    <group>_centers.npy      每个SKU类中心 (6, 512)，L2归一化
    <group>_classes.json     类中心对应的SKU名（与centers行一一对应）
    <group>_metadata.json    模型/预处理/版本等元数据
    <group>_stats.json       同类/异类相似度分布统计（用于标定阈值）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_resnet_classifier import EnsureRGB, SquarePad, build_model  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent
CLASSIFY_DIR = PROJECT_ROOT / "runs" / "classify"
DATA_ROOT = PROJECT_ROOT / "classification_dataset_from_videos"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "features"
GROUPS = ("bag", "bottle", "box", "cylinder")
LIBRARY_VERSION = 1
FEATURE_DIM = 512  # ResNet18 avgpool 输出维度
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建商品特征向量库。")
    parser.add_argument("--data", type=Path, default=DATA_ROOT, help="分类数据集根目录。")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="特征库输出目录。")
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="用哪个划分构建特征库，默认train（样本多）。",
    )
    parser.add_argument("--device", default="auto", help="设备：auto/0/cpu。")
    parser.add_argument(
        "--update-db",
        action="store_true",
        help="构建后把每个SKU的类中心标识写入 products.feature_index。",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{normalized}" if normalized.isdigit() else normalized)


def load_feature_model(group: str, device: torch.device):
    """加载ResNet18并去掉分类头（fc=Identity），输出512维特征。"""
    checkpoint_path = CLASSIFY_DIR / f"{group}_resnet18" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, _ = build_model(checkpoint["architecture"], len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.fc = torch.nn.Identity()
    model.to(device).eval()
    return model, checkpoint


def build_transform(img_size: int, mean, std) -> transforms.Compose:
    return transforms.Compose(
        [
            EnsureRGB(),
            SquarePad(),
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


@torch.inference_mode()
def extract_embedding(model, pil_image: Image.Image, transform, device: torch.device) -> np.ndarray:
    tensor = transform(pil_image).unsqueeze(0).to(device, non_blocking=True)
    feature = model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
    return l2_normalize(feature)  # 入库前 L2 归一化


def similarity_stats(matrix: np.ndarray, labels: list[str]) -> dict:
    """统计同类/异类相似度分布，用于阈值标定参考。"""
    similarities = matrix @ matrix.T
    same: list[float] = []
    diff: list[float] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            value = float(similarities[i, j])
            if labels[i] == labels[j]:
                same.append(value)
            else:
                diff.append(value)
    return {
        "same_count": len(same),
        "same_min": round(min(same), 4) if same else None,
        "same_max": round(max(same), 4) if same else None,
        "same_mean": round(float(np.mean(same)), 4) if same else None,
        "diff_count": len(diff),
        "diff_min": round(min(diff), 4) if diff else None,
        "diff_max": round(max(diff), 4) if diff else None,
        "diff_mean": round(float(np.mean(diff)), 4) if diff else None,
        # 同类最低与异类最高的重叠情况，辅助判断可分离性。
        "overlap": bool(diff and same and max(diff) > min(same)),
    }


def build_library(group: str, args: argparse.Namespace, device: torch.device) -> dict:
    model, checkpoint = load_feature_model(group, device)
    img_size = checkpoint["img_size"]
    mean = checkpoint["imagenet_mean"]
    std = checkpoint["imagenet_std"]
    transform = build_transform(img_size, mean, std)

    class_root = args.data / group / args.split
    images = sorted(
        path for path in class_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"没有{args.split}图片：{class_root}")

    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    for image_path in images:
        with Image.open(image_path) as image:
            embeddings.append(extract_embedding(model, image, transform, device))
        labels.append(image_path.parent.name)

    matrix = np.stack(embeddings)  # (N, 512)
    if matrix.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"特征维度异常：{matrix.shape[1]}，预期{FEATURE_DIM}")

    # 类中心：每个SKU样本均值后再次L2归一化。
    classes = sorted(set(labels))
    class_to_idx = {name: index for index, name in enumerate(classes)}
    centers = np.zeros((len(classes), FEATURE_DIM), dtype=np.float32)
    for name in classes:
        indices = [i for i, label in enumerate(labels) if label == name]
        centers[class_to_idx[name]] = l2_normalize(matrix[indices].mean(axis=0))

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / f"{group}_embeddings.npy", matrix)
    (args.output / f"{group}_labels.json").write_text(
        json.dumps(labels, ensure_ascii=False), encoding="utf-8"
    )
    np.save(args.output / f"{group}_centers.npy", centers)
    (args.output / f"{group}_classes.json").write_text(
        json.dumps(classes, ensure_ascii=False), encoding="utf-8"
    )

    # 元数据：记录模型、预处理、版本，模型重训后必须重建。
    checkpoint_path = CLASSIFY_DIR / f"{group}_resnet18" / "best.pt"
    checkpoint_stat = checkpoint_path.stat()
    metadata = {
        "library_version": LIBRARY_VERSION,
        "feature_dim": FEATURE_DIM,
        "architecture": checkpoint["architecture"],
        "group": group,
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_mtime": checkpoint_stat.st_mtime,
        "checkpoint_size": checkpoint_stat.st_size,
        "img_size": img_size,
        "imagenet_mean": list(mean),
        "imagenet_std": list(std),
        "samples": len(labels),
        "num_classes": len(classes),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "模型重新训练后必须重建特征库。",
    }
    (args.output / f"{group}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stats = similarity_stats(matrix, labels)
    stats.update({"classes": classes, "samples": len(labels)})
    (args.output / f"{group}_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[{group}] {len(labels)}个特征, {len(classes)}个类中心, "
        f"同类sim均值={stats['same_mean']} (min={stats['same_min']}), "
        f"异类sim均值={stats['diff_mean']} (max={stats['diff_max']})"
    )
    return {"classes": classes, "centers": centers, "metadata": metadata}


def update_db_feature_index(args: argparse.Namespace) -> None:
    """把每个SKU的类中心标识写入 products.feature_index（不存向量本体）。"""
    from database.goods_dao import GoodsDao

    dao = GoodsDao()
    updated = 0
    try:
        for group in GROUPS:
            classes_path = args.output / f"{group}_classes.json"
            if not classes_path.is_file():
                continue
            classes = json.loads(classes_path.read_text(encoding="utf-8"))
            for index, class_name in enumerate(classes):
                # 标识格式：lib<版本>_center<索引>_<SKU>，指向特征库中的类中心。
                marker = f"lib{LIBRARY_VERSION}_center{index}_{class_name}"
                ok = dao.update_by_model_class(class_name, feature_index=marker)
                if ok:
                    updated += 1
        print(f"已更新 {updated} 个SKU的 feature_index。")
    finally:
        dao.close()


def main() -> None:
    args = parse_args()
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    if not args.data.is_dir():
        raise FileNotFoundError(f"数据集目录不存在：{args.data}")

    device = resolve_device(args.device)
    print(f"设备：{device}，特征库版本：{LIBRARY_VERSION}")

    for group in GROUPS:
        build_library(group, args, device)

    print("特征库构建完成：", args.output)
    if args.update_db:
        update_db_feature_index(args)


if __name__ == "__main__":
    main()
