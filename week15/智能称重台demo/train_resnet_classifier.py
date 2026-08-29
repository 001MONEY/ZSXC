r"""训练智能称重台四个包装大类的SKU分类模型。

每个包装大类分别训练一个6分类ResNet，数据目录为：

    classification_dataset_from_videos/<group>/train/<SKU>/*.jpg
    classification_dataset_from_videos/<group>/val/<SKU>/*.jpg

安全起见，不带 ``--train`` 时只检查数据和运行环境，不会开始训练：

    D:\project\step1\env\python.exe train_resnet_classifier.py

显式启动全部四个模型的训练：

    D:\project\step1\env\python.exe train_resnet_classifier.py --train

只训练一个包装大类：

    D:\project\step1\env\python.exe train_resnet_classifier.py --train --groups bag
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT / "classification_dataset_from_videos"
DEFAULT_RUNS = PROJECT_ROOT / "runs" / "classify"
GROUPS = ("bag", "bottle", "box", "cylinder")
GROUP_PREFIXES = {
    "bag": "BAG_",
    "bottle": "BOTTLE_",
    "box": "BOX_",
    "cylinder": "CYLINDER_",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SquarePad:
    """保持商品宽高比，以灰色边缘补成正方形。"""

    def __init__(self, fill: tuple[int, int, int] = (114, 114, 114)) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        return ImageOps.expand(image, border=(left, top, right, bottom), fill=self.fill)


class EnsureRGB:
    """可被Windows DataLoader子进程序列化的RGB转换。"""

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练四个独立的ResNet SKU分类模型。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="分类数据集根目录。")
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS), help="训练的大类。")
    parser.add_argument("--architecture", choices=("resnet18", "resnet50"), default="resnet18")
    parser.add_argument("--epochs", type=int, default=40, help="最大训练轮数，默认40。")
    parser.add_argument("--batch", type=int, default=32, help="批大小，6GB显存默认32。")
    parser.add_argument("--img-size", type=int, default=224, help="模型输入尺寸，默认224。")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW初始学习率。")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW权重衰减。")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="交叉熵标签平滑。")
    parser.add_argument("--patience", type=int, default=8, help="验证集Top-1早停等待轮数。")
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=3,
        help="先冻结预训练骨干的轮数，默认3。",
    )
    parser.add_argument("--workers", type=int, default=4, help="数据加载进程数，默认4。")
    parser.add_argument("--device", default="0", help="训练设备：0、cuda:0、cpu或auto。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--project", type=Path, default=DEFAULT_RUNS, help="训练结果根目录。")
    parser.add_argument("--no-pretrained", action="store_true", help="不使用ImageNet预训练权重。")
    parser.add_argument("--exist-ok", action="store_true", help="允许复用并覆盖同名训练目录。")
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="跳过较慢的重复图片和train/val泄漏检查。",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="显式启动训练；不提供此参数时只检查数据和环境。",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch <= 0 or args.img_size <= 0:
        raise ValueError("epochs、batch和img-size必须大于0。")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate必须大于0，weight-decay不能小于0。")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("label-smoothing必须在0到1之间。")
    if args.patience < 0 or args.freeze_backbone_epochs < 0 or args.workers < 0:
        raise ValueError("patience、freeze-backbone-epochs和workers不能小于0。")
    if args.freeze_backbone_epochs >= args.epochs:
        raise ValueError("freeze-backbone-epochs必须小于epochs。")


def resolve_device(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized.isdigit():
        normalized = f"cuda:{normalized}"
    device = torch.device(normalized)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了GPU训练，但当前PyTorch无法使用CUDA。可添加 --device cpu。")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"GPU编号越界：{device}，当前共有{torch.cuda.device_count()}张GPU。")
    return device


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_images(files: list[Path], split: str, group: str) -> None:
    errors: list[str] = []
    for path in files:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:  # Pillow会给出具体的文件损坏原因。
            errors.append(f"{path}: {error}")
    if errors:
        raise ValueError(f"{group}/{split}存在{len(errors)}张损坏图片：\n" + "\n".join(errors[:20]))


def validate_group_dataset(
    data_root: Path,
    group: str,
    hash_check: bool,
) -> dict[str, Any]:
    group_root = data_root / group
    split_classes: dict[str, list[str]] = {}
    split_counts: dict[str, Counter[str]] = {}
    split_files: dict[str, list[Path]] = {}

    for split in ("train", "val"):
        split_root = group_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"数据目录不存在：{split_root}")
        class_dirs = sorted(path for path in split_root.iterdir() if path.is_dir())
        class_names = [path.name for path in class_dirs]
        if len(class_names) != 6:
            raise ValueError(f"{group}/{split}应有6个SKU目录，实际为{len(class_names)}：{class_names}")
        bad_names = [name for name in class_names if not name.upper().startswith(GROUP_PREFIXES[group])]
        if bad_names:
            raise ValueError(f"{group}/{split}存在前缀不正确的SKU目录：{bad_names}")

        counts: Counter[str] = Counter()
        files: list[Path] = []
        for class_dir in class_dirs:
            class_images = image_files(class_dir)
            if not class_images:
                raise ValueError(f"SKU目录没有图片：{class_dir}")
            counts[class_dir.name] = len(class_images)
            files.extend(class_images)
        validate_images(files, split, group)
        split_classes[split] = class_names
        split_counts[split] = counts
        split_files[split] = files

    if split_classes["train"] != split_classes["val"]:
        raise ValueError(
            f"{group}的train/val类别不一致：train={split_classes['train']}，val={split_classes['val']}"
        )

    duplicate_counts = {"train": 0, "val": 0}
    if hash_check:
        split_hashes: dict[str, dict[str, list[Path]]] = {}
        for split in ("train", "val"):
            hashes: dict[str, list[Path]] = defaultdict(list)
            for path in split_files[split]:
                hashes[sha256(path)].append(path)
            split_hashes[split] = hashes
            duplicate_counts[split] = sum(len(paths) - 1 for paths in hashes.values() if len(paths) > 1)
        overlap = sorted(set(split_hashes["train"]) & set(split_hashes["val"]))
        if overlap:
            examples = [
                f"train={split_hashes['train'][value][0]} | val={split_hashes['val'][value][0]}"
                for value in overlap[:10]
            ]
            raise ValueError(f"{group}发现{len(overlap)}组train/val完全相同图片：\n" + "\n".join(examples))

    return {
        "classes": split_classes["train"],
        "counts": split_counts,
        "files": split_files,
        "duplicates": duplicate_counts,
    }


def build_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            EnsureRGB(),
            SquarePad(),
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.02)],
                p=0.7,
            ),
            transforms.RandomRotation(8, interpolation=InterpolationMode.BILINEAR, fill=114),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            EnsureRGB(),
            SquarePad(),
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def build_model(architecture: str, num_classes: int, pretrained: bool) -> tuple[nn.Module, str | None]:
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif architecture == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:  # argparse已经限制取值，此分支用于防御式校验。
        raise ValueError(f"不支持的网络：{architecture}")
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)
    return model, weights.url if weights is not None else None


def cached_weight_path(architecture: str) -> Path:
    if architecture == "resnet18":
        url = models.ResNet18_Weights.DEFAULT.url
    else:
        url = models.ResNet50_Weights.DEFAULT.url
    return Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loaders(
    data_root: Path,
    group: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, DataLoader], datasets.ImageFolder, datasets.ImageFolder]:
    train_transform, val_transform = build_transforms(args.img_size)
    train_dataset = datasets.ImageFolder(data_root / group / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_root / group / "val", transform=val_transform)
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(f"{group}的ImageFolder类别映射不一致。")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common = {
        "batch_size": args.batch,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_worker,
    }
    loaders = {
        "train": DataLoader(train_dataset, shuffle=True, generator=generator, **common),
        "val": DataLoader(val_dataset, shuffle=False, **common),
    }
    return loaders, train_dataset, val_dataset


def class_weights(dataset: datasets.ImageFolder, device: torch.device) -> torch.Tensor:
    counts = Counter(target for _, target in dataset.samples)
    total = len(dataset)
    weights = [total / (len(dataset.classes) * counts[index]) for index in range(len(dataset.classes))]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def freeze_backbone(model: nn.Module, frozen: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = not frozen
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    backbone_frozen: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    if training:
        model.train()
        if backbone_frozen:
            for name, module in model.named_children():
                if name != "fc":
                    module.eval()
    else:
        model.eval()

    total_loss = 0.0
    total = 0
    top1_correct = 0
    top3_correct = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(inputs)
                loss = criterion(logits, targets)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = targets.size(0)
        total += batch_size
        total_loss += float(loss.detach()) * batch_size
        topk = logits.topk(k=min(3, logits.shape[1]), dim=1).indices
        top1_correct += int((topk[:, 0] == targets).sum())
        top3_correct += int((topk == targets.unsqueeze(1)).any(dim=1).sum())

    return {
        "loss": total_loss / max(total, 1),
        "top1": top1_correct / max(total, 1),
        "top3": top3_correct / max(total, 1),
    }


@torch.inference_mode()
def confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    model.eval()
    for inputs, targets in loader:
        predictions = model(inputs.to(device, non_blocking=True)).argmax(dim=1).cpu().tolist()
        for actual, predicted in zip(targets.tolist(), predictions, strict=True):
            matrix[actual][predicted] += 1
    return matrix


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_top1: float,
    group: str,
    classes: list[str],
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "architecture": args.architecture,
        "group": group,
        "classes": classes,
        "class_to_idx": {name: index for index, name in enumerate(classes)},
        "img_size": args.img_size,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "epoch": epoch,
        "best_val_top1": best_top1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_group(
    data_root: Path,
    group: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    run_dir = args.project / f"{group}_{args.architecture}"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.exist_ok:
        raise FileExistsError(f"训练目录已存在且非空：{run_dir}。确认覆盖时添加 --exist-ok。")
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders, train_dataset, val_dataset = make_loaders(data_root, group, args, device)
    pretrained = not args.no_pretrained
    model, weights_url = build_model(args.architecture, len(train_dataset.classes), pretrained)
    model.to(device)
    freeze_epochs = args.freeze_backbone_epochs if pretrained else 0
    freeze_backbone(model, freeze_epochs > 0)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_dataset, device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    config = {
        "group": group,
        "architecture": args.architecture,
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "epochs": args.epochs,
        "batch": args.batch,
        "img_size": args.img_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "patience": args.patience,
        "freeze_backbone_epochs": freeze_epochs,
        "seed": args.seed,
        "pretrained": pretrained,
        "pretrained_weights_url": weights_url,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[{group}] 开始训练：train={len(train_dataset)}，val={len(val_dataset)}")
    print(f"[{group}] 类别映射：{train_dataset.class_to_idx}")
    print(f"[{group}] 输出目录：{run_dir}")
    history: list[dict[str, Any]] = []
    best_top1 = -1.0
    epochs_without_improvement = 0
    started_at = time.time()

    for epoch in range(1, args.epochs + 1):
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            freeze_backbone(model, False)
            print(f"[{group}] 第{epoch}轮：已解冻ResNet骨干网络。")
        backbone_frozen = epoch <= freeze_epochs
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer,
            scaler,
            backbone_frozen=backbone_frozen,
        )
        val_metrics = run_epoch(model, loaders["val"], criterion, device, None, None)
        scheduler.step(val_metrics["top1"])
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "learning_rate": f"{current_lr:.8f}",
            "train_loss": f"{train_metrics['loss']:.6f}",
            "train_top1": f"{train_metrics['top1']:.6f}",
            "train_top3": f"{train_metrics['top3']:.6f}",
            "val_loss": f"{val_metrics['loss']:.6f}",
            "val_top1": f"{val_metrics['top1']:.6f}",
            "val_top3": f"{val_metrics['top3']:.6f}",
            "backbone_frozen": backbone_frozen,
        }
        history.append(row)
        write_history(run_dir / "history.csv", history)

        improved = val_metrics["top1"] > best_top1
        if improved:
            best_top1 = val_metrics["top1"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_top1,
            group,
            train_dataset.classes,
            args,
        )
        if improved:
            save_checkpoint(
                run_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_top1,
                group,
                train_dataset.classes,
                args,
            )

        print(
            f"[{group}] epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_top1={train_metrics['top1']:.2%} "
            f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1']:.2%} "
            f"val_top3={val_metrics['top3']:.2%} lr={current_lr:.2e}"
        )
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"[{group}] 验证集Top-1连续{args.patience}轮未提升，提前停止。")
            break

    best_checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    matrix = confusion_matrix(model, loaders["val"], device, len(train_dataset.classes))
    with (run_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *train_dataset.classes])
        for class_name, values in zip(train_dataset.classes, matrix, strict=True):
            writer.writerow([class_name, *values])

    result = {
        "group": group,
        "best_val_top1": best_top1,
        "best_epoch": int(best_checkpoint["epoch"]),
        "epochs_completed": len(history),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "best_checkpoint": str(run_dir / "best.pt"),
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[{group}] 训练完成，最佳Top-1={best_top1:.2%}：{run_dir / 'best.pt'}")
    return result


def print_dataset_summary(group: str, summary: dict[str, Any]) -> None:
    print(f"\n[{group}] 6个SKU：")
    for class_name in summary["classes"]:
        print(
            f"  {class_name}: train={summary['counts']['train'][class_name]}，"
            f"val={summary['counts']['val'][class_name]}"
        )
    print(
        f"  合计：train={len(summary['files']['train'])}，val={len(summary['files']['val'])}，"
        f"精确重复：train={summary['duplicates']['train']}，val={summary['duplicates']['val']}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.data = args.data.resolve()
    args.project = args.project.resolve()
    if not args.data.is_dir():
        raise FileNotFoundError(f"分类数据集根目录不存在：{args.data}")

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Python：{sys.version.split()[0]}")
    print(f"PyTorch：{torch.__version__}，torchvision：{__import__('torchvision').__version__}")
    print(f"设备：{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")
    print(f"网络：{args.architecture}，ImageNet预训练：{not args.no_pretrained}")
    if not args.no_pretrained:
        weight_path = cached_weight_path(args.architecture)
        print(f"预训练权重缓存：{'已存在' if weight_path.is_file() else '训练启动时自动下载'}（{weight_path}）")

    summaries: dict[str, dict[str, Any]] = {}
    for group in args.groups:
        summary = validate_group_dataset(args.data, group, not args.skip_hash_check)
        summaries[group] = summary
        print_dataset_summary(group, summary)

    # 无权重构建用于提前发现torchvision模型接口或分类头配置问题，不触发下载。
    smoke_model, _ = build_model(args.architecture, num_classes=6, pretrained=False)
    with torch.inference_mode():
        smoke_output = smoke_model(torch.zeros(1, 3, args.img_size, args.img_size))
    if tuple(smoke_output.shape) != (1, 6):
        raise RuntimeError(f"模型输出形状不正确：{tuple(smoke_output.shape)}，预期(1, 6)。")
    print(f"\n模型结构检查通过：输入=(1, 3, {args.img_size}, {args.img_size})，输出=(1, 6)。")

    if not args.train:
        print("数据集、运行环境和模型结构检查通过；未开始训练。显式添加 --train 才会启动训练。")
        return

    results = [train_group(args.data, group, args, device) for group in args.groups]
    args.project.mkdir(parents=True, exist_ok=True)
    (args.project / f"{args.architecture}_training_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n全部指定大类训练完成。")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
