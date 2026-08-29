"""批量缩短YOLO图片名称，并同步重命名已有标签文件。

脚本默认只预览，添加 --apply 后才会真正执行：

    python rename_yolo_images.py
    python rename_yolo_images.py --apply

最终文件名示例：yolo_train_0001.jpg、yolo_test_0001.jpg。
"""

from __future__ import annotations

import argparse
import csv
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class RenameItem:
    split: str
    source_image: Path
    target_image: Path
    source_label: Path | None
    target_label: Path | None


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="批量重命名YOLO图片和对应标签。")
    parser.add_argument(
        "--root",
        type=Path,
        default=project_root / "yolo_dataset_raw",
        help="包含images和labels目录的YOLO数据集根目录。",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="需要处理的分组，默认处理train、val和test。",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正执行重命名；不添加该参数时仅显示预览。",
    )
    return parser.parse_args()


def build_plan(root: Path, splits: list[str]) -> list[RenameItem]:
    plan = []
    for split in splits:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir():
            continue

        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for index, source_image in enumerate(images, start=1):
            target_stem = f"yolo_{split}_{index:04d}"
            target_image = image_dir / f"{target_stem}{source_image.suffix.lower()}"
            possible_label = label_dir / f"{source_image.stem}.txt"
            source_label = possible_label if possible_label.exists() else None
            target_label = label_dir / f"{target_stem}.txt" if source_label else None
            plan.append(
                RenameItem(
                    split=split,
                    source_image=source_image,
                    target_image=target_image,
                    source_label=source_label,
                    target_label=target_label,
                )
            )
    return plan


def validate_plan(plan: list[RenameItem]) -> None:
    image_sources = {item.source_image.resolve() for item in plan}
    image_targets: set[Path] = set()
    label_sources = {
        item.source_label.resolve() for item in plan if item.source_label is not None
    }
    label_targets: set[Path] = set()

    for item in plan:
        image_target = item.target_image.resolve()
        if image_target in image_targets:
            raise RuntimeError(f"出现重复的目标图片名称：{item.target_image}")
        image_targets.add(image_target)
        if item.target_image.exists() and image_target not in image_sources:
            raise RuntimeError(f"目标图片已经存在：{item.target_image}")

        if item.target_label is not None:
            label_target = item.target_label.resolve()
            if label_target in label_targets:
                raise RuntimeError(f"出现重复的目标标签名称：{item.target_label}")
            label_targets.add(label_target)
            if item.target_label.exists() and label_target not in label_sources:
                raise RuntimeError(f"目标标签已经存在：{item.target_label}")


def print_preview(plan: list[RenameItem]) -> None:
    for split in sorted({item.split for item in plan}):
        items = [item for item in plan if item.split == split]
        print(f"\n[{split}] 共{len(items)}张图片")
        preview = items[:3]
        if len(items) > 4:
            preview += items[-1:]
        for item in preview:
            marker = "（无需修改）" if item.source_image == item.target_image else ""
            print(f"  {item.source_image.name} -> {item.target_image.name}{marker}")
        if len(items) > 4:
            print(f"  ……其余{len(items) - 4}张")
    print(f"\n图片总数：{len(plan)}")


def apply_plan(root: Path, plan: list[RenameItem]) -> Path:
    changed = [item for item in plan if item.source_image != item.target_image]
    token = uuid.uuid4().hex
    temporary_moves: list[tuple[Path, Path, Path]] = []

    try:
        # 第一阶段：所有图片和标签先改成唯一的临时名称，避免名称冲突。
        for index, item in enumerate(changed, start=1):
            temporary_image = item.source_image.with_name(
                f".__yolo_tmp_{token}_{index:06d}{item.source_image.suffix.lower()}"
            )
            item.source_image.rename(temporary_image)
            temporary_moves.append((temporary_image, item.source_image, item.target_image))

            if item.source_label is not None and item.target_label is not None:
                temporary_label = item.source_label.with_name(
                    f".__yolo_tmp_{token}_{index:06d}.txt"
                )
                item.source_label.rename(temporary_label)
                temporary_moves.append((temporary_label, item.source_label, item.target_label))

        # 第二阶段：将临时文件改为最终的连续短名称。
        for temporary, _source, target in temporary_moves:
            temporary.rename(target)
    except Exception:
        # 中途失败时尽可能恢复原名称。
        for temporary, source, target in reversed(temporary_moves):
            if temporary.exists() and not source.exists():
                temporary.rename(source)
            elif target.exists() and not source.exists():
                target.rename(source)
        raise

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = root / f"rename_manifest_{timestamp}.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["split", "old_image", "new_image", "old_label", "new_label"]
        )
        for item in changed:
            writer.writerow(
                [
                    item.split,
                    item.source_image.relative_to(root),
                    item.target_image.relative_to(root),
                    item.source_label.relative_to(root) if item.source_label else "",
                    item.target_label.relative_to(root) if item.target_label else "",
                ]
            )
    return manifest


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"YOLO数据集目录不存在：{root}")

    plan = build_plan(root, args.splits)
    if not plan:
        raise SystemExit("没有找到可以重命名的图片。")
    validate_plan(plan)
    print_preview(plan)

    if not args.apply:
        print("\n当前仅为预览，添加 --apply 后才会真正执行重命名。")
        return

    manifest = apply_plan(root, plan)
    print("\nYOLO图片重命名完成。")
    print(f"新旧名称对照表：{manifest}")


if __name__ == "__main__":
    main()
