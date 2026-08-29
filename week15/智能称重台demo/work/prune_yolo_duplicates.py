"""清理YOLO训练集中的连续近重复帧。

默认仅预览；添加 ``--apply`` 后，会把选中的图片（以及同名标签）
移动到 ``yolo_dataset_raw/rejected_duplicates``，不会永久删除。
"""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path


# 逐页人工查看接触表后选出的近重复帧。
# 保留了多商品、手部进入、遮挡、商品增减和少量空白背景等有效样本。
REJECT_NUMBERS = (
    2,
    7, 8, 9,
    13, 14,
    23,
    29,
    32,
    43, 44,
    51, 52,
    58,
    66,
    71, 72,
    75, 76,
    82, 83,
    90, 91,
    98, 99, 100, 102, 103, 104,
    110,
    127,
    129,
    135,
    151,
    158,
    161,
    168,
    180, 181, 183, 184,
    221,
    230,
    233,
    253,
    284,
    312,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="移出YOLO训练集中的连续近重复帧。")
    parser.add_argument(
        "--root",
        type=Path,
        default=project_root / "yolo_dataset_raw",
        help="YOLO数据集根目录。",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正移动文件；不添加时仅预览。",
    )
    return parser.parse_args()


def ensure_inside(child: Path, parent: Path) -> None:
    """确认操作目标位于指定目录内，避免误移动其他文件。"""
    child.resolve().relative_to(parent.resolve())


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    reject_root = root / "rejected_duplicates"
    reject_image_dir = reject_root / "images" / "train"
    reject_label_dir = reject_root / "labels" / "train"

    if not image_dir.is_dir():
        raise SystemExit(f"训练图片目录不存在：{image_dir}")

    ensure_inside(image_dir, root)
    ensure_inside(reject_root, root)

    rows: list[tuple[Path, Path, Path | None, Path | None]] = []
    missing: list[str] = []
    for number in REJECT_NUMBERS:
        source_image = image_dir / f"yolo_train_{number:04d}.jpg"
        target_image = reject_image_dir / source_image.name
        source_label = label_dir / f"{source_image.stem}.txt"
        target_label = reject_label_dir / source_label.name
        if not source_image.is_file():
            missing.append(source_image.name)
            continue
        if target_image.exists():
            raise SystemExit(f"暂存目录已有同名图片，请先检查：{target_image}")
        rows.append(
            (
                source_image,
                target_image,
                source_label if source_label.is_file() else None,
                target_label if source_label.is_file() else None,
            )
        )

    print(f"计划移出近重复图片：{len(rows)} 张")
    print(f"预计保留训练图片：{len(list(image_dir.glob('*.jpg'))) - len(rows)} 张")
    if missing:
        print("未找到：" + "、".join(missing))
    for source_image, _, _, _ in rows:
        print(f"  {source_image.name}")

    if missing:
        raise SystemExit("候选文件不完整，已停止，未修改任何文件。")
    if not args.apply:
        print("\n当前仅为预览，添加 --apply 后才会真正移动。")
        return

    reject_image_dir.mkdir(parents=True, exist_ok=True)
    moved: list[tuple[Path, Path]] = []
    try:
        for source_image, target_image, source_label, target_label in rows:
            shutil.move(str(source_image), str(target_image))
            moved.append((target_image, source_image))
            if source_label is not None and target_label is not None:
                reject_label_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_label), str(target_label))
                moved.append((target_label, source_label))
    except Exception:
        # 中途失败时，尽可能把已经移动的文件恢复到原位置。
        for moved_path, original_path in reversed(moved):
            if moved_path.exists() and not original_path.exists():
                original_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(moved_path), str(original_path))
        raise

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = reject_root / f"prune_manifest_{timestamp}.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["original_image", "rejected_image", "original_label", "rejected_label"])
        for source_image, target_image, source_label, target_label in rows:
            writer.writerow(
                [
                    source_image.relative_to(root),
                    target_image.relative_to(root),
                    source_label.relative_to(root) if source_label else "",
                    target_label.relative_to(root) if target_label else "",
                ]
            )

    print("\n近重复帧已移到可恢复目录。")
    print(f"清理清单：{manifest}")


if __name__ == "__main__":
    main()
