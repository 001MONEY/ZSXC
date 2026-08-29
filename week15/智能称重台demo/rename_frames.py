"""将人工筛选后的分类图片批量改为简短、连续的文件名。

脚本默认只预览，不会修改文件。使用示例：

    python rename_frames.py
    python rename_frames.py --apply

最终文件名格式为 BOTTLE_01_0001.jpg。正式执行成功后，脚本会生成一份
CSV 对照表，记录每张图片修改前后的路径，方便后续追溯。
"""

from __future__ import annotations

import argparse
import csv
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SKU_PATTERN = re.compile(r"^(BOTTLE_\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class RenameItem:
    source: Path
    target: Path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="安全地批量重命名分类图片。")
    parser.add_argument(
        "--root",
        type=Path,
        default=project_root / "classification_dataset_raw" / "train",
        help="数据集目录，该目录下的每个一级子目录代表一个 SKU 类别。",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正执行重命名；不添加该参数时仅显示预览。",
    )
    return parser.parse_args()


def class_prefix(directory_name: str) -> str:
    match = SKU_PATTERN.match(directory_name)
    if not match:
        raise ValueError(
            f"类别目录名称必须以 BOTTLE_<数字> 开头：{directory_name}"
        )
    return match.group(1).upper()


def build_plan(root: Path) -> dict[Path, list[RenameItem]]:
    plan: dict[Path, list[RenameItem]] = {}
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    for class_dir in class_dirs:
        prefix = class_prefix(class_dir.name)
        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        items = [
            RenameItem(
                source=image,
                target=class_dir / f"{prefix}_{index:04d}{image.suffix.lower()}",
            )
            for index, image in enumerate(images, start=1)
        ]
        plan[class_dir] = items
    return plan


def validate_plan(plan: dict[Path, list[RenameItem]]) -> None:
    all_sources = {item.source.resolve() for items in plan.values() for item in items}
    all_targets: set[Path] = set()
    for items in plan.values():
        for item in items:
            target = item.target.resolve()
            if target in all_targets:
                raise RuntimeError(f"重命名计划中存在重复的目标名称：{item.target}")
            all_targets.add(target)
            if item.target.exists() and target not in all_sources:
                raise RuntimeError(f"目标文件已经存在且不在本次重命名范围内：{item.target}")


def print_preview(root: Path, plan: dict[Path, list[RenameItem]]) -> None:
    total = 0
    for class_dir, items in plan.items():
        total += len(items)
        print(f"\n[{class_dir.name}] 共 {len(items)} 张图片")
        preview = items[:3]
        if len(items) > 4:
            preview += items[-1:]
        for item in preview:
            marker = "（无需修改）" if item.source == item.target else ""
            print(f"  {item.source.name} -> {item.target.name}{marker}")
        if len(items) > 4:
            print(f"  ……其余 {len(items) - 4} 张")
    print(f"\n图片总数：{total}")
    print(f"数据集目录：{root}")


def apply_plan(root: Path, plan: dict[Path, list[RenameItem]]) -> Path:
    changed = [item for items in plan.values() for item in items if item.source != item.target]
    if not changed:
        print("所有图片名称已经符合要求，无需再次修改。")
        return root.parent / "rename_manifest_no_changes.csv"

    # 第一阶段：先把所有待处理文件改成唯一的临时名称。
    # 这样可以避免最终名称与旧名称（例如 BOTTLE_01_0001.jpg）发生冲突。
    temporary_moves: list[tuple[Path, Path, Path]] = []
    token = uuid.uuid4().hex
    try:
        for index, item in enumerate(changed, start=1):
            temporary = item.source.with_name(
                f".__rename_tmp_{token}_{index:06d}{item.source.suffix.lower()}"
            )
            item.source.rename(temporary)
            temporary_moves.append((temporary, item.source, item.target))

        # 第二阶段：把临时文件依次改为最终的简短名称。
        for temporary, _source, target in temporary_moves:
            temporary.rename(target)
    except Exception:
        # 如果中途发生异常，尽可能把已经改动的文件恢复为原名称。
        for temporary, source, target in reversed(temporary_moves):
            if temporary.exists() and not source.exists():
                temporary.rename(source)
            elif target.exists() and not source.exists():
                target.rename(source)
        raise

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = root.parent / f"rename_manifest_{timestamp}.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["old_relative_path", "new_relative_path"])
        for item in changed:
            writer.writerow(
                [
                    item.source.relative_to(root.parent),
                    item.target.relative_to(root.parent),
                ]
            )
    return manifest


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"数据集目录不存在：{root}")

    plan = build_plan(root)
    validate_plan(plan)
    print_preview(root, plan)

    if not args.apply:
        print("\n当前仅为预览。添加 --apply 参数后才会真正执行重命名。")
        return

    manifest = apply_plan(root, plan)
    print("\n批量重命名已成功完成。")
    print(f"新旧名称对照表：{manifest}")


if __name__ == "__main__":
    main()
