"""从YOLO检测视频中按固定帧率抽图，并生成标准数据集目录。

默认输出结构：

    yolo_dataset_raw/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/

`labels` 目录暂时为空，后续从Label Studio导出标注后再放入。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from extract_frames import extract_video


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="从YOLO检测视频中按时间间隔抽取图片。")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=project_root / "video" / "YOLO Data",
        help="包含train、val、test视频目录的YOLO视频根目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "yolo_dataset_raw",
        help="YOLO原始图片和标签目录。",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="需要处理的数据集分组，默认处理train、val和test。",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="每秒视频导出的图片数量，默认为1。",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG图片质量，取值范围为1～100，默认为95。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的同名图片；默认跳过已有图片。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_fps <= 0:
        raise SystemExit("--sample-fps 必须大于0")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality 必须在1～100之间")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    results = []

    for split in args.splits:
        input_dir = input_root / split
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        videos = sorted(input_dir.glob("*.mp4")) if input_dir.is_dir() else []
        if not videos:
            print(f"[跳过] {input_dir} 中没有MP4视频")
            continue

        print(f"\n[{split}] 共{len(videos)}段视频")
        for video in videos:
            result = extract_video(
                video_path=video,
                sku=split,
                output_dir=image_dir,
                sample_fps=args.sample_fps,
                jpeg_quality=args.jpeg_quality,
                overwrite=args.overwrite,
            )
            results.append((split, result))
            print(
                f"  {video.name}：时长{result.duration_seconds:.1f}秒，"
                f"导出{result.exported}张，跳过已有图片{result.skipped_existing}张"
            )

    if not results:
        raise SystemExit("没有处理任何YOLO视频。")

    manifest = output_root / "extraction_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["split", "video", "duration_seconds", "exported", "skipped_existing"]
        )
        for split, result in results:
            writer.writerow(
                [
                    split,
                    result.video,
                    f"{result.duration_seconds:.3f}",
                    result.exported,
                    result.skipped_existing,
                ]
            )

    total_exported = sum(result.exported for _split, result in results)
    total_skipped = sum(result.skipped_existing for _split, result in results)
    print("\nYOLO抽帧完成")
    print(f"已处理视频：{len(results)}段")
    print(f"已导出图片：{total_exported}张")
    print(f"已跳过图片：{total_skipped}张")
    print(f"输出目录：{output_root}")
    print(f"抽帧记录：{manifest}")


if __name__ == "__main__":
    main()
