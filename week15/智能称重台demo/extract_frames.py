"""按照固定时间间隔，从分类商品视频中提取图片。

默认输出目录结构如下，可直接兼容 torchvision 的 ImageFolder：

    classification_dataset_raw/
        train/
            BAG_01_kebike_chips/
            BOX_01_strawberry_cookie/
            CYLINDER_01_cocacola/
            ...

使用示例：
    python extract_frames.py
    python extract_frames.py --groups BAG BOX CYLINDER
    python extract_frames.py --sample-fps 2
    python extract_frames.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoResult:
    sku: str
    video: str
    duration_seconds: float
    exported: int
    skipped_existing: int


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="从分类商品视频中按时间间隔抽取图片。")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=project_root / "video",
        help="包含 BAG、BOX、BOTTLE、CYLINDER 等大类目录的视频根目录。",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["BAG", "BOX", "CYLINDER"],
        help="需要处理的商品大类；默认处理 BAG、BOX 和 CYLINDER，不重复抽取BOTTLE。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "classification_dataset_raw",
        help="用于保存抽帧图片的输出目录。",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="需要处理的数据集分组，例如 train、val 或 test。",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="每秒视频导出的图片数量，默认为 1。",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG 图片质量，取值范围为 1～100，默认为 95。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的同名图片；默认跳过已有图片。",
    )
    return parser.parse_args()


def write_jpeg(path: Path, frame, quality: int) -> None:
    # Windows 路径包含中文时，cv2.imwrite 可能静默写入失败，
    # 因此先在内存中编码，再使用 pathlib 写入文件。
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"JPEG 图片编码失败：{path}")
    path.write_bytes(encoded.tobytes())


def extract_video(
    video_path: Path,
    sku: str,
    output_dir: Path,
    sample_fps: float,
    jpeg_quality: int,
    overwrite: bool,
) -> VideoResult:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(f"视频元数据无效：{video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_interval = source_fps / sample_fps
    # 从每个采样时间段的中间位置抽帧，例如 0.5 秒、1.5 秒、2.5 秒……
    # 这样可以避开视频刚开始和刚结束时可能出现的准备动作。
    next_sample_index = frame_interval / 2.0
    sample_number = 0
    frame_index = 0
    exported = 0
    skipped_existing = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index + 1e-9 >= next_sample_index:
            timestamp_ms = round(frame_index / source_fps * 1000)
            filename = (
                f"{video_path.stem}__s{sample_number:04d}"
                f"__t{timestamp_ms:07d}ms__f{frame_index:06d}.jpg"
            )
            output_path = output_dir / filename
            if output_path.exists() and not overwrite:
                skipped_existing += 1
            else:
                write_jpeg(output_path, frame, jpeg_quality)
                exported += 1

            sample_number += 1
            next_sample_index = (sample_number + 0.5) * frame_interval

        frame_index += 1

    cap.release()
    return VideoResult(
        sku=sku,
        video=video_path.name,
        duration_seconds=frame_count / source_fps,
        exported=exported,
        skipped_existing=skipped_existing,
    )


def main() -> None:
    args = parse_args()
    if args.sample_fps <= 0:
        raise SystemExit("--sample-fps 必须大于 0")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality 必须在 1～100 之间")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    results: list[VideoResult] = []
    found_group = False
    for group in args.groups:
        group_dir = input_root / group
        if not group_dir.is_dir():
            print(f"[跳过] 商品大类目录不存在：{group_dir}")
            continue
        found_group = True
        sku_dirs = sorted(path for path in group_dir.iterdir() if path.is_dir())
        print(f"\n========== {group} ==========")
        for sku_dir in sku_dirs:
            video_dir = sku_dir / args.split
            videos = sorted(video_dir.glob("*.mp4")) if video_dir.is_dir() else []
            if not videos:
                print(f"[跳过] {sku_dir.name}：{video_dir} 中没有 MP4 文件")
                continue

            sku_output = output_root / args.split / sku_dir.name
            print(f"\n[{sku_dir.name}] 共 {len(videos)} 段视频")
            for video in videos:
                result = extract_video(
                    video_path=video,
                    sku=sku_dir.name,
                    output_dir=sku_output,
                    sample_fps=args.sample_fps,
                    jpeg_quality=args.jpeg_quality,
                    overwrite=args.overwrite,
                )
                results.append(result)
                print(
                    f"  {video.name}：时长 {result.duration_seconds:.1f} 秒，"
                    f"导出 {result.exported} 张，跳过已有图片 {result.skipped_existing} 张"
                )

    if not found_group:
        raise SystemExit(f"在视频根目录下没有找到指定的商品大类：{input_root}")
    if not results:
        raise SystemExit("没有处理任何视频。")

    group_tag = "_".join(group.upper() for group in args.groups)
    manifest_path = output_root / f"{args.split}_extraction_manifest_{group_tag}.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["sku", "video", "duration_seconds", "exported", "skipped_existing"]
        )
        for result in results:
            writer.writerow(
                [
                    result.sku,
                    result.video,
                    f"{result.duration_seconds:.3f}",
                    result.exported,
                    result.skipped_existing,
                ]
            )

    total_exported = sum(result.exported for result in results)
    total_skipped = sum(result.skipped_existing for result in results)
    print("\n抽帧完成")
    print(f"已处理视频：{len(results)} 段")
    print(f"已导出图片：{total_exported} 张")
    print(f"已跳过图片：{total_skipped} 张")
    print(f"输出目录：{output_root / args.split}")
    print(f"抽帧记录：{manifest_path}")


if __name__ == "__main__":
    main()
