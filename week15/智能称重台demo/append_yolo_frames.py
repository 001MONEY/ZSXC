"""增量抽取后来补充的YOLO视频，并从现有短文件名后继续编号。

脚本通过 `yolo_dataset_raw/extraction_manifest.csv` 判断哪些视频已经处理，
因此不会重复抽取旧视频。默认每秒抽取1张图片。
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2


@dataclass
class VideoResult:
    split: str
    video: str
    duration_seconds: float
    exported: int


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="只抽取尚未处理的新增YOLO视频。")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=project_root / "video" / "YOLO Data",
        help="包含train、val和test视频目录的根目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "yolo_dataset_raw",
        help="已经存在的YOLO图片数据集根目录。",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="需要检查的分组，默认检查train、val和test。",
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
    return parser.parse_args()


def load_processed_videos(manifest: Path) -> set[tuple[str, str]]:
    if not manifest.exists():
        return set()
    with manifest.open("r", newline="", encoding="utf-8-sig") as file:
        return {(row["split"], row["video"]) for row in csv.DictReader(file)}


def next_image_number(image_dir: Path, split: str) -> int:
    pattern = re.compile(rf"^yolo_{re.escape(split)}_(\d+)$")
    numbers = []
    for image in image_dir.glob("*.*"):
        match = pattern.match(image.stem)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def write_jpeg(path: Path, frame, quality: int) -> None:
    # 避免OpenCV在Windows中文路径下静默写入失败。
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"JPEG编码失败：{path}")
    path.write_bytes(encoded.tobytes())


def extract_new_video(
    video: Path,
    split: str,
    image_dir: Path,
    start_number: int,
    sample_fps: float,
    jpeg_quality: int,
):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(f"视频元数据无效：{video}")

    frame_interval = source_fps / sample_fps
    next_sample_index = frame_interval / 2.0
    sample_number = 0
    frame_index = 0
    image_number = start_number
    frame_rows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index + 1e-9 >= next_sample_index:
            timestamp_ms = round(frame_index / source_fps * 1000)
            filename = f"yolo_{split}_{image_number:04d}.jpg"
            output_path = image_dir / filename
            if output_path.exists():
                cap.release()
                raise RuntimeError(f"目标图片已经存在：{output_path}")
            write_jpeg(output_path, frame, jpeg_quality)
            frame_rows.append(
                {
                    "split": split,
                    "source_video": video.name,
                    "sample_number": sample_number,
                    "timestamp_ms": timestamp_ms,
                    "frame_index": frame_index,
                    "output_image": filename,
                }
            )
            image_number += 1
            sample_number += 1
            next_sample_index = (sample_number + 0.5) * frame_interval
        frame_index += 1

    cap.release()
    return (
        VideoResult(
            split=split,
            video=video.name,
            duration_seconds=frame_count / source_fps,
            exported=len(frame_rows),
        ),
        frame_rows,
        image_number,
    )


def write_main_manifest(manifest: Path, existing_rows: list[dict], results: list[VideoResult]) -> None:
    rows = list(existing_rows)
    for result in results:
        rows.append(
            {
                "split": result.split,
                "video": result.video,
                "duration_seconds": f"{result.duration_seconds:.3f}",
                "exported": result.exported,
                "skipped_existing": 0,
            }
        )
    with manifest.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "video", "duration_seconds", "exported", "skipped_existing"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.sample_fps <= 0:
        raise SystemExit("--sample-fps 必须大于0")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality 必须在1～100之间")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    manifest = output_root / "extraction_manifest.csv"
    existing_rows = []
    if manifest.exists():
        with manifest.open("r", newline="", encoding="utf-8-sig") as file:
            existing_rows = list(csv.DictReader(file))
    processed = {(row["split"], row["video"]) for row in existing_rows}

    results = []
    all_frame_rows = []
    for split in args.splits:
        video_dir = input_root / split
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        videos = [
            video
            for video in sorted(video_dir.glob("*.mp4"))
            if (split, video.name) not in processed
        ] if video_dir.is_dir() else []
        if not videos:
            print(f"[{split}] 没有发现新增视频")
            continue

        image_number = next_image_number(image_dir, split)
        print(f"\n[{split}] 发现{len(videos)}段新增视频，从编号{image_number:04d}开始")
        for video in videos:
            result, frame_rows, image_number = extract_new_video(
                video=video,
                split=split,
                image_dir=image_dir,
                start_number=image_number,
                sample_fps=args.sample_fps,
                jpeg_quality=args.jpeg_quality,
            )
            results.append(result)
            all_frame_rows.extend(frame_rows)
            print(
                f"  {video.name}：时长{result.duration_seconds:.1f}秒，"
                f"导出{result.exported}张"
            )

    if not results:
        print("没有需要增量抽取的视频。")
        return

    write_main_manifest(manifest, existing_rows, results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_manifest = output_root / f"incremental_frame_manifest_{timestamp}.csv"
    with frame_manifest.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "source_video",
                "sample_number",
                "timestamp_ms",
                "frame_index",
                "output_image",
            ],
        )
        writer.writeheader()
        writer.writerows(all_frame_rows)

    print("\n增量抽帧完成")
    print(f"新增视频：{len(results)}段")
    print(f"新增图片：{len(all_frame_rows)}张")
    print(f"逐帧对照表：{frame_manifest}")


if __name__ == "__main__":
    main()
