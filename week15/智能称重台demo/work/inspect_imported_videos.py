"""批量检查新导入的分类视频和YOLO视频，并生成代表帧拼图。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path(r"D:\project\step1\week15\智能称重台demo")
VIDEO_ROOT = PROJECT / "video"
OUTPUT = PROJECT / "work" / "imported_video_inspection"
CLASS_GROUPS = ["BAG", "BOX", "CYLINDER"]


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized = cv2.resize(
        frame,
        (round(source_width * scale), round(source_height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def read_video(path: Path, sample_positions: list[float]):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], {"path": str(path.relative_to(PROJECT)), "error": "open_failed"}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    samples = []
    for position in sample_positions:
        frame_index = min(max(0, round((frame_count - 1) * position)), frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append((position, frame))
        samples.append(
            {
                "position": position,
                "brightness": round(float(gray.mean()), 2),
                "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
            }
        )
    cap.release()
    return frames, {
        "path": str(path.relative_to(PROJECT)),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "duration_sec": round(frame_count / fps, 2) if fps > 0 else 0,
        "samples": samples,
    }


def save_jpeg(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError(f"无法编码图片：{path}")
    path.write_bytes(encoded.tobytes())


def build_classification_sheet(group: str):
    sku_dirs = sorted(path for path in (VIDEO_ROOT / group).iterdir() if path.is_dir())
    cell_width, frame_height, label_height = 300, 169, 31
    videos_per_sku, samples_per_video = 3, 2
    columns = videos_per_sku * samples_per_video
    sheet = np.full(
        (len(sku_dirs) * (frame_height + label_height), columns * cell_width, 3),
        245,
        dtype=np.uint8,
    )
    reports = []

    for row, sku_dir in enumerate(sku_dirs):
        train_videos = sorted((sku_dir / "train").glob("*.mp4"))
        misplaced_videos = sorted(sku_dir.glob("*.mp4"))
        videos = train_videos + misplaced_videos
        sku_report = {
            "sku": sku_dir.name,
            "train_video_count": len(train_videos),
            "misplaced_video_count": len(misplaced_videos),
            "videos": [],
        }
        for video_index, video in enumerate(videos[:videos_per_sku]):
            frames, report = read_video(video, [0.30, 0.70])
            sku_report["videos"].append(report)
            for sample_index, (position, frame) in enumerate(frames):
                column = video_index * samples_per_video + sample_index
                x0 = column * cell_width
                y0 = row * (frame_height + label_height)
                cell = np.full((frame_height + label_height, cell_width, 3), 245, dtype=np.uint8)
                cell[:frame_height] = letterbox(frame, cell_width, frame_height)
                location = "ROOT" if video.parent == sku_dir else "train"
                label = f"{sku_dir.name.split('_')[1]} {video.stem[-6:]} {int(position * 100)}% {location}"
                cv2.putText(
                    cell,
                    label,
                    (6, frame_height + 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (20, 20, 20),
                    1,
                    cv2.LINE_AA,
                )
                sheet[y0 : y0 + cell.shape[0], x0 : x0 + cell.shape[1]] = cell
        reports.append(sku_report)

    save_jpeg(OUTPUT / f"{group}_contact_sheet.jpg", sheet)
    return reports


def build_yolo_sheet():
    entries = []
    for split in ["train", "val", "test"]:
        for video in sorted((VIDEO_ROOT / "YOLO Data" / split).glob("*.mp4")):
            entries.append((split, video))

    cell_width, frame_height, label_height = 360, 203, 32
    positions = [0.15, 0.40, 0.65, 0.90]
    sheet = np.full(
        (len(entries) * (frame_height + label_height), len(positions) * cell_width, 3),
        245,
        dtype=np.uint8,
    )
    reports = []
    for row, (split, video) in enumerate(entries):
        frames, report = read_video(video, positions)
        report["split"] = split
        reports.append(report)
        for column, (position, frame) in enumerate(frames):
            x0 = column * cell_width
            y0 = row * (frame_height + label_height)
            cell = np.full((frame_height + label_height, cell_width, 3), 245, dtype=np.uint8)
            cell[:frame_height] = letterbox(frame, cell_width, frame_height)
            label = f"{split} {video.stem[-6:]} {int(position * 100)}%"
            cv2.putText(
                cell,
                label,
                (7, frame_height + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            sheet[y0 : y0 + cell.shape[0], x0 : x0 + cell.shape[1]] = cell

    save_jpeg(OUTPUT / "YOLO_contact_sheet.jpg", sheet)
    return reports


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {group: build_classification_sheet(group) for group in CLASS_GROUPS}
    report["YOLO"] = build_yolo_sheet()
    (OUTPUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"检查报告：{OUTPUT / 'report.json'}")


if __name__ == "__main__":
    main()
