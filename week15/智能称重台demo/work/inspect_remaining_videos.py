from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"D:\project\step1\week15\智能称重台demo")
OUT = ROOT / "work" / "video_inspection"
SKUS = [
    "BOTTLE_01_greentea",
    "BOTTLE_02_orange juice",
    "BOTTLE_03_yogurt",
    "BOTTLE_04_mizone",
]

CELL_W = 360
CELL_H = 220
FRAME_H = 190
SAMPLE_POSITIONS = [0.10, 0.35, 0.60, 0.85]


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    x = (width - nw) // 2
    y = (height - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def inspect_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, {"path": str(path), "error": "open_failed"}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0

    frames = []
    metrics = []
    for position in SAMPLE_POSITIONS:
        frame_index = min(max(0, int(frame_count * position)), max(0, frame_count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        frames.append((position, frame))
        metrics.append(
            {
                "position": position,
                "frame_index": frame_index,
                "brightness": round(brightness, 2),
                "sharpness": round(sharpness, 2),
            }
        )
    cap.release()

    report = {
        "path": str(path.relative_to(ROOT)),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "duration_sec": round(duration, 2),
        "samples": metrics,
    }
    return frames, report


def build_sheet(sku: str, videos: list[Path]):
    sheet = np.full((len(videos) * CELL_H, len(SAMPLE_POSITIONS) * CELL_W, 3), 245, dtype=np.uint8)
    reports = []

    for row, video in enumerate(videos):
        frames, report = inspect_video(video)
        reports.append(report)
        y0 = row * CELL_H
        for col in range(len(SAMPLE_POSITIONS)):
            x0 = col * CELL_W
            cell = np.full((CELL_H, CELL_W, 3), 245, dtype=np.uint8)
            if frames and col < len(frames):
                pos, frame = frames[col]
                cell[:FRAME_H] = letterbox(frame, CELL_W, FRAME_H)
                label = f"{video.stem}  {int(pos * 100)}%"
            else:
                label = f"{video.stem}  missing"
            cv2.putText(cell, label, (8, 211), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
            sheet[y0 : y0 + CELL_H, x0 : x0 + CELL_W] = cell

    ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError(f"Failed to encode contact sheet for {sku}")
    (OUT / f"{sku}_contact_sheet.jpg").write_bytes(encoded.tobytes())
    return reports


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_reports = {}
    for sku in SKUS:
        videos = sorted((ROOT / sku / "train").glob("*.mp4"))
        all_reports[sku] = build_sheet(sku, videos)
    (OUT / "report.json").write_text(json.dumps(all_reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(all_reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
