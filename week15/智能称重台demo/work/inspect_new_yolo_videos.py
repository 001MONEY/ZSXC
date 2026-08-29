"""检查未记录在抽帧清单中的新增YOLO训练视频。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path(r"D:\project\step1\week15\智能称重台demo")
VIDEO_DIR = PROJECT / "video" / "YOLO Data" / "train"
MANIFEST = PROJECT / "yolo_dataset_raw" / "extraction_manifest.csv"
OUTPUT = PROJECT / "work" / "new_yolo_inspection"
POSITIONS = [0.15, 0.40, 0.65, 0.90]


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


processed = set()
if MANIFEST.exists():
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        processed = {row["video"] for row in csv.DictReader(file) if row["split"] == "train"}

videos = [path for path in sorted(VIDEO_DIR.glob("*.mp4")) if path.name not in processed]
OUTPUT.mkdir(parents=True, exist_ok=True)
cell_w, frame_h, label_h = 360, 203, 32
sheet = np.full((len(videos) * (frame_h + label_h), len(POSITIONS) * cell_w, 3), 245, dtype=np.uint8)
reports = []

for row, video in enumerate(videos):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        reports.append({"video": video.name, "error": "open_failed"})
        continue
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    report = {
        "video": video.name,
        "duration_sec": round(count / fps, 2),
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "samples": [],
    }
    for col, position in enumerate(POSITIONS):
        index = min(round((count - 1) * position), count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        report["samples"].append(
            {
                "position": position,
                "brightness": round(float(gray.mean()), 2),
                "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
            }
        )
        cell = np.full((frame_h + label_h, cell_w, 3), 245, dtype=np.uint8)
        cell[:frame_h] = letterbox(frame, cell_w, frame_h)
        label = f"{video.stem[-6:]} {int(position * 100)}%"
        cv2.putText(cell, label, (7, frame_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        y0 = row * (frame_h + label_h)
        x0 = col * cell_w
        sheet[y0 : y0 + cell.shape[0], x0 : x0 + cell.shape[1]] = cell
    cap.release()
    reports.append(report)

ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
if not ok:
    raise RuntimeError("拼图编码失败")
(OUTPUT / "new_yolo_contact_sheet.jpg").write_bytes(encoded.tobytes())
(OUTPUT / "report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(reports, ensure_ascii=False, indent=2))
