"""生成YOLO训练图片总览页，并计算相邻图片相似度供人工复核。"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path(r"D:\project\step1\week15\智能称重台demo")
IMAGE_DIR = PROJECT / "yolo_dataset_raw" / "images" / "train"
OUTPUT = PROJECT / "work" / "yolo_duplicate_review"
PAGE_SIZE = 48
COLS = 8
CELL_W = 240
FRAME_H = 135
LABEL_H = 25


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片：{path}")
    return image


def thumbnail(image: np.ndarray, width: int, height: int) -> np.ndarray:
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized = cv2.resize(
        image,
        (round(source_w * scale), round(source_h * scale)),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def compare(previous: np.ndarray, current: np.ndarray):
    prev_small = cv2.resize(previous, (320, 180), interpolation=cv2.INTER_AREA)
    curr_small = cv2.resize(current, (320, 180), interpolation=cv2.INTER_AREA)
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
    absolute = cv2.absdiff(prev_gray, curr_gray)
    mean_difference = float(absolute.mean())
    changed_ratio = float((absolute > 20).mean())
    return mean_difference, changed_ratio


OUTPUT.mkdir(parents=True, exist_ok=True)
images = sorted(IMAGE_DIR.glob("*.jpg"))
rows = []
previous = None

for index, path in enumerate(images):
    image = read_image(path)
    if previous is None:
        mean_difference, changed_ratio = 999.0, 1.0
    else:
        mean_difference, changed_ratio = compare(previous, image)
    rows.append(
        {
            "image": path.name,
            "mean_difference_from_previous": round(mean_difference, 4),
            "changed_pixel_ratio": round(changed_ratio, 6),
        }
    )
    previous = image

for page_index, start in enumerate(range(0, len(images), PAGE_SIZE), start=1):
    page_paths = images[start : start + PAGE_SIZE]
    page_rows = (len(page_paths) + COLS - 1) // COLS
    sheet = np.full((page_rows * (FRAME_H + LABEL_H), COLS * CELL_W, 3), 245, dtype=np.uint8)
    for local_index, path in enumerate(page_paths):
        image = read_image(path)
        row, col = divmod(local_index, COLS)
        cell = np.full((FRAME_H + LABEL_H, CELL_W, 3), 245, dtype=np.uint8)
        cell[:FRAME_H] = thumbnail(image, CELL_W, FRAME_H)
        metric = rows[start + local_index]
        label = f"{path.stem[-4:]} d={metric['mean_difference_from_previous']:.1f} c={metric['changed_pixel_ratio']:.3f}"
        cv2.putText(cell, label, (5, FRAME_H + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        y0 = row * (FRAME_H + LABEL_H)
        x0 = col * CELL_W
        sheet[y0 : y0 + cell.shape[0], x0 : x0 + cell.shape[1]] = cell
    ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 93])
    if not ok:
        raise RuntimeError("总览页编码失败")
    (OUTPUT / f"page_{page_index:02d}.jpg").write_bytes(encoded.tobytes())

with (OUTPUT / "adjacent_similarity.csv").open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"图片数量：{len(images)}")
print(f"总览页数：{(len(images) + PAGE_SIZE - 1) // PAGE_SIZE}")
