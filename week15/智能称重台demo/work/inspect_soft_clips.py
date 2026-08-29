from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"D:\project\step1\week15\智能称重台demo")
OUT = ROOT / "work" / "video_inspection"
VIDEOS = [
    ROOT / "BOTTLE_01_greentea" / "train" / "VID_20260825_225638.mp4",
    ROOT / "BOTTLE_03_yogurt" / "train" / "VID_20260825_225110.mp4",
]


def letterbox(frame, width=320, height=180):
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


for video in VIDEOS:
    cap = cv2.VideoCapture(str(video))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sheet = np.full((4 * 205, 5 * 320, 3), 245, dtype=np.uint8)
    for i, pos in enumerate(np.linspace(0.025, 0.975, 20)):
        frame_index = min(int(count * pos), count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        cell = np.full((205, 320, 3), 245, dtype=np.uint8)
        cell[:180] = letterbox(frame)
        label = f"{frame_index / fps:04.1f}s  sharp={sharp:04.1f}"
        cv2.putText(cell, label, (6, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        row, col = divmod(i, 5)
        sheet[row * 205:(row + 1) * 205, col * 320:(col + 1) * 320] = cell
    cap.release()
    ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError(video)
    (OUT / f"{video.stem}_detail.jpg").write_bytes(encoded.tobytes())
