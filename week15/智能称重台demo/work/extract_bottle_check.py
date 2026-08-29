"""提取测试视频中星巴克相关帧的瓶子裁切图，人工确认那个瓶子到底是什么。"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_demo import CLASS_NAMES, expand_box, load_yolo, YOLO_MODEL

VIDEO = Path(r"video\YOLO Data\val\VID_20260826_110333.mp4")
TARGET_FRAMES = [47, 377, 378, 900, 1000]  # 星巴克top2出现的帧 + 中段帧
OUT_DIR = Path(r"work\starbucks_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)

yolo = load_yolo(YOLO_MODEL, "0")
capture = cv2.VideoCapture(str(VIDEO))

for frame_index in TARGET_FRAMES:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        continue
    height, width = frame.shape[:2]
    results = yolo.predict(source=frame, conf=0.25, imgsz=640, device="0", verbose=False)[0]
    # 保存整帧
    cv2.imwrite(str(OUT_DIR / f"frame_{frame_index}_full.jpg"), frame)
    if results.boxes is not None:
        for i, box in enumerate(results.boxes):
            yolo_class = int(box.cls.item())
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            left, top, right, bottom = expand_box(box.xyxy[0], width, height, 0.05)
            crop = frame[top:bottom, left:right]
            if crop.size == 0:
                continue
            name = f"frame_{frame_index}_box{i}_{CLASS_NAMES[yolo_class]}.jpg"
            cv2.imwrite(str(OUT_DIR / name), crop)
            print("saved:", name, f"({x2-x1}x{y2-y1})")
    else:
        print(f"frame {frame_index}: 无检测")
capture.release()
print("输出目录:", OUT_DIR)
