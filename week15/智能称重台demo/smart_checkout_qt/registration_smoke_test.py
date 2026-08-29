"""在线注册回归：阿萨姆竖放/横放均立即识别，并且同帧重复框只计一件。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
from PySide6.QtCore import QCoreApplication

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference_controller import InferenceController  # noqa: E402


def main() -> int:
    app = QCoreApplication.instance() or QCoreApplication(sys.argv)
    del app
    video = PROJECT_ROOT / "video" / "asm milktea.mp4"
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开测试视频：{video}")

    controller = InferenceController()
    evidence = []
    try:
        for frame_index in (0, 1200, 1900):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"无法读取测试帧：{frame_index}")
            _, detections, cart = controller.process_frame(frame)
            assert len(detections) == 1, f"第{frame_index}帧动态SKU未去重：{len(detections)}框"
            assert detections[0]["model_class"] == "BOTTLE_07_asm milktea"
            assert detections[0]["found"] is True
            assert cart["total_quantity"] == 1
            assert cart["total_amount"] == 3.0
            evidence.append(
                {
                    "frame": frame_index,
                    "detected_group": detections[0]["detected_package_type"],
                    "matched_group": detections[0]["package_type"],
                    "similarity": round(detections[0]["similarity"], 4),
                    "margin": round(detections[0]["margin"], 4),
                    "quantity": cart["total_quantity"],
                    "amount": cart["total_amount"],
                }
            )
    finally:
        capture.release()

    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    print("[PASS] 在线注册商品在竖放、横放和YOLO跨组场景下均立即识别为1件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
