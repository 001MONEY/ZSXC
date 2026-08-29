r"""Qt 界面冒烟测试：验证窗口创建 + ONNX 推理链路跑一帧（非交互）。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from inference_worker import InferenceWorker  # noqa: E402
from main import MainWindow  # noqa: E402

VIDEO = PROJECT_ROOT / "video" / "YOLO Data" / "val" / "VID_20260826_110333.mp4"


def smoke() -> int:
    app = QApplication(sys.argv)

    # 1) 窗口创建
    window = MainWindow()
    window.show()
    print("[OK] 主窗口创建成功：", window.windowTitle())

    # 2) 推理链路：手动打开视频，跑一帧
    worker = InferenceWorker()
    cap = cv2.VideoCapture(str(VIDEO))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("[FAIL] 无法读取测试视频帧")
        return 1

    yolo, library, dao = None, None, None
    from onnx_engine import OnnxFeatureLibrary, YoloOnnxDetector
    from database.goods_dao import GoodsDao
    from pipeline_demo import load_font

    yolo = YoloOnnxDetector(worker._cap if False else (PROJECT_ROOT / "runs" / "onnx" / "yolov8n_det.onnx"))
    library = OnnxFeatureLibrary()
    dao = GoodsDao()
    font = load_font(22)
    height, width = frame.shape[:2]
    annotated, detections, cart = worker._process_frame(frame, width, height, yolo, library, dao, font)
    print(f"[OK] 单帧推理完成：{len(detections)}个目标")
    for det in detections:
        if det["found"]:
            print(f"      {det['label']} -> {det['name']} ¥{det['price']:.2f} (sim={det['class_conf']:.3f}, margin={det['margin']:.3f})")
        else:
            print(f"      {det['label']} -> 未注册({det['unknown_reason']})")
    print(f"[OK] 购物车：{cart['total_quantity']}件 ¥{cart['total_amount']:.2f}")
    print(f"[OK] 标注帧形状：{annotated.shape}")

    dao.close()
    QTimer.singleShot(500, app.quit)
    app.exec()
    print("[OK] 界面事件循环正常退出")
    return 0


if __name__ == "__main__":
    sys.exit(smoke())
