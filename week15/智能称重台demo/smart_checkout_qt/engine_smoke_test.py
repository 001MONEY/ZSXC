"""读取测试视频首帧，验证 ONNX、特征库和 MySQL 链路；不会打开摄像头。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
from PySide6.QtCore import QCoreApplication

from inference_controller import InferenceController, PROJECT_ROOT


TEST_VIDEO = PROJECT_ROOT / "video" / "YOLO Data" / "val" / "VID_20260826_110333.mp4"
TEST_FRAME = 49


def main() -> int:
    QCoreApplication.instance() or QCoreApplication(sys.argv)
    if not TEST_VIDEO.is_file():
        print(f"[FAIL] 测试视频不存在：{TEST_VIDEO}")
        return 1
    capture = cv2.VideoCapture(str(TEST_VIDEO))
    capture.set(cv2.CAP_PROP_POS_FRAMES, TEST_FRAME)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        print(f"[FAIL] 无法读取测试视频：{TEST_VIDEO}")
        return 1

    controller = InferenceController()
    annotated, detections, cart = controller.process_frame(frame)
    providers = controller._detector.session.get_providers()  # 测试运行时提供器
    if annotated.shape != frame.shape:
        print(f"[FAIL] 标注帧尺寸异常：{annotated.shape} != {frame.shape}")
        return 1
    print(f"[OK] ONNX 提供器：{providers}")
    if not detections:
        print(f"[FAIL] 第 {TEST_FRAME} 帧未检出目标")
        return 1
    if not any(item["found"] for item in detections):
        print(f"[FAIL] 第 {TEST_FRAME} 帧没有通过开放集阈值的已注册商品")
        return 1
    if cart["total_quantity"] < 1:
        print("[FAIL] 商品检索成功但未进入稳定购物车")
        return 1
    print(f"[OK] 第 {TEST_FRAME} 帧推理：{len(detections)} 个目标")
    print(f"[OK] 稳定购物车：{cart['total_quantity']} 件，CNY {cart['total_amount']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
