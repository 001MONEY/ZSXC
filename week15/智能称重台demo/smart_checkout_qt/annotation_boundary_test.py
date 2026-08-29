"""标注越界框回归测试，不加载模型、数据库或摄像头。"""

from __future__ import annotations

import numpy as np

from inference_controller import _annotate_frame, _load_chinese_font


def main() -> int:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    base = {
        "yolo_class": 1,
        "package_type": "bottle",
        "found": False,
        "reason": "类别间隔 0.06 < 0.15",
    }
    detections = [
        {**base, "box": [-30, -50, 100, 130]},
        {**base, "box": [250, -20, 380, 90]},
        {**base, "box": [-40, 180, 90, 280]},
        {**base, "box": [250, 180, 380, 280]},
    ]
    annotated = _annotate_frame(frame, detections, _load_chinese_font())
    if annotated.shape != frame.shape:
        raise AssertionError(f"标注帧尺寸异常：{annotated.shape} != {frame.shape}")
    print("[PASS] 四个方向的越界检测框均可安全绘制")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
