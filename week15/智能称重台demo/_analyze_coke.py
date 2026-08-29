r"""快速分析可口可乐视频：确认未注册识别 + 采集样本裁剪图。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
from onnx_engine import OnnxFeatureLibrary, YoloOnnxDetector, retrieval_match_onnx  # noqa: E402
from pipeline_demo import CLASS_NAMES, expand_box  # noqa: E402
from database.goods_dao import GoodsDao  # noqa: E402

VIDEO = PROJECT_ROOT / "video" / "VID_20260827_140812.mp4"
OUT_DIR = PROJECT_ROOT / "work" / "coke_samples"


def main() -> int:
    yolo = YoloOnnxDetector(PROJECT_ROOT / "runs" / "onnx" / "yolov8n_det.onnx")
    library = OnnxFeatureLibrary()
    dao = GoodsDao()

    cap = cv2.VideoCapture(str(VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频：{total}帧，{width}x{height}")

    step = 12
    frame_idx = 0
    bottle_unknown = 0
    registered = 0
    sample_crops: list[tuple[int, object]] = []
    sims: list[float] = []
    margins: list[float] = []
    top1s: dict[str, int] = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue
        frame_idx += 1
        results = yolo.predict(source=frame, conf=0.25, iou=0.45)[0]
        if results.boxes is None:
            continue
        for box in results.boxes:
            yolo_class = int(box.cls.item())
            group = CLASS_NAMES[yolo_class]
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue
            left, top, right, bottom = expand_box(box.xyxy[0], width, height, 0.05)
            crop = frame[top:bottom, left:right]
            if crop.size == 0:
                continue
            model_class, sim, top2, sim2, margin = retrieval_match_onnx(crop, library, group)
            goods = dao.get_by_model_class(model_class)
            found = goods is not None and not ((sim < 0.80) or (margin < 0.15))
            if found:
                registered += 1
            else:
                bottle_unknown += 1
                top1s[f"{group}->{model_class}"] = top1s.get(f"{group}->{model_class}", 0) + 1
                if len(sample_crops) < 15:
                    sample_crops.append((frame_idx, group, crop.copy()))
                sims.append(sim)
                margins.append(margin)
    cap.release()

    print(f"\n未注册 {bottle_unknown} 帧，已注册 {registered} 帧")
    print(f"未注册 Top1 命中：{top1s}")
    if sims:
        print(f"未注册 sim 范围：{min(sims):.3f} ~ {max(sims):.3f}（阈值 <0.80）")
        print(f"未注册 margin 范围：{min(margins):.3f} ~ {max(margins):.3f}（阈值 <0.15）")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, (fi, group, crop) in enumerate(sample_crops[:10]):
        cv2.imwrite(str(OUT_DIR / f"{group}_f{fi}_{i}.jpg"), crop)
    print(f"\n已保存 {min(len(sample_crops), 10)} 张样本到 {OUT_DIR}")

    if bottle_unknown == 0:
        print("\n[WARN] 未检测到未注册目标——该视频可能未检出或识别为已注册商品")
        return 1
    print("\n[OK] 可口可乐确认为未注册，视频可用于注册测试")
    return 0


if __name__ == "__main__":
    sys.exit(main())
