r"""PT 与 ONNX 推理一致性验证（导出的前提：阈值是 PT 特征标定的）。

验证内容：
  1) 特征一致性：对 4 个大类的 val 图，分别用 PT 模型与 ONNX 模型提取 512 维特征，
     计算余弦相似度（要求平均 ≥ 0.999，最低 ≥ 0.99）。
  2) 检测一致性：对默认测试视频抽帧，对比 PT YOLO 与 ONNX YOLO 的检测框
     （IoU、类别一致率、置信度差）。

用法：
  D:\project\step1\env\python.exe verify_onnx.py            # 特征 + 检测
  D:\project\step1\env\python.exe verify_onnx.py --feat-only
  D:\project\step1\env\python.exe verify_onnx.py --det-only --video <路径>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from onnx_engine import OnnxFeatureLibrary, YoloOnnxDetector  # noqa: E402
from pipeline_demo import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_SOURCE,
    load_feature_library,
    load_yolo,
)

GROUPS = ("bag", "bottle", "box", "cylinder")
DATA_ROOT = PROJECT_ROOT / "classification_dataset_from_videos"
FEAT_MIN_SIM = 0.99     # 允许的最低余弦相似度
FEAT_AVG_SIM = 0.999    # 要求的平均余弦相似度
FEAT_SAMPLES = 50       # 每类验证图片数
DET_FRAMES = 20         # 检测验证抽帧数
DET_IOU = 0.5           # 检测框匹配 IoU 阈值
DET_CONF_DELTA = 0.05   # 置信度最大允许差


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PT vs ONNX 一致性验证。")
    parser.add_argument("--feat-only", action="store_true", help="只验证特征一致性。")
    parser.add_argument("--det-only", action="store_true", help="只验证检测一致性。")
    parser.add_argument("--video", type=Path, default=DEFAULT_SOURCE, help="检测验证用视频。")
    parser.add_argument("--device", default="0", help="PT 推理设备：0/cpu。")
    return parser.parse_args()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def verify_features(device: str) -> bool:
    """特征一致性：PT vs ONNX 余弦相似度。"""
    print("\n========== 1) 特征一致性验证 ==========")
    import torch

    pt_lib = load_feature_library(torch.device(f"cuda:{device}" if device.isdigit() else "cpu"))
    onnx_lib = OnnxFeatureLibrary()

    all_sims: list[float] = []
    ok = True
    for group in GROUPS:
        images: list[Path] = []
        for sku_dir in sorted((DATA_ROOT / group / "val").iterdir()):
            if sku_dir.is_dir():
                images.extend(sorted(sku_dir.glob("*.jpg"))[: max(1, FEAT_SAMPLES // 6)])
        images = images[:FEAT_SAMPLES]
        if not images:
            print(f"  {group}: 无验证图片，跳过")
            continue

        sims: list[float] = []
        for path in images:
            from PIL import Image

            img = Image.open(path).convert("RGB")
            model_device = next(pt_lib[group]["model"].parameters()).device
            pt_tensor = pt_lib[group]["transform"](img).unsqueeze(0).to(model_device)
            with torch.inference_mode():
                pt_feat = pt_lib[group]["model"](pt_tensor).squeeze(0).cpu().numpy()
            pt_feat = pt_feat / (np.linalg.norm(pt_feat) + 1e-12)

            crop_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            onnx_feat = onnx_lib.extract_feature(group, crop_bgr)
            sims.append(cosine(pt_feat, onnx_feat))

        arr = np.array(sims)
        min_sim, avg_sim = float(arr.min()), float(arr.mean())
        flag = "[OK]" if (avg_sim >= FEAT_AVG_SIM and min_sim >= FEAT_MIN_SIM) else "[FAIL]"
        print(f"  {flag} {group}: 平均={avg_sim:.6f}  最低={min_sim:.6f}  (n={len(sims)})")
        all_sims.extend(sims)
        if not (avg_sim >= FEAT_AVG_SIM and min_sim >= FEAT_MIN_SIM):
            ok = False

    if all_sims:
        print(f"  总体：平均={np.mean(all_sims):.6f}  最低={np.min(all_sims):.6f}")
    return ok


def verify_detection(device: str, video: Path) -> bool:
    """检测一致性：PT vs ONNX 检测框对比。"""
    print("\n========== 2) 检测一致性验证 ==========")
    import torch

    pt_yolo = load_yolo(PROJECT_ROOT / "runs" / "detect" / "smart_checkout_yolov8n" / "weights" / "best.pt", device)
    onnx_yolo = YoloOnnxDetector(PROJECT_ROOT / "runs" / "onnx" / "yolov8n_det.onnx")
    torch_device = torch.device(f"cuda:{device}" if device.isdigit() else "cpu")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        print(f"  无法打开视频：{video}")
        return False

    frame_idx = 0
    checked = 0
    total_iou, total_conf_delta, matched = 0.0, 0.0, 0
    class_agree = 0
    ok = True
    while checked < DET_FRAMES:
        ret, frame = capture.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 10 != 1:  # 抽帧
            continue

        pt_res = pt_yolo.predict(source=frame, conf=0.25, iou=0.45, imgsz=640, device=torch_device, verbose=False)[0]
        onnx_res = onnx_yolo.predict(source=frame, conf=0.25, iou=0.45)[0]

        pt_boxes = []
        if pt_res.boxes is not None:
            for box in pt_res.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                pt_boxes.append((np.array([x1, y1, x2, y2], dtype=float), int(box.cls.item()), float(box.conf.item())))
        onnx_boxes = []
        if onnx_res.boxes is not None:
            for box in onnx_res.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                onnx_boxes.append((np.array([x1, y1, x2, y2], dtype=float), int(box.cls.item()), float(box.conf.item())))

        if len(pt_boxes) != len(onnx_boxes):
            print(f"  [WARN] 帧{frame_idx}: 检测数量不一致 PT={len(pt_boxes)} ONNX={len(onnx_boxes)}")
            ok = False
        checked += 1

        # 贪心匹配 ONNX 框到 PT 框（IoU 最大）
        used = set()
        for ob in onnx_boxes:
            best_iou, best_idx = 0.0, -1
            for i, pb in enumerate(pt_boxes):
                if i in used:
                    continue
                iov = box_iou(ob[0], pb[0])
                if iov > best_iou:
                    best_iou, best_idx = iov, i
            if best_idx >= 0 and best_iou >= DET_IOU:
                used.add(best_idx)
                matched += 1
                total_iou += best_iou
                total_conf_delta += abs(ob[2] - pt_boxes[best_idx][2])
                if ob[1] == pt_boxes[best_idx][1]:
                    class_agree += 1
                else:
                    print(f"  [FAIL] 帧{frame_idx}: 类别不一致 ONNX={CLASS_NAMES[ob[1]]} PT={CLASS_NAMES[pt_boxes[best_idx][1]]}")
                    ok = False
            else:
                print(f"  [FAIL] 帧{frame_idx}: ONNX 框未匹配到 PT 框 IoU={best_iou:.2f}")
                ok = False
        for pb in pt_boxes:
            if all(box_iou(pb[0], ob[0]) < DET_IOU for ob in onnx_boxes):
                print(f"  [FAIL] 帧{frame_idx}: PT 框未匹配到 ONNX 框")
                ok = False
                break

    capture.release()
    if matched:
        print(f"  匹配 {matched} 个检测框，平均IoU={total_iou/matched:.4f}，类别一致率={class_agree/matched*100:.1f}%")
        print(f"  平均置信度差={total_conf_delta/matched:.4f}（阈值 {DET_CONF_DELTA}）")
    return ok


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (union + 1e-9)


def main() -> None:
    args = parse_args()
    results: list[bool] = []
    if not args.det_only:
        results.append(verify_features(args.device))
    if not args.feat_only:
        results.append(verify_detection(args.device, args.video))

    print("\n========== 结论 ==========")
    if all(results):
        print("[OK] 一致性验证全部通过：ONNX 可安全替代 PT 推理（阈值不变）")
    else:
        print("[FAIL] 存在不一致，请检查 ONNX 导出/推理代码")
        sys.exit(1)


if __name__ == "__main__":
    main()
