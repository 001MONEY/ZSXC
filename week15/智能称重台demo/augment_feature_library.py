r"""用补充视频扩充特征库（解决真实场景域差异，无需重训模型）。

适用场景：某SKU在真实场景识别不出来（如星巴克），
用户补拍该SKU的真实场景视频后，把裁切特征追加进对应特征库。

用法：
    D:\project\step1\env\python.exe augment_feature_library.py --video video/starbucks_supplement/星巴克_1.mp4 --group bottle --sku BOTTLE_06_starbucks

    --video 可多次指定多个视频；--group 包装类型；--sku 目标SKU名。
    可选 --frame-step 抽帧间隔（默认15帧取1帧），--verify 用分类模型过滤非目标样本。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_demo import (  # noqa: E402
    CLASS_NAMES,
    CLASSIFY_DIR,
    expand_box,
    load_classifiers,
    load_feature_library,
    load_yolo,
    YOLO_MODEL,
)
from train_resnet_classifier import EnsureRGB, SquarePad, build_model  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_DIR = PROJECT_ROOT / "runs" / "features"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用补充视频扩充特征库。")
    parser.add_argument("--video", action="append", required=True, help="补充视频路径（可多次指定）。")
    parser.add_argument("--group", choices=["bag", "bottle", "box", "cylinder"], required=True)
    parser.add_argument("--sku", required=True, help="目标SKU名，如 BOTTLE_06_starbucks")
    parser.add_argument("--frame-step", type=int, default=15, help="抽帧间隔，默认15帧取1帧。")
    parser.add_argument("--verify", action="store_true", help="用分类模型过滤非目标样本。")
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO置信度阈值（特写视频可提高）。")
    parser.add_argument("--device", default="auto", help="设备：auto/0/cpu。")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{normalized}" if normalized.isdigit() else normalized)


@torch.inference_mode()
def extract_query_feature(model, crop_bgr, transform, device):
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    tensor = transform(pil).unsqueeze(0).to(device, non_blocking=True)
    feature = model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
    norm = float(np.linalg.norm(feature))
    return feature / max(norm, 1e-12)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    # Ultralytics 不接受 auto，需转换为 '0' 或 'cpu'。
    yolo_device = f"{device.index}" if device.type == "cuda" else "cpu"
    print(f"设备：{device}，目标：{args.group} / {args.sku}")

    yolo = load_yolo(YOLO_MODEL, yolo_device)
    feature_library = load_feature_library(device)
    lib = feature_library[args.group]

    # 验证目标SKU存在于特征库。
    if args.sku not in lib["classes"]:
        raise ValueError(f"特征库{args.group}中不存在SKU：{args.sku}")

    # 验证分类模型（可选）。
    verify_model = None
    if args.verify:
        classifier = load_classifiers(device)[args.group]
        verify_model = classifier

    new_features: list[np.ndarray] = []
    for video_path in map(Path, args.video):
        if not video_path.is_file():
            raise FileNotFoundError(f"视频不存在：{video_path}")
        capture = cv2.VideoCapture(str(video_path))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"处理视频：{video_path}（{total}帧）")
        frame_index = 0
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % args.frame_step != 0:
                frame_index += 1
                continue
            height, width = frame.shape[:2]
            results = yolo.predict(
                source=frame, conf=args.conf, imgsz=640, device=yolo_device, verbose=False
            )[0]
            if results.boxes is not None:
                for box in results.boxes:
                    yolo_class = int(box.cls.item())
                    if CLASS_NAMES[yolo_class] != args.group:
                        continue
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    if x2 - x1 < 30 or y2 - y1 < 30:
                        continue
                    left, top, right, bottom = expand_box(box.xyxy[0], width, height, 0.05)
                    crop = frame[top:bottom, left:right]
                    if crop.size == 0:
                        continue
                    # 可选：用分类模型确认是目标SKU。
                    if verify_model is not None:
                        from pipeline_demo import classify_crop

                        predicted, confid = classify_crop(crop, verify_model, device)
                        if predicted != args.sku or confid < 0.5:
                            continue
                    feature = extract_query_feature(lib["model"], crop, lib["transform"], device)
                    new_features.append(feature)
                    count += 1
            frame_index += 1
        capture.release()
        print(f"  提取 {count} 个特征")

    if not new_features:
        raise RuntimeError("没有提取到任何特征，请检查视频内容或参数。")

    new_matrix = np.stack(new_features)
    # 追加进特征库。
    old_embeddings = lib["embeddings"]
    old_labels = lib["labels"]
    combined_embeddings = np.vstack([old_embeddings, new_matrix])
    combined_labels = old_labels + [args.sku] * len(new_features)

    # 重新计算类中心（每个SKU均值后L2归一化）。
    classes = sorted(set(combined_labels))
    class_to_idx = {name: index for index, name in enumerate(classes)}
    centers = np.zeros((len(classes), lib["embeddings"].shape[1]), dtype=np.float32)
    for name in classes:
        indices = [i for i, label in enumerate(combined_labels) if label == name]
        center = combined_embeddings[indices].mean(axis=0)
        norm = float(np.linalg.norm(center))
        centers[class_to_idx[name]] = center / max(norm, 1e-12)

    # 保存（覆盖对应group文件）。
    np.save(FEATURES_DIR / f"{args.group}_embeddings.npy", combined_embeddings)
    (FEATURES_DIR / f"{args.group}_labels.json").write_text(
        json.dumps(combined_labels, ensure_ascii=False), encoding="utf-8"
    )
    np.save(FEATURES_DIR / f"{args.group}_centers.npy", centers)
    (FEATURES_DIR / f"{args.group}_classes.json").write_text(
        json.dumps(classes, ensure_ascii=False), encoding="utf-8"
    )
    # 更新元数据（记录补充）。
    metadata_path = FEATURES_DIR / f"{args.group}_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["library_version"] = int(metadata.get("library_version", 1)) + 1
    metadata["augmented_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata["augment_sources"] = [str(v) for v in args.video]
    metadata["samples"] = len(combined_labels)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n特征库已扩充：{args.group} 从 {len(old_labels)} 个特征 -> {len(combined_labels)} 个特征")
    print(f"  {args.sku} 新增 {len(new_features)} 个真实场景样本")
    print("请重新运行 pipeline 验证识别效果。")


if __name__ == "__main__":
    main()
