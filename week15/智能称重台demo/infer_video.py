r"""使用训练好的YOLOv8检测模型对视频或图片目录跑一遍推理。

默认命令：

    D:\project\step1\env\python.exe infer_video.py

输入可以是单个视频文件，也可以是包含图片的目录。
默认使用训练产出的 best.pt，标注四类商品（bag/bottle/box/cylinder）
并保存标注结果与逐帧统计CSV。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"
DEFAULT_MODEL = PROJECT_ROOT / "runs" / "detect" / "smart_checkout_yolov8n" / "weights" / "best.pt"
DEFAULT_SOURCE = PROJECT_ROOT / "video" / "YOLO Data" / "test" / "VID_20260826_110333.mp4"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "infer"
CLASS_NAMES = {0: "bag", 1: "bottle", 2: "box", 3: "cylinder"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用训练好的YOLOv8模型推理视频。")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="训练好的YOLO权重，默认best.pt。")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="待推理的视频文件。")
    parser.add_argument("--output", type=Path, default=DEFAULT_PROJECT, help="结果输出根目录。")
    parser.add_argument("--name", default="video_infer", help="本次推理输出目录名。")
    parser.add_argument("--conf", type=float, default=0.25, help="最低检测置信度，默认0.25。")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS交并比阈值，默认0.45。")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸，默认640。")
    parser.add_argument("--max-det", type=int, default=30, help="单帧最多保留的检测框数，默认30。")
    parser.add_argument("--device", default="0", help="推理设备，例如0或cpu，默认第0张GPU。")
    parser.add_argument("--vid-stride", type=int, default=1, help="视频抽帧步长，1表示逐帧处理。")
    parser.add_argument("--exist-ok", action="store_true", help="允许复用同名输出目录。")
    return parser.parse_args()


def load_model(model_file: Path, device: str):
    """加载项目内置的Ultralytics源码与训练权重。"""
    if not THIRD_PARTY_ROOT.is_dir():
        raise FileNotFoundError(f"项目内Ultralytics源码不存在：{THIRD_PARTY_ROOT}")
    sys.path.insert(0, str(THIRD_PARTY_ROOT))

    import torch
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != "8.4.113":
        raise RuntimeError(f"Ultralytics版本不正确：{ultralytics.__version__}，预期8.4.113")
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("指定了GPU推理，但当前PyTorch无法使用CUDA。可添加 --device cpu 改用CPU。")
    if not model_file.is_file():
        raise FileNotFoundError(f"模型权重不存在：{model_file}")

    print(f"Python：{sys.version.split()[0]}")
    print(f"PyTorch：{torch.__version__}")
    print(f"Ultralytics：{ultralytics.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA：{torch.version.cuda}，GPU：{torch.cuda.get_device_name(0)}")
    else:
        print("推理设备：CPU")
    print(f"模型权重：{model_file}")
    return YOLO(str(model_file))


def write_frame_stats(csv_path: Path, results) -> None:
    """把逐帧检测结果写入CSV，便于后续统计。"""
    rows: list[dict[str, object]] = []
    for frame_index, result in enumerate(results):
        for box in result.boxes:
            if box.cls is None:
                continue
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            rows.append(
                {
                    "frame": frame_index,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES.get(class_id, "unknown"),
                    "conf": round(confidence, 4),
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                }
            )

    if rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
            file.write("frame,class_id,class_name,conf,x1,y1,x2,y2\n")


def main() -> None:
    args = parse_args()
    args.model = args.model.resolve()
    args.source = args.source.resolve()
    args.output = args.output.resolve()

    if not args.source.exists():
        raise FileNotFoundError(f"待推理的输入不存在：{args.source}")

    # 将Ultralytics运行配置放在项目目录内，避免污染用户全局配置。
    config_dir = PROJECT_ROOT / "work" / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    model = load_model(args.model, args.device)

    # 区分输入类型：视频文件或图片目录。
    if args.source.is_dir():
        image_files = sorted(
            path for path in args.source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_files:
            raise RuntimeError(f"目录中没有图片：{args.source}")
        print(f"输入目录：{args.source}")
        print(f"图片数量：{len(image_files)}")
    else:
        capture = cv2.VideoCapture(str(args.source))
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频：{args.source}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        print(f"输入视频：{args.source}")
        print(f"分辨率：{width}x{height}，帧率：{fps:.2f}，总帧数：{total_frames}")

    print(f"开始推理，置信度阈值：{args.conf}，IOU阈值：{args.iou}")
    results = model.predict(
        source=str(args.source),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        vid_stride=args.vid_stride,
        project=str(args.output),
        name=args.name,
        save=True,
        exist_ok=args.exist_ok,
        verbose=True,
    )

    run_dir = Path(results[0].save_dir) if results else args.output / args.name
    print("推理完成。")

    # 统计总体检测结果。
    class_counter: Counter[int] = Counter()
    total_boxes = 0
    for result in results:
        if result.boxes is None:
            continue
        for class_id in result.boxes.cls:
            class_counter[int(class_id.item())] += 1
            total_boxes += 1

    print(f"检测目标总数：{total_boxes}")
    for class_id, name in sorted(CLASS_NAMES.items()):
        print(f"  {name}: {class_counter[class_id]}")

    # 保存逐帧统计CSV。
    csv_path = run_dir / "frame_stats.csv"
    write_frame_stats(csv_path, results)
    print(f"逐帧统计：{csv_path}")
    print(f"标注视频：{run_dir}")


if __name__ == "__main__":
    main()
