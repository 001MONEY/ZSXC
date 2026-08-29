r"""用训练好的YOLOv8模型从24个SKU原始视频生成分类数据集。

默认处理 video/BAG、BOTTLE、BOX、CYLINDER 下全部train视频，输出结构：

    classification_dataset_from_videos/
    ├── bag/
    │   ├── train/BAG_01_kebike_chips/
    │   ├── val/
    │   └── test/
    ├── bottle/
    ├── box/
    ├── cylinder/
    ├── pipeline_manifest_train.csv
    └── pipeline_summary_train.csv

运行命令：

    D:\project\step1\env\python.exe build_classification_dataset_from_videos.py

流水线只用YOLO定位商品，SKU标签始终继承视频所在的SKU目录。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"
DEFAULT_INPUT = PROJECT_ROOT / "video"
DEFAULT_OUTPUT = PROJECT_ROOT / "classification_dataset_from_videos"
DEFAULT_MODEL = PROJECT_ROOT / "runs" / "detect" / "smart_checkout_yolov8n" / "weights" / "best.pt"
GROUPS = ("BAG", "BOTTLE", "BOX", "CYLINDER")
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
PREFIX_TO_CLASS = {
    "BAG_": (0, "bag"),
    "BOTTLE_": (1, "bottle"),
    "BOX_": (2, "box"),
    "CYLINDER_": (3, "cylinder"),
}


@dataclass(frozen=True)
class VideoJob:
    group: str
    sku: str
    path: Path
    expected_class_id: int
    expected_class_name: str


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]
    area_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从SKU视频自动检测、跟踪、裁切并去重分类图片。")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT, help="分类视频根目录。")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT, help="分类数据集输出目录。")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="训练好的YOLO权重。")
    parser.add_argument("--groups", nargs="+", default=list(GROUPS), choices=GROUPS, help="需要处理的大类。")
    parser.add_argument("--split", default="train", help="视频和输出的数据分组，默认train。")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="每秒送入YOLO的采样帧数，默认2。")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO推理尺寸，默认640。")
    parser.add_argument("--batch", type=int, default=16, help="YOLO批量推理帧数，默认16。")
    parser.add_argument("--device", default="0", help="推理设备，例如0或cpu，默认第0张GPU。")
    parser.add_argument("--predict-conf", type=float, default=0.10, help="YOLO最低输出置信度，默认0.10。")
    parser.add_argument("--anchor-conf", type=float, default=0.25, help="建立目标轨迹的同大类置信度，默认0.25。")
    parser.add_argument("--track-conf", type=float, default=0.15, help="轨迹建立后接受其他大类框的最低置信度。")
    parser.add_argument("--track-iou", type=float, default=0.10, help="与上一框匹配所需的最低IoU。")
    parser.add_argument("--max-center-distance", type=float, default=0.30, help="目标移动的最大归一化中心距离。")
    parser.add_argument("--max-track-gap", type=int, default=3, help="连续丢失多少个采样帧后重建轨迹。")
    parser.add_argument("--min-area-ratio", type=float, default=0.03, help="检测框至少占原图面积比例。")
    parser.add_argument("--min-aspect-ratio", type=float, default=0.18, help="检测框最小宽高比。")
    parser.add_argument("--max-aspect-ratio", type=float, default=5.5, help="检测框最大宽高比。")
    parser.add_argument("--padding", type=float, default=0.05, help="裁切框四周扩展比例，默认5%%。")
    parser.add_argument(
        "--allow-horizontal-edge",
        action="store_true",
        help="允许裁切框贴住原图左右边缘；默认丢弃可能未完整入镜的商品。",
    )
    parser.add_argument("--min-blur-score", type=float, default=25.0, help="拉普拉斯清晰度最低值。")
    parser.add_argument("--min-hash-distance", type=int, default=5, help="相邻保存图片的感知哈希最小距离。")
    parser.add_argument("--force-save-seconds", type=float, default=3.0, help="即使画面相似也至少每隔几秒保存一张。")
    parser.add_argument("--max-per-video", type=int, default=80, help="每段视频最多保存图片数，默认80。")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG保存质量，默认95。")
    parser.add_argument("--max-videos", type=int, default=None, help="仅处理前N段视频，用于冒烟测试。")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已经存在的同名裁切图片。")
    parser.add_argument("--check-only", action="store_true", help="只检查模型、视频和环境，不开始处理。")
    return parser.parse_args()


def expected_class_for_sku(sku: str) -> tuple[int, str]:
    for prefix, class_info in PREFIX_TO_CLASS.items():
        if sku.upper().startswith(prefix):
            return class_info
    raise ValueError(f"无法根据SKU目录确定大类：{sku}")


def collect_video_jobs(input_root: Path, groups: list[str], split: str) -> list[VideoJob]:
    jobs: list[VideoJob] = []
    for group in groups:
        group_dir = input_root / group
        if not group_dir.is_dir():
            print(f"[跳过] 大类目录不存在：{group_dir}")
            continue
        for sku_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            expected_id, expected_name = expected_class_for_sku(sku_dir.name)
            video_dir = sku_dir / split
            videos = sorted(
                path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
            ) if video_dir.is_dir() else []
            for video in videos:
                jobs.append(VideoJob(group, sku_dir.name, video, expected_id, expected_name))
    return jobs


def load_model(model_path: Path, device: str):
    if not THIRD_PARTY_ROOT.is_dir():
        raise FileNotFoundError(f"项目内Ultralytics源码不存在：{THIRD_PARTY_ROOT}")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO权重不存在：{model_path}")

    sys.path.insert(0, str(THIRD_PARTY_ROOT))
    import torch
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != "8.4.113":
        raise RuntimeError(f"Ultralytics版本不正确：{ultralytics.__version__}，预期8.4.113")
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("指定了GPU推理，但当前PyTorch无法使用CUDA。可添加 --device cpu。")

    print(f"Ultralytics：{ultralytics.__version__}")
    print(f"模型权重：{model_path}")
    print(f"推理设备：{torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'}")
    return YOLO(str(model_path))


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps必须大于0。")
    if args.batch <= 0 or args.max_track_gap < 0 or args.max_per_video <= 0:
        raise ValueError("批大小、最大轨迹间隔和单视频上限参数不正确。")
    for name in ("predict_conf", "anchor_conf", "track_conf", "track_iou", "min_area_ratio"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')}必须在0到1之间。")
    if args.predict_conf > min(args.anchor_conf, args.track_conf):
        raise ValueError("--predict-conf不能高于anchor-conf或track-conf。")
    if not 0.0 < args.min_aspect_ratio <= args.max_aspect_ratio:
        raise ValueError("检测框宽高比范围不正确。")
    if not 0.0 <= args.padding <= 0.5:
        raise ValueError("--padding必须在0到0.5之间。")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality必须在1到100之间。")


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(min(ax2, bx2) - max(ax1, bx1), 0.0)
    inter_h = max(min(ay2, by2) - max(ay1, by1), 0.0)
    intersection = inter_w * inter_h
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    return intersection / max(area_a + area_b - intersection, 1.0)


def center_distance(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> float:
    ax = (box_a[0] + box_a[2]) / 2.0
    ay = (box_a[1] + box_a[3]) / 2.0
    bx = (box_b[0] + box_b[2]) / 2.0
    by = (box_b[1] + box_b[3]) / 2.0
    return math.hypot(ax - bx, ay - by) / max(math.hypot(image_width, image_height), 1.0)


def area_size_ratio(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    area_a = max(box_a[2] - box_a[0], 0.0) * max(box_a[3] - box_a[1], 0.0)
    area_b = max(box_b[2] - box_b[0], 0.0) * max(box_b[3] - box_b[1], 0.0)
    return area_a / max(area_b, 1.0)


def parse_detections(result, args: argparse.Namespace) -> list[Detection]:
    detections: list[Detection] = []
    image_height, image_width = result.orig_shape
    image_area = max(float(image_width * image_height), 1.0)
    if result.boxes is None:
        return detections

    for class_id, confidence, xyxy in zip(
        result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist(), strict=True
    ):
        x1, y1, x2, y2 = (float(value) for value in xyxy)
        box_width = max(x2 - x1, 0.0)
        box_height = max(y2 - y1, 0.0)
        area_ratio = box_width * box_height / image_area
        aspect_ratio = box_width / max(box_height, 1.0)
        if area_ratio < args.min_area_ratio:
            continue
        if not args.min_aspect_ratio <= aspect_ratio <= args.max_aspect_ratio:
            continue
        detected_id = int(class_id)
        detections.append(
            Detection(
                class_id=detected_id,
                class_name=str(result.names.get(detected_id, detected_id)),
                confidence=float(confidence),
                box=(x1, y1, x2, y2),
                area_ratio=area_ratio,
            )
        )
    return detections


def select_detection(
    detections: list[Detection],
    expected_class_id: int,
    last_box: tuple[float, float, float, float] | None,
    image_width: int,
    image_height: int,
    args: argparse.Namespace,
) -> tuple[Detection | None, str]:
    # 同大类高置信度检测用于首次建立轨迹，也能在跟踪漂移后重新校正。
    anchors = [
        detection
        for detection in detections
        if detection.class_id == expected_class_id and detection.confidence >= args.anchor_conf
    ]
    if anchors:
        if last_box is None:
            return max(anchors, key=lambda item: item.confidence + 0.15 * item.area_ratio), "expected_anchor"
        return max(
            anchors,
            key=lambda item: 0.65 * item.confidence + 0.25 * box_iou(last_box, item.box)
            + 0.10 * (1.0 - center_distance(last_box, item.box, image_width, image_height)),
        ), "expected_anchor"

    # 未建立轨迹时不接受其他大类框，避免把背景误检当作商品。
    if last_box is None:
        return None, "skip_no_anchor"

    matched: list[tuple[float, Detection]] = []
    for detection in detections:
        if detection.confidence < args.track_conf:
            continue
        iou = box_iou(last_box, detection.box)
        distance = center_distance(last_box, detection.box, image_width, image_height)
        size_ratio = area_size_ratio(detection.box, last_box)
        spatial_match = iou >= args.track_iou or (
            distance <= args.max_center_distance and 0.25 <= size_ratio <= 4.0
        )
        if not spatial_match:
            continue
        score = 0.45 * detection.confidence + 0.35 * iou + 0.20 * max(1.0 - distance, 0.0)
        matched.append((score, detection))

    if not matched:
        return None, "skip_track_mismatch"
    selected = max(matched, key=lambda item: item[0])[1]
    method = "tracked_expected" if selected.class_id == expected_class_id else "tracked_other_class"
    return selected, method


def expand_box(
    box: tuple[float, float, float, float], image_width: int, image_height: int, padding: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    left = max(0, int(x1 - width * padding))
    top = max(0, int(y1 - height * padding))
    right = min(image_width, int(x2 + width * padding + 0.999))
    bottom = min(image_height, int(y2 + height * padding + 0.999))
    return left, top, right, bottom


def blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def difference_hash(image: np.ndarray, hash_size: int = 8) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(hash_a: int, hash_b: int) -> int:
    return (hash_a ^ hash_b).bit_count()


def write_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"JPEG编码失败：{path}")
    path.write_bytes(encoded.tobytes())


def make_manifest_row(job: VideoJob, frame_info: dict[str, object]) -> dict[str, object]:
    return {
        "group": job.group,
        "sku": job.sku,
        "video": str(job.path),
        **frame_info,
    }


def process_video(
    job: VideoJob,
    model,
    output_root: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], Counter[str]]:
    cap = cv2.VideoCapture(str(job.path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{job.path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(f"视频元数据无效：{job.path}")

    # 四个大类分别训练独立的ResNet模型，因此按“大类/数据划分/SKU”保存。
    output_dir = output_root / job.group.lower() / args.split / job.sku
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_interval = source_fps / args.sample_fps
    next_sample_index = frame_interval / 2.0
    frame_index = 0
    sample_number = 0
    pending: list[tuple[np.ndarray, int, int, int]] = []
    manifest_rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()

    last_box: tuple[float, float, float, float] | None = None
    lost_samples = 0
    last_saved_hash: int | None = None
    last_saved_time_ms: int | None = None
    saved_count = 0

    def run_batch(batch_items: list[tuple[np.ndarray, int, int, int]]) -> None:
        nonlocal last_box, lost_samples, last_saved_hash, last_saved_time_ms, saved_count
        if not batch_items:
            return
        frames = [item[0] for item in batch_items]
        results = model.predict(
            source=frames,
            imgsz=args.imgsz,
            conf=args.predict_conf,
            device=args.device,
            verbose=False,
        )

        for (frame, sampled_frame_index, timestamp_ms, current_sample_number), result in zip(
            batch_items, results, strict=True
        ):
            height, width = frame.shape[:2]
            detections = parse_detections(result, args)
            selected, method = select_detection(
                detections,
                job.expected_class_id,
                last_box,
                width,
                height,
                args,
            )
            base_row: dict[str, object] = {
                "sample_number": current_sample_number,
                "frame_index": sampled_frame_index,
                "timestamp_ms": timestamp_ms,
                "expected_class_id": job.expected_class_id,
                "expected_class_name": job.expected_class_name,
                "detected_class_id": "",
                "detected_class_name": "",
                "confidence": "",
                "selection_method": method,
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "area_ratio": "",
                "blur_score": "",
                "hash_distance": "",
                "output": "",
                "status": "",
            }

            if selected is None:
                lost_samples += 1
                if lost_samples > args.max_track_gap:
                    last_box = None
                base_row["status"] = method
                counts[method] += 1
                manifest_rows.append(make_manifest_row(job, base_row))
                continue

            last_box = selected.box
            lost_samples = 0
            crop_box = expand_box(selected.box, width, height, args.padding)
            left, top, right, bottom = crop_box
            base_row.update(
                {
                    "detected_class_id": selected.class_id,
                    "detected_class_name": selected.class_name,
                    "confidence": f"{selected.confidence:.6f}",
                    "x1": left,
                    "y1": top,
                    "x2": right,
                    "y2": bottom,
                    "area_ratio": f"{selected.area_ratio:.6f}",
                }
            )
            # 左右贴边通常表示商品尚未完全进入画面或已经移出画面。
            # 上下贴边不直接过滤，因为商品平放时检测框经常自然接近画面底部。
            if not args.allow_horizontal_edge and (left == 0 or right == width):
                base_row["status"] = "skip_horizontal_edge"
                counts["skip_horizontal_edge"] += 1
                manifest_rows.append(make_manifest_row(job, base_row))
                continue

            crop = frame[top:bottom, left:right]
            if crop.size == 0:
                base_row["status"] = "skip_empty_crop"
                counts["skip_empty_crop"] += 1
                manifest_rows.append(make_manifest_row(job, base_row))
                continue

            current_blur = blur_score(crop)
            base_row["blur_score"] = f"{current_blur:.3f}"
            if current_blur < args.min_blur_score:
                base_row["status"] = "skip_blurry"
                counts["skip_blurry"] += 1
                manifest_rows.append(make_manifest_row(job, base_row))
                continue

            current_hash = difference_hash(crop)
            hash_distance = 64 if last_saved_hash is None else hamming_distance(last_saved_hash, current_hash)
            base_row["hash_distance"] = hash_distance
            time_since_last = (
                math.inf if last_saved_time_ms is None else (timestamp_ms - last_saved_time_ms) / 1000.0
            )
            if hash_distance < args.min_hash_distance and time_since_last < args.force_save_seconds:
                base_row["status"] = "skip_duplicate"
                counts["skip_duplicate"] += 1
                manifest_rows.append(make_manifest_row(job, base_row))
                continue
            if saved_count >= args.max_per_video:
                base_row["status"] = "skip_video_limit"
                counts["skip_video_limit"] += 1
                manifest_rows.append(make_manifest_row(job, base_row))
                continue

            filename = (
                f"{job.path.stem}__s{current_sample_number:04d}"
                f"__t{timestamp_ms:07d}ms__f{sampled_frame_index:06d}.jpg"
            )
            output_path = output_dir / filename
            if output_path.exists() and not args.overwrite:
                status = "skip_existing"
            else:
                write_jpeg(output_path, crop, args.jpeg_quality)
                status = "saved"
            base_row["output"] = str(output_path)
            base_row["status"] = status
            counts[status] += 1
            saved_count += 1
            last_saved_hash = current_hash
            last_saved_time_ms = timestamp_ms
            manifest_rows.append(make_manifest_row(job, base_row))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index + 1e-9 >= next_sample_index:
            timestamp_ms = round(frame_index / source_fps * 1000)
            pending.append((frame.copy(), frame_index, timestamp_ms, sample_number))
            sample_number += 1
            next_sample_index = (sample_number + 0.5) * frame_interval
            if len(pending) >= args.batch:
                run_batch(pending)
                pending.clear()
        frame_index += 1

    run_batch(pending)
    cap.release()
    return manifest_rows, counts


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.model = args.model.resolve()

    config_dir = PROJECT_ROOT / "work" / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    jobs = collect_video_jobs(args.input_root, args.groups, args.split)
    if args.max_videos is not None:
        jobs = jobs[: args.max_videos]
    if not jobs:
        raise RuntimeError("没有找到需要处理的分类视频。")

    sku_count = len({job.sku for job in jobs})
    print(f"待处理：{len(jobs)}段视频，{sku_count}个SKU")
    model = load_model(args.model, args.device)
    if args.check_only:
        print("模型、环境和视频目录检查通过，未开始生成数据集。")
        return

    all_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    total_counts: Counter[str] = Counter()
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {job.sku} / {job.path.name}")
        rows, counts = process_video(job, model, args.output_root, args)
        all_rows.extend(rows)
        total_counts.update(counts)
        summary_rows.append(
            {
                "group": job.group,
                "sku": job.sku,
                "video": str(job.path),
                "sampled": len(rows),
                "saved": counts["saved"] + counts["skip_existing"],
                "expected_anchor": sum(
                    1 for row in rows if row["selection_method"] == "expected_anchor" and row["status"] in {"saved", "skip_existing"}
                ),
                "tracked_other_class": sum(
                    1 for row in rows if row["selection_method"] == "tracked_other_class" and row["status"] in {"saved", "skip_existing"}
                ),
                "skipped": len(rows) - counts["saved"] - counts["skip_existing"],
            }
        )
        print(
            f"  采样{len(rows)}帧，保存{counts['saved'] + counts['skip_existing']}张，"
            f"无锚点{counts['skip_no_anchor']}，跟踪不匹配{counts['skip_track_mismatch']}，"
            f"模糊{counts['skip_blurry']}，重复{counts['skip_duplicate']}"
        )

    manifest_fields = [
        "group", "sku", "video", "sample_number", "frame_index", "timestamp_ms",
        "expected_class_id", "expected_class_name", "detected_class_id", "detected_class_name",
        "confidence", "selection_method", "x1", "y1", "x2", "y2", "area_ratio",
        "blur_score", "hash_distance", "output", "status",
    ]
    summary_fields = [
        "group", "sku", "video", "sampled", "saved", "expected_anchor", "tracked_other_class", "skipped"
    ]
    # 不同数据划分分别保留流水线记录，避免生成val时覆盖此前的train清单。
    manifest_path = args.output_root / f"pipeline_manifest_{args.split}.csv"
    summary_path = args.output_root / f"pipeline_summary_{args.split}.csv"
    write_csv(manifest_path, all_rows, manifest_fields)
    write_csv(summary_path, summary_rows, summary_fields)

    print("\n视频推理流水线完成")
    print(f"处理视频：{len(jobs)}段，SKU：{sku_count}个")
    print(f"采样帧：{len(all_rows)}，保存图片：{total_counts['saved'] + total_counts['skip_existing']}")
    print(f"输出目录：{args.output_root}（按大类/{args.split}/SKU组织）")
    print(f"逐帧清单：{manifest_path}")
    print(f"视频汇总：{summary_path}")


if __name__ == "__main__":
    main()
