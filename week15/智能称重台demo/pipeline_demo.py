r"""端到端识别：YOLO检测 → ResNet分类 → MySQL查价 → 汇总结算。

支持两种输入方式：
  本地视频：   D:\project\step1\env\python.exe pipeline_demo.py --source 视频.mp4 --name demo
  摄像头实时： D:\project\step1\env\python.exe pipeline_demo.py --camera 0

视频模式输出：标注视频、逐帧识别明细CSV、最终结算JSON。
摄像头模式：实时窗口显示识别结果与当前购物车，按 S 结算、R 重置、ESC 退出。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Windows 控制台/管道重定向时避免 GBK 编码崩溃（如 ¥ 字符）。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_resnet_classifier import EnsureRGB, SquarePad, build_model  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"
YOLO_MODEL = PROJECT_ROOT / "runs" / "detect" / "smart_checkout_yolov8n" / "weights" / "best.pt"
CLASSIFY_DIR = PROJECT_ROOT / "runs" / "classify"
DEFAULT_SOURCE = PROJECT_ROOT / "video" / "YOLO Data" / "val" / "VID_20260826_110333.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "pipeline"

CLASS_NAMES = {0: "bag", 1: "bottle", 2: "box", 3: "cylinder"}
BOX_COLORS = {0: (0, 200, 0), 1: (200, 0, 0), 2: (0, 140, 255), 3: (0, 220, 220)}
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="端到端识别联调：检测+分类+查价+结算。")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="输入视频（视频模式）。")
    parser.add_argument("--camera", type=int, default=None, help="摄像头索引（如0），启用摄像头实时模式。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出目录。")
    parser.add_argument("--name", default="pipeline_demo", help="本次运行名称。")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO置信度阈值。")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS阈值。")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO推理尺寸。")
    parser.add_argument("--padding", type=float, default=0.05, help="裁切区域扩展比例。")
    parser.add_argument("--min-box-size", type=int, default=24, help="小于该尺寸的检测框忽略。")
    parser.add_argument(
        "--unknown-threshold",
        type=float,
        default=0.5,
        help="分类置信度低于该值视为未注册商品，默认0.5（classify模式）。",
    )
    parser.add_argument(
        "--mode",
        choices=["classify", "retrieval"],
        default="retrieval",
        help="识别方式：retrieval=特征向量检索（推荐，可判断未注册），classify=Softmax分类。",
    )
    parser.add_argument(
        "--engine",
        choices=["pt", "onnx"],
        default="onnx",
        help="推理引擎：onnx=ONNX Runtime（默认，GPU优先），pt=PyTorch（需 third_party）。",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.80,
        help="检索模式相似度阈值，低于该值视为未注册商品，默认0.80（已用已注册/未注册标定）。",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.15,
        help="检索模式Top1-Top2间隔阈值，间隔过小说明最像与次像接近，视为未注册，默认0.15（已标定）。",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.5,
        help="摄像头模式显示窗口放大倍数，默认1.5。",
    )
    parser.add_argument("--device", default="0", help="推理设备：0、cpu或auto。")
    return parser.parse_args()


def load_yolo(model_file: Path, device: str):
    if not THIRD_PARTY_ROOT.is_dir():
        raise FileNotFoundError(f"项目内Ultralytics源码不存在：{THIRD_PARTY_ROOT}")
    sys.path.insert(0, str(THIRD_PARTY_ROOT))
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != "8.4.113":
        raise RuntimeError(f"Ultralytics版本不正确：{ultralytics.__version__}，预期8.4.113")
    return YOLO(str(model_file))


def load_classifiers(device: torch.device):
    """加载四个ResNet分类模型及对应的预处理变换。"""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    classifiers: dict[str, dict] = {}
    for group in ("bag", "bottle", "box", "cylinder"):
        checkpoint_path = CLASSIFY_DIR / f"{group}_resnet18" / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"分类模型不存在：{checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model, _ = build_model(checkpoint["architecture"], len(checkpoint["classes"]), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()

        img_size = checkpoint["img_size"]
        mean = checkpoint["imagenet_mean"]
        std = checkpoint["imagenet_std"]
        transform = transforms.Compose(
            [
                EnsureRGB(),
                SquarePad(),
                transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        classifiers[group] = {
            "model": model,
            "idx_to_class": {index: name for name, index in checkpoint["class_to_idx"].items()},
            "transform": transform,
            "img_size": img_size,
        }
    return classifiers


def expand_box(box: np.ndarray, width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    """按检测框宽高比例向四周扩展，并限制在图片边界内。"""
    x1, y1, x2, y2 = (float(value) for value in box)
    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)
    pad_x = box_width * padding
    pad_y = box_height * padding
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(width, int(x2 + pad_x + 0.999))
    bottom = min(height, int(y2 + pad_y + 0.999))
    return left, top, right, bottom


@torch.inference_mode()
def classify_crop(crop_bgr: np.ndarray, classifier: dict, device: torch.device) -> tuple[str, float]:
    """对裁切区域执行SKU分类，返回(model_class, 置信度)。"""
    pil_image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    tensor = classifier["transform"](pil_image).unsqueeze(0).to(device, non_blocking=True)
    logits = classifier["model"](tensor)
    probabilities = logits.softmax(dim=1)[0]
    index = int(logits.argmax(dim=1).item())
    return classifier["idx_to_class"][index], float(probabilities[index].item())


def load_feature_library(device: torch.device) -> dict[str, dict]:
    """加载四个包装大类的特征向量库与特征提取模型。

    返回：{group: {model, embeddings, labels, transform, img_size}}
    """
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    features_dir = PROJECT_ROOT / "runs" / "features"
    if not features_dir.is_dir():
        raise FileNotFoundError(f"特征库不存在：{features_dir}。请先运行 build_feature_library.py。")

    library: dict[str, dict] = {}
    for group in ("bag", "bottle", "box", "cylinder"):
        checkpoint_path = CLASSIFY_DIR / f"{group}_resnet18" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model, _ = build_model(checkpoint["architecture"], len(checkpoint["classes"]), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.fc = torch.nn.Identity()  # 去掉分类头，输出512维特征
        model.to(device).eval()

        img_size = checkpoint["img_size"]
        mean = checkpoint["imagenet_mean"]
        std = checkpoint["imagenet_std"]
        transform = transforms.Compose(
            [
                EnsureRGB(),
                SquarePad(),
                transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        embeddings = np.load(features_dir / f"{group}_embeddings.npy")
        labels = json.loads((features_dir / f"{group}_labels.json").read_text(encoding="utf-8"))
        centers = np.load(features_dir / f"{group}_centers.npy")
        classes = json.loads((features_dir / f"{group}_classes.json").read_text(encoding="utf-8"))
        library[group] = {
            "model": model,
            "embeddings": embeddings,
            "labels": labels,
            "centers": centers,
            "classes": classes,
            "transform": transform,
            "img_size": img_size,
        }
    return library


@torch.inference_mode()
def retrieval_match(
    crop_bgr: np.ndarray,
    lib: dict,
    device: torch.device,
    topk: int = 5,
    center_weight: float = 0.7,
) -> tuple[str, float, str, float, float]:
    """对裁切区域做特征检索，返回(Top1 SKU, Top1得分, Top2 SKU, Top2得分, 类别间隔)。

    每类得分 = 0.7 × 类中心相似度 + 0.3 × 该类Top-K样本平均相似度。
    相比"每类取最高样本相似度"，对域外/陌生商品更稳健，不易偶然撞上某张训练图。
    """
    pil_image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    tensor = lib["transform"](pil_image).unsqueeze(0).to(device, non_blocking=True)
    feature = lib["model"](tensor).squeeze(0).cpu().numpy()
    norm = float(np.linalg.norm(feature))
    feature = feature / max(norm, 1e-12)

    center_sims = lib["centers"] @ feature  # (6,)，与 lib["classes"] 行对应
    sample_sims = lib["embeddings"] @ feature  # (N,)
    labels = lib["labels"]

    class_scores: dict[str, float] = {}
    for index, class_name in enumerate(lib["classes"]):
        center_sim = float(center_sims[index])
        indices = [i for i, label in enumerate(labels) if label == class_name]
        k = min(topk, len(indices))
        if k > 0:
            values = sample_sims[indices]
            topk_mean = float(np.partition(values, -k)[-k:].mean())
        else:
            topk_mean = center_sim
        class_scores[class_name] = center_weight * center_sim + (1 - center_weight) * topk_mean

    ranked = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
    top1_class, top1_score = ranked[0]
    top2_class, top2_score = ranked[1] if len(ranked) > 1 else ("", 0.0)
    return top1_class, top1_score, top2_class, top2_score, top1_score - top2_score


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def annotate_frame(
    frame_bgr: np.ndarray,
    detections: list[dict],
    summary: dict,
    font: ImageFont.FreeTypeFont,
) -> np.ndarray:
    """在帧上绘制检测框、商品名/价格，以及底部结算条。"""
    pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        color = BOX_COLORS[detection["yolo_class"]]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = detection["label"]
        if detection["found"]:
            extra = f" {detection['class_conf']:.2f}"
            if detection.get("margin") is not None:
                extra += f" m:{detection['margin']:.2f}"
            text = f"{label} {detection['name']} ¥{detection['price']:.2f}{extra}"
        else:
            reason = detection.get("unknown_reason", "")
            text = f"{label}  未注册({reason})"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([x1, y1, text_bbox[2] + 4, text_bbox[3] + 4], fill=color)
        draw.text((x1 + 2, y1 + 2), text, fill=(0, 0, 0), font=font)

    # 底部结算条。
    width, height = pil_image.size
    bar_height = 40
    draw.rectangle([0, height - bar_height, width, height], fill=(20, 20, 20))
    bar_text = (
        f"本帧: {summary['frame_count']}件  ¥{summary['frame_amount']:.2f}    "
        f"累计组合: {summary['total_quantity']}件  ¥{summary['total_amount']:.2f}"
    )
    draw.text((12, height - bar_height + 8), bar_text, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def process_frame(
    frame: np.ndarray,
    frame_index: int,
    yolo,
    classifiers: dict,
    feature_library: dict | None,
    dao,
    device: torch.device,
    font: ImageFont.FreeTypeFont,
    args: argparse.Namespace,
    width: int,
    height: int,
) -> tuple[np.ndarray, list[dict], Counter[str], dict[str, Any]]:
    """对单帧执行 检测→识别→查价→标注，返回(标注帧, 检测列表, 计数, 汇总)。"""
    if getattr(args, "engine", "pt") == "onnx" and hasattr(
        yolo,
        "predict_with_rotation_fallback",
    ):
        results = yolo.predict_with_rotation_fallback(
            source=frame,
            conf=args.conf,
            iou=args.iou,
        )[0]
    else:
        results = yolo.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

    detections: list[dict] = []
    frame_counts: Counter[str] = Counter()
    if results.boxes is not None:
        for box in results.boxes:
            yolo_class = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            if x2 - x1 < args.min_box_size or y2 - y1 < args.min_box_size:
                continue
            group = CLASS_NAMES[yolo_class]
            left, top, right, bottom = expand_box(box.xyxy[0], width, height, args.padding)
            crop = frame[top:bottom, left:right]
            if crop.size == 0:
                continue
            if args.mode == "retrieval":
                effective_group = group
                if getattr(args, "engine", "pt") == "onnx":
                    from onnx_engine import retrieval_match_onnx

                    model_class, sim_top1, top2_class, sim_top2, margin = retrieval_match_onnx(
                        crop, feature_library, group
                    )
                    registered_dynamic = feature_library.registered_classes.get(group, set())
                    if registered_dynamic and model_class not in registered_dynamic:
                        model_class, sim_top1, top2_class, sim_top2, margin = retrieval_match_onnx(
                            crop,
                            feature_library,
                            group,
                            excluded_classes=registered_dynamic,
                        )
                else:
                    model_class, sim_top1, top2_class, sim_top2, margin = retrieval_match(
                        crop, feature_library[group], device
                    )
                class_conf = sim_top1  # 标签仍显示相似度
                threshold = args.similarity_threshold
                margin_threshold = args.margin_threshold
                if getattr(args, "engine", "pt") == "onnx":
                    threshold, margin_threshold = feature_library.thresholds_for(
                        effective_group,
                        model_class,
                        args.similarity_threshold,
                        args.margin_threshold,
                    )
                margin_value = round(margin, 4)
                top2_class_value = top2_class
                top2_sim_value = round(sim_top2, 4)
            else:
                effective_group = group
                model_class, class_conf = classify_crop(crop, classifiers[group], device)
                threshold = args.unknown_threshold
                margin_value = None
                top2_class_value = None
                top2_sim_value = None
                margin_threshold = args.margin_threshold
            goods = dao.get_by_model_class(model_class)
            registered = goods is not None
            # 未注册判断：SKU不在库，或相似度/置信度过低，或Top1-Top2间隔过小（检索模式）。
            unknown_by_conf = class_conf < threshold
            if args.mode == "retrieval":
                unknown_by_conf = unknown_by_conf or (margin_value < margin_threshold)
            found = registered and not unknown_by_conf
            if (
                not found
                and args.mode == "retrieval"
                and getattr(args, "engine", "pt") == "onnx"
                and model_class in feature_library.registered_classes.get(effective_group, set())
            ):
                dynamic_classes = feature_library.registered_classes[effective_group]
                original_class, original_top1, original_top2_class, original_top2, original_margin = retrieval_match_onnx(
                    crop,
                    feature_library,
                    effective_group,
                    excluded_classes=dynamic_classes,
                )
                original_goods = dao.get_by_model_class(original_class)
                if (
                    original_goods is not None
                    and original_top1 >= args.similarity_threshold
                    and original_margin >= args.margin_threshold
                ):
                    model_class = original_class
                    class_conf = original_top1
                    top2_class_value = original_top2_class
                    top2_sim_value = original_top2
                    margin_value = original_margin
                    goods = original_goods
                    threshold = args.similarity_threshold
                    margin_threshold = args.margin_threshold
                    registered = True
                    unknown_by_conf = False
                    found = True
            if (
                not found
                and args.mode == "retrieval"
                and getattr(args, "engine", "pt") == "onnx"
            ):
                fallback_candidates = []
                for candidate_group in CLASS_NAMES.values():
                    if candidate_group == group:
                        continue
                    dynamic_classes = feature_library.registered_classes.get(candidate_group, set())
                    if not dynamic_classes:
                        continue
                    candidate_class, candidate_top1, candidate_top2_class, candidate_top2, candidate_margin = retrieval_match_onnx(
                        crop,
                        feature_library,
                        candidate_group,
                    )
                    if candidate_class not in dynamic_classes:
                        continue
                    candidate_goods = dao.get_by_model_class(candidate_class)
                    candidate_sim_threshold, candidate_margin_threshold = feature_library.thresholds_for(
                        candidate_group,
                        candidate_class,
                        args.similarity_threshold,
                        args.margin_threshold,
                    )
                    if (
                        candidate_goods is not None
                        and candidate_top1 >= candidate_sim_threshold
                        and candidate_margin >= candidate_margin_threshold
                    ):
                        fallback_candidates.append(
                            (
                                candidate_top1,
                                candidate_margin,
                                candidate_group,
                                candidate_class,
                                candidate_top2_class,
                                candidate_top2,
                                candidate_goods,
                                candidate_sim_threshold,
                                candidate_margin_threshold,
                            )
                        )
                if fallback_candidates:
                    (
                        class_conf,
                        margin_value,
                        effective_group,
                        model_class,
                        top2_class_value,
                        top2_sim_value,
                        goods,
                        threshold,
                        margin_threshold,
                    ) = max(fallback_candidates, key=lambda item: (item[0], item[1]))
                    registered = True
                    unknown_by_conf = False
                    found = True
            if not found:
                if not registered:
                    unknown_reason = "未在商品库"
                elif args.mode == "retrieval" and margin_value is not None and margin_value < margin_threshold:
                    unknown_reason = f"间隔小({margin_value:.2f})"
                else:
                    unknown_reason = f"置信度低({class_conf:.2f})"
            else:
                unknown_reason = ""
            detections.append(
                {
                    "frame": frame_index,
                    "yolo_class": yolo_class,
                    "group": effective_group,
                    "detected_group": group,
                    "model_class": model_class,
                    "class_conf": round(class_conf, 4),
                    "margin": margin_value,
                    "top2_class": top2_class_value,
                    "top2_sim": top2_sim_value,
                    "det_conf": round(confidence, 4),
                    "box": [x1, y1, x2, y2],
                    "label": group,
                    "found": found,
                    "unknown_reason": unknown_reason,
                    "name": goods["product_name"] if found else None,
                    "price": float(goods["unit_price"]) if found else None,
                }
            )
    static_detections = []
    dynamic_best = {}
    for detection in detections:
        model_class = detection["model_class"]
        effective_group = detection["group"]
        is_dynamic = (
            detection["found"]
            and args.mode == "retrieval"
            and getattr(args, "engine", "pt") == "onnx"
            and model_class
            in feature_library.registered_classes.get(effective_group, set())
        )
        if not is_dynamic:
            static_detections.append(detection)
            continue
        previous = dynamic_best.get(model_class)
        score = (detection["class_conf"], detection["det_conf"])
        previous_score = (
            (previous["class_conf"], previous["det_conf"])
            if previous is not None
            else (-1.0, -1.0)
        )
        if score > previous_score:
            dynamic_best[model_class] = detection
    detections = static_detections + list(dynamic_best.values())
    for detection in detections:
        if detection["found"]:
            frame_counts[detection["model_class"]] += 1

    frame_summary = dao.summarize(dict(frame_counts))
    annotated = annotate_frame(
        frame,
        detections,
        {
            "frame_count": frame_summary["total_quantity"],
            "frame_amount": frame_summary["total_amount"],
            "total_quantity": frame_summary["total_quantity"],
            "total_amount": frame_summary["total_amount"],
        },
        font,
    )
    return annotated, detections, frame_counts, frame_summary


def print_settle(result: dict[str, Any], title: str = "结算") -> None:
    """打印结算结果。"""
    print(f"\n================ {title} ================")
    for item in result["details"]:
        if item.get("found", True):
            print(f"  {item['name']}  ¥{item['unit_price']:.2f} × {item['quantity']} = ¥{item['amount']:.2f}")
        else:
            print(f"  未注册商品：{item['model_class']}")
    print(f"  合计：{result['total_quantity']}件  ¥{result['total_amount']:.2f}")


def draw_cart_panel(
    frame_bgr: np.ndarray,
    cart: dict[str, Any],
    font: ImageFont.FreeTypeFont,
    extra_text: str = "",
) -> np.ndarray:
    """在帧顶部绘制当前购物车面板（摄像头模式）。"""
    pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    width, height = pil_image.size
    panel_height = 160
    draw.rectangle([0, 0, width, panel_height], fill=(30, 30, 30))
    draw.text((12, 8), "当前购物车（滑动窗口稳定组合）", fill=(255, 255, 0), font=font)
    draw.text((width - 420, 8), "S=结算  R=重置  ESC=退出", fill=(200, 200, 200), font=font)
    y = 38
    for item in cart["details"]:
        line = f"  {item['name']}  ¥{item['unit_price']:.2f} x {item['quantity']} = ¥{item['amount']:.2f}"
        draw.text((12, y), line, fill=(255, 255, 255), font=font)
        y += 26
    draw.text((12, y), f"合计：{cart['total_quantity']}件  ¥{cart['total_amount']:.2f}", fill=(0, 255, 0), font=font)
    if extra_text:
        draw.text((12, panel_height - 26), extra_text, fill=(255, 180, 0), font=font)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def run_video(args, yolo, classifiers, feature_library, dao, device, font) -> None:
    """视频模式：逐帧识别、保存标注视频、最终结算。"""
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{args.source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频：{width}x{height}，{fps:.2f}fps，{total_frames}帧")

    run_dir = args.output / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    video_path = run_dir / "result.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_records: list[dict] = []
    frame_signatures: list[tuple] = []

    for frame_index in range(total_frames):
        ok, frame = capture.read()
        if not ok:
            break
        annotated, detections, frame_counts, frame_summary = process_frame(
            frame, frame_index, yolo, classifiers, feature_library, dao, device, font, args, width, height
        )
        frame_records.extend(detections)
        frame_signatures.append(tuple(sorted(frame_counts.items())))
        writer.write(annotated)
        if frame_index % 100 == 0:
            print(f"  帧 {frame_index}/{total_frames}：{len(detections)}个商品，"
                  f"本帧{frame_summary['total_quantity']}件 ¥{frame_summary['total_amount']:.2f}")

    capture.release()
    writer.release()
    print("标注视频已保存：", video_path)

    if frame_records:
        with (run_dir / "frame_records.csv").open("w", newline="", encoding="utf-8-sig") as file:
            fields = ["frame", "yolo_class", "group", "model_class", "class_conf", "margin",
                      "top2_class", "top2_sim", "det_conf", "name", "price", "found",
                      "unknown_reason"]
            writer_csv = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer_csv.writeheader()
            writer_csv.writerows(frame_records)

    # 最终结算：取视频最后一段稳定期（最后20%%帧，至少30帧）中出现次数最多的组合。
    tail_size = max(30, int(total_frames * 0.2))
    tail_signatures = frame_signatures[-tail_size:]
    tail_counter = Counter(tail_signatures)
    final_signature, frequency = tail_counter.most_common(1)[0]
    final_result = dao.summarize(dict(final_signature))
    final_result["video_frames"] = total_frames
    final_result["settle_window_frames"] = len(tail_signatures)
    final_result["signature_frequency"] = frequency
    (run_dir / "result.json").write_text(
        json.dumps(final_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_settle(final_result, "最终结算")
    print(f"  该组合在最后{len(tail_signatures)}帧中出现 {frequency} 次")
    print(f"结果文件：{run_dir}")


def run_camera(args, yolo, classifiers, feature_library, dao, device, font) -> None:
    """摄像头模式：实时窗口识别，S结算、R重置、ESC退出。"""
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开摄像头：{args.camera}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头已打开：{width}x{height}，按 S 结算、R 重置、ESC 退出")

    window: deque = deque(maxlen=25)
    status_text = ""
    last_settle: dict[str, Any] | None = None
    frame_index = 0
    last_time = time.time()

    while True:
        ok, frame = capture.read()
        if not ok:
            print("无法读取摄像头帧，退出。")
            break
        frame_index += 1
        annotated, detections, frame_counts, _ = process_frame(
            frame, frame_index, yolo, classifiers, feature_library, dao, device, font, args, width, height
        )

        # 滑动窗口稳定组合，避免单帧抖动。
        window.append(tuple(sorted(frame_counts.items())))
        stable = Counter(window).most_common(1)[0][0]
        cart = dao.summarize(dict(stable))
        fps_now = 1.0 / max(time.time() - last_time, 1e-6)
        last_time = time.time()

        annotated = draw_cart_panel(annotated, cart, font, status_text)
        cv2.putText(
            annotated,
            f"FPS: {fps_now:.1f}  识别: {len(detections)}件",
            (12, annotated.shape[0] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        if args.display_scale != 1.0:
            annotated = cv2.resize(
                annotated,
                None,
                fx=args.display_scale,
                fy=args.display_scale,
                interpolation=cv2.INTER_LINEAR,
            )
        cv2.imshow("智能称重台-实时识别", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("已退出摄像头模式。")
            break
        elif key in (ord("s"), ord("S")):
            last_settle = cart
            status_text = f"已结算 {cart['total_quantity']}件 ¥{cart['total_amount']:.2f} @ {time.strftime('%H:%M:%S')}"
            print_settle(cart, "结算")
        elif key in (ord("r"), ord("R")):
            window.clear()
            status_text = "已重置购物车"
            print("购物车已重置")

    capture.release()
    cv2.destroyAllWindows()

    if last_settle is not None:
        run_dir = args.output / args.name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "camera_settle.json").write_text(
            json.dumps(last_settle, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("摄像头结算已保存：", run_dir / "camera_settle.json")


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()

    config_dir = PROJECT_ROOT / "work" / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    if args.device.lower() == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    print(f"推理设备：{device}")

    if args.engine == "onnx":
        from onnx_engine import OnnxFeatureLibrary, YoloOnnxDetector

        if args.mode != "retrieval":
            raise ValueError("ONNX 引擎当前仅支持 retrieval 模式（特征检索）。")
        yolo = YoloOnnxDetector(PROJECT_ROOT / "runs" / "onnx" / "yolov8n_det.onnx")
        classifiers = None
        feature_library = OnnxFeatureLibrary()
        engine_name = "ONNX Runtime"
    else:
        yolo = load_yolo(YOLO_MODEL, args.device)
        if args.mode == "retrieval":
            classifiers = None
            feature_library = load_feature_library(device)
        else:
            classifiers = load_classifiers(device)
            feature_library = None
        engine_name = "PyTorch"
    mode_name = "特征向量检索" if args.mode == "retrieval" else "Softmax分类"
    print(f"已加载模型（引擎：{engine_name}，识别方式：{mode_name}）")

    from database.goods_dao import GoodsDao

    dao = GoodsDao()
    font = load_font(22)

    if args.camera is not None:
        run_camera(args, yolo, classifiers, feature_library, dao, device, font)
    else:
        args.source = args.source.resolve()
        if not args.source.is_file():
            raise FileNotFoundError(f"输入视频不存在：{args.source}")
        run_video(args, yolo, classifiers, feature_library, dao, device, font)
    dao.close()


if __name__ == "__main__":
    main()
