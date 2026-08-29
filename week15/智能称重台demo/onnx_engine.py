r"""ONNX 推理引擎：YOLO 检测 + ResNet 特征检索（替代 PT 推理，接口与 pipeline_demo 兼容）。

模型来源（export_onnx.py 导出到 runs/onnx/）：
  - yolov8n_det.onnx                检测，输入 1x3x640x640，输出 1x8x8400（4类）
  - {group}_resnet18_feat.onnx      特征，输入 1x3x224x224，输出 1x512

关键设计：
  1. YoloOnnxDetector.predict() 返回的 results.boxes 与 ultralytics 接口兼容，
     因此 pipeline_demo.process_frame 无需改动即可切换推理引擎。
  2. 特征预处理（EnsureRGB+SquarePad+Resize+Normalize）用 PIL 精确复现 PT 侧，
     保证 ONNX 特征与 PT 特征一致（一致性由 verify_onnx.py 校验）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent
ONNX_DIR = PROJECT_ROOT / "runs" / "onnx"
FEATURES_DIR = PROJECT_ROOT / "runs" / "features"
GROUPS = ("bag", "bottle", "box", "cylinder")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_TORCH_CUDA_PATCHED = False


def _ensure_torch_cuda_dlls() -> None:
    """将 torch 自带的 CUDA/cuDNN 运行库（torch/lib）加入 PATH。

    onnxruntime GPU 版依赖 cublasLt64_12.dll / cudnn64_9.dll / cudart64_12.dll 等，
    这些 DLL 随 PyTorch cu121 安装在 torch/lib 下，但不在系统 PATH 中，
    导致 onnxruntime 找不到而回退到 CPU。此处把 torch/lib 注入 PATH 以启用 CUDA EP。
    """
    global _TORCH_CUDA_PATCHED
    if _TORCH_CUDA_PATCHED:
        return
    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
    except Exception:
        torch_lib = None
    if torch_lib is not None and torch_lib.is_dir():
        env_path = os.environ.get("PATH", "")
        if str(torch_lib) not in env_path:
            os.environ["PATH"] = str(torch_lib) + os.pathsep + env_path
    _TORCH_CUDA_PATCHED = True


# ---------------------------------------------------------------------------
# YOLO 检测（onnxruntime）
# ---------------------------------------------------------------------------

class _Box:
    """与 ultralytics Boxes 迭代项接口兼容的轻量对象。"""

    def __init__(self, xyxy: np.ndarray, cls_id: int, conf: float) -> None:
        import torch

        self.xyxy = torch.from_numpy(xyxy.reshape(1, 4).astype(np.float32))  # 1x4
        self.cls = torch.tensor([cls_id], dtype=torch.float32)
        self.conf = torch.tensor([conf], dtype=torch.float32)


class _Boxes:
    def __init__(self, boxes: list[_Box]) -> None:
        self._boxes = boxes

    def __iter__(self):
        return iter(self._boxes)


class _YoloResults:
    """与 ultralytics predict()[0] 接口兼容。"""

    def __init__(self, boxes: list[_Box] | None) -> None:
        self.boxes = _Boxes(boxes) if boxes else None


def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """等比缩放 + 灰边补到 new_shape，返回(图, 缩放比, (dw, dh)中心偏移)。"""
    height, width = img.shape[:2]
    target_h, target_w = new_shape
    ratio = min(target_h / height, target_w / width)
    new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
    dw, dh = (target_w - new_w) / 2.0, (target_h - new_h) / 2.0

    if (width, height) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    conf_thres: float,
    iou_thres: float,
) -> np.ndarray:
    """NMS，返回 [N, 6] = (x1, y1, x2, y2, conf, cls)。"""
    indices = np.where(scores >= conf_thres)[0]
    if indices.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    boxes = boxes[indices]
    scores = scores[indices]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = order[1:][iou <= iou_thres]
    return np.stack([x1[keep], y1[keep], x2[keep], y2[keep],
                     scores[keep], indices[keep].astype(np.float32)], axis=1)


class YoloOnnxDetector:
    """YOLOv8 ONNX 检测器（接口兼容 yolo.predict()）。"""

    def __init__(self, onnx_path: Path | str, num_classes: int = 4, input_size: int = 640) -> None:
        import onnxruntime as ort

        _ensure_torch_cuda_dlls()
        self.input_size = input_size
        self.num_classes = num_classes
        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, source, conf=0.25, iou=0.45, imgsz=640, device=None, verbose=False):
        """与 ultralytics YOLO.predict() 兼容的入口，返回 _YoloResults。"""
        del imgsz, device, verbose  # 统一走固定 input_size
        frame = np.ascontiguousarray(source, dtype=np.uint8)
        letterboxed, ratio, (dw, dh) = letterbox(frame, (self.input_size, self.input_size))
        blob = letterboxed[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        blob = blob[None, ...]  # 1x3xHxW

        out = self.session.run(None, {self.input_name: blob})[0]  # 1x(4+nc)x8400
        out = out[0]  # (4+nc, N)
        preds = np.transpose(out)  # (N, 4+nc)
        xywh = preds[:, :4]  # 归一化 xywh
        class_scores = preds[:, 4:]  # (N, nc)

        # xywh -> xyxy。注意：ultralytics 导出的 YOLOv8 ONNX 输出 box 是
        # 像素坐标（相对 letterbox 后的输入尺寸），不是 0-1 归一化，不能再乘 input_size。
        x_center, y_center, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        boxes_letterbox = np.stack([x1, y1, x2, y2], axis=1)

        # 逆 letterbox -> 原图坐标
        boxes_letterbox[:, [0, 2]] -= dw
        boxes_letterbox[:, [1, 3]] -= dh
        boxes_letterbox /= ratio

        # 过滤 + NMS（第6列为原始8400索引，用于取回类别）
        cls_ids = np.argmax(class_scores, axis=1)
        max_scores = np.max(class_scores, axis=1)
        out_boxes = non_max_suppression(boxes_letterbox, max_scores, conf, iou)

        result_boxes: list[_Box] = []
        for row in out_boxes:
            x1b, y1b, x2b, y2b, score, orig_idx = row
            result_boxes.append(_Box(
                xyxy=np.array([x1b, y1b, x2b, y2b], dtype=np.float32),
                cls_id=int(cls_ids[int(orig_idx)]),
                conf=float(score),
            ))
        # 与 ultralytics 一致：predict 返回 [Results]，调用方用 [0] 取结果。
        return [_YoloResults(result_boxes if result_boxes else None)]

    @staticmethod
    def _restore_rotated_box(
        xyxy: np.ndarray,
        original_width: int,
        original_height: int,
        rotation: int,
    ) -> np.ndarray:
        """把90度旋转图上的 xyxy 框映射回原始帧。"""
        x1, y1, x2, y2 = (float(value) for value in xyxy)
        corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        restored = []
        for x, y in corners:
            if rotation == cv2.ROTATE_90_CLOCKWISE:
                restored.append((y, original_height - x))
            elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
                restored.append((original_width - y, x))
            else:
                raise ValueError(f"不支持的旋转方式：{rotation}")
        xs = [point[0] for point in restored]
        ys = [point[1] for point in restored]
        return np.array(
            [
                np.clip(min(xs), 0, original_width),
                np.clip(min(ys), 0, original_height),
                np.clip(max(xs), 0, original_width),
                np.clip(max(ys), 0, original_height),
            ],
            dtype=np.float32,
        )

    def predict_with_rotation_fallback(
        self,
        source: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
    ):
        """原图无检测时先做重叠方形分块，再尝试两个90度方向。

        只有原图完全没有框才触发额外推理，因此常规答辩场景的速度不受影响。
        分块可缓解手机竖屏视频在 letterbox 后主体过小的问题；旋转用于横放商品。
        所有检测框最终都会映射回原始帧坐标。
        """
        primary = self.predict(source, conf=conf, iou=iou)
        if primary[0].boxes is not None:
            return primary

        original_height, original_width = source.shape[:2]
        # 手机竖屏/横屏视频先切成3个重叠方形区域，使商品在640输入中占比更大。
        if max(original_height, original_width) / max(min(original_height, original_width), 1) >= 1.35:
            side = min(original_height, original_width)
            travel = max(original_height, original_width) - side
            offsets = sorted(set(np.linspace(0, travel, 3, dtype=int).tolist()))
            tiled_boxes: list[_Box] = []
            for offset in offsets:
                if original_height > original_width:
                    tile = source[offset : offset + side, :]
                    x_offset, y_offset = 0, offset
                else:
                    tile = source[:, offset : offset + side]
                    x_offset, y_offset = offset, 0
                result = self.predict(tile, conf=conf, iou=iou)[0]
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    coordinates = box.xyxy[0].numpy().astype(np.float32)
                    coordinates[[0, 2]] += x_offset
                    coordinates[[1, 3]] += y_offset
                    tiled_boxes.append(
                        _Box(coordinates, int(box.cls.item()), float(box.conf.item()))
                    )
            if tiled_boxes:
                # 同一目标可能出现在相邻重叠块中，按类别做一次简单NMS。
                kept: list[_Box] = []
                for candidate in sorted(
                    tiled_boxes,
                    key=lambda item: float(item.conf.item()),
                    reverse=True,
                ):
                    candidate_box = candidate.xyxy[0].numpy()
                    candidate_cls = int(candidate.cls.item())
                    duplicate = False
                    for old in kept:
                        if int(old.cls.item()) != candidate_cls:
                            continue
                        old_box = old.xyxy[0].numpy()
                        xx1 = max(candidate_box[0], old_box[0])
                        yy1 = max(candidate_box[1], old_box[1])
                        xx2 = min(candidate_box[2], old_box[2])
                        yy2 = min(candidate_box[3], old_box[3])
                        intersection = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
                        area_a = max(0.0, candidate_box[2] - candidate_box[0]) * max(
                            0.0, candidate_box[3] - candidate_box[1]
                        )
                        area_b = max(0.0, old_box[2] - old_box[0]) * max(
                            0.0, old_box[3] - old_box[1]
                        )
                        overlap = intersection / max(area_a + area_b - intersection, 1e-9)
                        if overlap > iou:
                            duplicate = True
                            break
                    if not duplicate:
                        kept.append(candidate)
                return [_YoloResults(kept)]

        for rotation in (
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ):
            rotated = cv2.rotate(source, rotation)
            result = self.predict(rotated, conf=conf, iou=iou)[0]
            if result.boxes is None:
                continue
            restored_boxes = []
            for box in result.boxes:
                restored_boxes.append(
                    _Box(
                        self._restore_rotated_box(
                            box.xyxy[0].numpy(),
                            original_width,
                            original_height,
                            rotation,
                        ),
                        int(box.cls.item()),
                        float(box.conf.item()),
                    )
                )
            return [_YoloResults(restored_boxes)]
        return primary


# ---------------------------------------------------------------------------
# 特征提取（onnxruntime）+ 检索
# ---------------------------------------------------------------------------

def preprocess_crop(crop_bgr: np.ndarray, img_size: int) -> np.ndarray:
    """与 PT 侧完全一致的预处理（EnsureRGB+SquarePad+Resize+Normalize），返回 1x3xHxW。"""
    pil_image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")
    width, height = pil_image.size
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    right = side - width - left
    bottom = side - height - top
    pil_image = ImageOps.expand(pil_image, border=(left, top, right, bottom), fill=(114, 114, 114))
    pil_image = pil_image.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(pil_image, dtype=np.float32) / 255.0  # HWC 0-1
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # 1x3xHxW
    return np.ascontiguousarray(arr, dtype=np.float32)


class OnnxFeatureLibrary:
    """4 个大类的特征库 + ONNX 特征提取会话。"""

    def __init__(self, onnx_dir: Path = ONNX_DIR, features_dir: Path = FEATURES_DIR) -> None:
        import onnxruntime as ort

        _ensure_torch_cuda_dlls()
        self.sessions: dict[str, ort.InferenceSession] = {}
        self.input_names: dict[str, str] = {}
        self.embeddings: dict[str, np.ndarray] = {}
        self.labels: dict[str, list] = {}
        self.centers: dict[str, np.ndarray] = {}
        self.classes: dict[str, list] = {}
        self.prototypes: dict[str, np.ndarray] = {}
        self.prototype_labels: dict[str, list[str]] = {}
        self.registered_classes: dict[str, set[str]] = {}
        self.registration_thresholds: dict[str, dict[str, dict[str, float]]] = {}
        self.img_size: dict[str, int] = {}

        for group in GROUPS:
            model_path = onnx_dir / f"{group}_resnet18_feat.onnx"
            if not model_path.is_file():
                raise FileNotFoundError(f"ONNX 特征模型不存在：{model_path}，请先运行 export_onnx.py")
            session = ort.InferenceSession(
                str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.sessions[group] = session
            self.input_names[group] = session.get_inputs()[0].name
            self.img_size[group] = session.get_inputs()[0].shape[2]
            self.embeddings[group] = np.load(features_dir / f"{group}_embeddings.npy")
            self.labels[group] = json.loads((features_dir / f"{group}_labels.json").read_text(encoding="utf-8"))
            self.centers[group] = np.load(features_dir / f"{group}_centers.npy")
            self.classes[group] = json.loads((features_dir / f"{group}_classes.json").read_text(encoding="utf-8"))
            prototypes_path = features_dir / f"{group}_prototypes.npy"
            prototype_labels_path = features_dir / f"{group}_prototype_labels.json"
            if prototypes_path.is_file() and prototype_labels_path.is_file():
                self.prototypes[group] = np.load(prototypes_path)
                self.prototype_labels[group] = json.loads(
                    prototype_labels_path.read_text(encoding="utf-8")
                )
            else:
                # 冻结的24 SKU库没有额外原型文件时，单类中心就是唯一原型，
                # 因此分数与原0.7中心+0.3样本公式完全一致。
                self.prototypes[group] = self.centers[group].copy()
                self.prototype_labels[group] = list(self.classes[group])
            metadata_path = features_dir / f"{group}_metadata.json"
            metadata = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.is_file()
                else {}
            )
            self.registered_classes[group] = set(metadata.get("registered_classes", []))
            self.registration_thresholds[group] = metadata.get(
                "registration_thresholds",
                {},
            )

    def thresholds_for(
        self,
        group: str,
        class_name: str,
        default_similarity: float = 0.80,
        default_margin: float = 0.15,
    ) -> tuple[float, float]:
        """返回类别判定阈值；仅在线注册类使用更严格相似度和较小间隔。"""
        values = self.registration_thresholds.get(group, {}).get(class_name, {})
        return (
            float(values.get("similarity", default_similarity)),
            float(values.get("margin", default_margin)),
        )

    def extract_feature(self, group: str, crop_bgr: np.ndarray) -> np.ndarray:
        """提取 512 维 L2 归一化特征。"""
        blob = preprocess_crop(crop_bgr, self.img_size[group])
        feature = self.sessions[group].run(None, {self.input_names[group]: blob})[0][0]
        norm = float(np.linalg.norm(feature))
        return feature / max(norm, 1e-12)


def retrieval_match_onnx(
    crop_bgr: np.ndarray,
    lib: OnnxFeatureLibrary,
    group: str,
    topk: int = 5,
    center_weight: float = 0.7,
    prototype_weight: float = 0.35,
    excluded_classes: set[str] | None = None,
) -> tuple[str, float, str, float, float]:
    """与 pipeline_demo.retrieval_match 相同的检索逻辑（类中心+TopK平均）。

    返回 (Top1 SKU, Top1得分, Top2 SKU, Top2得分, 类别间隔)。
    """
    feature = lib.extract_feature(group, crop_bgr)
    center_sims = lib.centers[group] @ feature
    sample_sims = lib.embeddings[group] @ feature
    prototype_sims = lib.prototypes[group] @ feature
    labels = lib.labels[group]
    prototype_labels = np.asarray(lib.prototype_labels[group])

    class_scores: dict[str, float] = {}
    for index, class_name in enumerate(lib.classes[group]):
        if excluded_classes and class_name in excluded_classes:
            continue
        class_mask = np.array(labels) == class_name
        if not class_mask.any():
            continue
        center_score = float(center_sims[index])
        top_k = np.sort(sample_sims[class_mask])[-topk:]
        sample_score = float(top_k.mean()) if top_k.size else 0.0
        prototype_mask = prototype_labels == class_name
        prototype_score = (
            float(prototype_sims[prototype_mask].max())
            if prototype_mask.any()
            else center_score
        )
        # 旧类只有一个与中心相同的原型，公式仍等价于0.7中心+0.3样本。
        # 新增类可有多个姿态原型，避免不同角度被全局均值中心稀释。
        prototype_weight = min(max(prototype_weight, 0.0), center_weight)
        class_scores[class_name] = (
            (center_weight - prototype_weight) * center_score
            + prototype_weight * prototype_score
            + (1.0 - center_weight) * sample_score
        )

    ordered = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
    if not ordered:
        raise ValueError(f"{group} 特征库在排除指定类别后没有可检索类别。")
    top1_class, top1_score = ordered[0]
    top2_class, top2_score = ordered[1] if len(ordered) > 1 else (top1_class, top1_score)
    margin = top1_score - top2_score
    return top1_class, top1_score, top2_class, top2_score, margin


def load_onnx_engine(onnx_dir: Path = ONNX_DIR) -> tuple[YoloOnnxDetector, OnnxFeatureLibrary]:
    """加载 YOLO 检测器与特征库（retrieval 模式）。"""
    detector = YoloOnnxDetector(onnx_dir / "yolov8n_det.onnx")
    library = OnnxFeatureLibrary(onnx_dir)
    return detector, library
