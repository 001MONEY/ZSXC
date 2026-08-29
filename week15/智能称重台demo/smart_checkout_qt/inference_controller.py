"""智能称重台 Qt 后台推理控制器。

本模块只负责界面运行期的读取与推理，不训练模型、不重建特征库，也不修改数据库。
冻结参数：YOLO conf=0.25、IoU=0.45、特征相似度>=0.80、Top1/Top2间隔>=0.15。
"""

from __future__ import annotations

import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QThread, Signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.goods_dao import GoodsDao  # noqa: E402
from onnx_engine import (  # noqa: E402
    OnnxFeatureLibrary,
    YoloOnnxDetector,
    retrieval_match_onnx,
)


CLASS_NAMES = {0: "bag", 1: "bottle", 2: "box", 3: "cylinder"}
PACKAGE_NAMES = {
    "bag": "袋装",
    "bottle": "瓶装",
    "box": "盒装",
    "cylinder": "罐装",
}
BOX_COLORS = {
    0: (52, 211, 153),
    1: (96, 165, 250),
    2: (251, 146, 60),
    3: (250, 204, 21),
}
ONNX_DIR = PROJECT_ROOT / "runs" / "onnx"


def _load_chinese_font(size: int = 22) -> ImageFont.FreeTypeFont:
    for path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _expand_box(
    coordinates: np.ndarray, width: int, height: int, padding: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in coordinates)
    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)
    return (
        max(0, int(x1 - box_width * padding)),
        max(0, int(y1 - box_height * padding)),
        min(width, int(x2 + box_width * padding + 0.999)),
        min(height, int(y2 + box_height * padding + 0.999)),
    )


def _clip_box(
    coordinates: Any, width: int, height: int
) -> tuple[int, int, int, int] | None:
    """将检测框裁剪到图像内；完全无效的框返回 None。"""
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = (int(float(value)) for value in coordinates)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    left = min(max(left, 0), width - 1)
    right = min(max(right, 0), width - 1)
    top = min(max(top, 0), height - 1)
    bottom = min(max(bottom, 0), height - 1)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _summarize(
    counts: dict[str, int], products: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    total_quantity = 0
    total_amount = 0.0
    for model_class, quantity in sorted(counts.items()):
        product = products.get(model_class)
        if product is None:
            continue
        unit_price = float(product["unit_price"])
        amount = round(unit_price * quantity, 2)
        details.append(
            {
                "model_class": model_class,
                "sku_code": product["sku_code"],
                "name": product["product_name"],
                "unit_price": unit_price,
                "quantity": quantity,
                "amount": amount,
            }
        )
        total_quantity += quantity
        total_amount += amount
    return {
        "details": details,
        "total_quantity": total_quantity,
        "total_amount": round(total_amount, 2),
    }


def _compact_registration_crop(crop: np.ndarray, max_side: int = 640) -> np.ndarray:
    """限制未知样本缓存尺寸，保留比例并降低长视频采集的内存占用。"""
    height, width = crop.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return crop.copy()
    scale = max_side / float(longest)
    return cv2.resize(
        crop,
        (max(24, int(round(width * scale))), max(24, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _annotate_frame(
    frame_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    font: ImageFont.FreeTypeFont,
) -> np.ndarray:
    """绘制商品框；购物车信息由 Qt 侧展示，不重复压在画面底部。"""
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for detection in detections:
        clipped_box = _clip_box(detection["box"], image.width, image.height)
        if clipped_box is None:
            continue
        x1, y1, x2, y2 = clipped_box
        color = BOX_COLORS[detection["yolo_class"]]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        if detection["found"]:
            label = (
                f"{detection['name']}  ¥{detection['price']:.2f}  "
                f"S:{detection['similarity']:.2f} M:{detection['margin']:.2f}"
            )
        else:
            label = f"{PACKAGE_NAMES[detection['package_type']]}商品未注册  {detection['reason']}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        background_width = min(image.width, text_width + 16)
        background_height = min(image.height, text_height + 12)
        text_left = min(x1, max(0, image.width - background_width))
        if y1 >= background_height:
            text_top = y1 - background_height
        else:
            text_top = min(y1, max(0, image.height - background_height))
        text_right = min(image.width - 1, text_left + background_width)
        text_bottom = min(image.height - 1, text_top + background_height)
        draw.rounded_rectangle(
            (text_left, text_top, text_right, text_bottom),
            radius=5,
            fill=color,
        )
        draw.text(
            (text_left + 8, text_top + 3),
            label,
            font=font,
            fill=(15, 23, 42),
        )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


class InferenceController(QThread):
    """在独立线程中完成视频读取、ONNX推理和稳定购物车统计。"""

    frame_ready = Signal(object, object, object, float)
    state_changed = Signal(str)
    provider_ready = Signal(str)
    catalog_ready = Signal(int)
    failed = Signal(str)
    source_finished = Signal(str)

    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    SIMILARITY_THRESHOLD = 0.80
    MARGIN_THRESHOLD = 0.15
    PADDING = 0.05
    MIN_BOX_SIZE = 24
    STABLE_WINDOW = 25
    UNKNOWN_CAPTURE_INTERVAL = 3
    UNKNOWN_BUFFER_SIZE = 160

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._source: int | str = 0
        self._stop_requested = threading.Event()
        self._pause_requested = threading.Event()
        self._reload_requested = threading.Event()
        self._cart_lock = threading.Lock()
        self._unknown_lock = threading.Lock()
        self._signatures: deque[tuple[tuple[str, int], ...]] = deque(
            maxlen=self.STABLE_WINDOW
        )
        self._unknown_crops: dict[str, deque[tuple[int, np.ndarray]]] = {
            package_type: deque(maxlen=self.UNKNOWN_BUFFER_SIZE)
            for package_type in PACKAGE_NAMES
        }
        self._last_unknown_capture: dict[str, int] = {
            package_type: -self.UNKNOWN_CAPTURE_INTERVAL
            for package_type in PACKAGE_NAMES
        }
        self._frame_sequence = 0
        self._detector: YoloOnnxDetector | None = None
        self._library: OnnxFeatureLibrary | None = None
        self._products: dict[str, dict[str, Any]] = {}
        self._font = _load_chinese_font()

    def set_source(self, source: int | str) -> None:
        if self.isRunning():
            raise RuntimeError("推理运行中不能切换输入源")
        self._source = source
        self.clear_unknown_crops()

    def request_stop(self) -> None:
        self._stop_requested.set()
        self._pause_requested.clear()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._pause_requested.set()
        else:
            self._pause_requested.clear()

    def is_paused(self) -> bool:
        return self._pause_requested.is_set()

    def reset_cart(self) -> None:
        with self._cart_lock:
            self._signatures.clear()

    def get_unknown_crops(
        self, package_type: str, max_count: int = 64
    ) -> list[np.ndarray]:
        """返回覆盖整个采集时间的未知商品裁剪图，不再只取末尾连续帧。"""
        if package_type not in self._unknown_crops:
            return []
        with self._unknown_lock:
            entries = list(self._unknown_crops[package_type])
            if len(entries) > max_count:
                indices = sorted(
                    set(np.linspace(0, len(entries) - 1, max_count, dtype=int).tolist())
                )
                entries = [entries[index] for index in indices]
            return [crop.copy() for _, crop in entries]

    def clear_unknown_crops(self, package_type: str | None = None) -> None:
        """注册成功后清除旧样本，避免污染下一件同包装类型商品。"""
        with self._unknown_lock:
            if package_type is None:
                for name, crops in self._unknown_crops.items():
                    crops.clear()
                    self._last_unknown_capture[name] = (
                        self._frame_sequence - self.UNKNOWN_CAPTURE_INTERVAL
                    )
            elif package_type in self._unknown_crops:
                self._unknown_crops[package_type].clear()
                self._last_unknown_capture[package_type] = (
                    self._frame_sequence - self.UNKNOWN_CAPTURE_INTERVAL
                )

    def request_reload_library(self) -> None:
        """请求在推理线程内安全热重载特征库和商品目录。"""
        self._reload_requested.set()

    def _load_engine(self) -> None:
        if self._detector is None or self._library is None:
            self.state_changed.emit("正在加载 ONNX 模型与特征库…")
            detector_path = ONNX_DIR / "yolov8n_det.onnx"
            if not detector_path.is_file():
                raise FileNotFoundError(f"检测模型不存在：{detector_path}")
            self._detector = YoloOnnxDetector(detector_path)
            self._library = OnnxFeatureLibrary()
            providers = self._detector.session.get_providers()
            provider = "CUDA GPU" if "CUDAExecutionProvider" in providers else "CPU"
            self.provider_ready.emit(provider)

        dao = GoodsDao()
        try:
            products = dao.list_all()
        finally:
            dao.close()
        if not products:
            raise RuntimeError("商品数据库为空或连接失败")
        self._products = {item["model_class"]: item for item in products}
        # 注册前的25帧大多是“未知/空购物车”；若不清空，新类即使命中也要
        # 等旧窗口被冲掉。重载后从下一帧重新稳定，满足注册后立即反馈。
        with self._cart_lock:
            self._signatures.clear()
        self.catalog_ready.emit(len(self._products))

    def _reload_runtime_catalog(self) -> None:
        """新商品注册后重新载入特征文件和数据库快照。"""
        self.state_changed.emit("正在加载新商品特征…")
        self._library = OnnxFeatureLibrary()
        dao = GoodsDao()
        try:
            products = dao.list_all()
        finally:
            dao.close()
        if not products:
            raise RuntimeError("商品数据库为空或连接失败")
        self._products = {item["model_class"]: item for item in products}
        with self._cart_lock:
            self._signatures.clear()
        self.catalog_ready.emit(len(self._products))
        self._reload_requested.clear()
        self.state_changed.emit("识别运行中")

    def run(self) -> None:
        capture: cv2.VideoCapture | None = None
        self._stop_requested.clear()
        self._pause_requested.clear()
        with self._cart_lock:
            self._signatures.clear()
        try:
            self._load_engine()
            self.state_changed.emit("正在打开输入源…")
            capture = cv2.VideoCapture(self._source)
            if not capture.isOpened():
                raise RuntimeError(f"无法打开输入源：{self._source}")

            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.state_changed.emit("识别运行中")
            last_time = time.perf_counter()
            smoothed_fps = 0.0

            while not self._stop_requested.is_set():
                if self._reload_requested.is_set():
                    self._reload_runtime_catalog()
                if self._pause_requested.is_set():
                    self.msleep(30)
                    continue

                ok, frame = capture.read()
                if not ok:
                    reason = "视频播放完毕" if isinstance(self._source, str) else "输入源已断开"
                    self.source_finished.emit(reason)
                    break

                annotated, detections, cart = self.process_frame(frame, width, height)
                now = time.perf_counter()
                instant_fps = 1.0 / max(now - last_time, 1e-6)
                smoothed_fps = instant_fps if smoothed_fps == 0 else 0.85 * smoothed_fps + 0.15 * instant_fps
                last_time = now
                self.frame_ready.emit(annotated, detections, cart, smoothed_fps)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        finally:
            if capture is not None:
                capture.release()
            self._pause_requested.clear()
            self.state_changed.emit("已停止")

    def process_frame(
        self, frame: np.ndarray, width: int | None = None, height: int | None = None
    ) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
        """处理单帧。用于运行线程，也便于不打开摄像头的链路测试。"""
        if self._detector is None or self._library is None:
            self._load_engine()
        assert self._detector is not None
        assert self._library is not None

        actual_height, actual_width = frame.shape[:2]
        # VideoCapture 上报的尺寸偶尔与实际帧不一致，所有裁剪均以实际帧为准。
        width = actual_width
        height = actual_height
        result = self._detector.predict_with_rotation_fallback(
            source=frame,
            conf=self.CONFIDENCE_THRESHOLD,
            iou=self.IOU_THRESHOLD,
        )[0]

        detections: list[dict[str, Any]] = []
        frame_counts: Counter[str] = Counter()
        self._frame_sequence += 1
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                if class_id not in CLASS_NAMES:
                    continue
                clipped_box = _clip_box(box.xyxy[0].tolist(), width, height)
                if clipped_box is None:
                    continue
                x1, y1, x2, y2 = clipped_box
                if x2 - x1 < self.MIN_BOX_SIZE or y2 - y1 < self.MIN_BOX_SIZE:
                    continue

                package_type = CLASS_NAMES[class_id]
                left, top, right, bottom = _expand_box(
                    box.xyxy[0], width, height, self.PADDING
                )
                crop = frame[top:bottom, left:right]
                if crop.size == 0:
                    continue

                model_class, top1, top2_class, top2, margin = retrieval_match_onnx(
                    crop, self._library, package_type
                )
                match_group = package_type
                registered_dynamic = self._library.registered_classes.get(match_group, set())
                if registered_dynamic and model_class not in registered_dynamic:
                    # 原24 SKU继续只在冻结类别之间计算Top1/Top2间隔，避免新增类
                    # 作为第二名压低旧商品margin，造成注册后旧商品反而被拒绝。
                    model_class, top1, top2_class, top2, margin = retrieval_match_onnx(
                        crop,
                        self._library,
                        match_group,
                        excluded_classes=registered_dynamic,
                    )
                product = self._products.get(model_class)
                similarity_threshold, margin_threshold = self._library.thresholds_for(
                    match_group,
                    model_class,
                    self.SIMILARITY_THRESHOLD,
                    self.MARGIN_THRESHOLD,
                )
                similarity_ok = top1 >= similarity_threshold
                margin_ok = margin >= margin_threshold
                found = product is not None and similarity_ok and margin_ok

                # 动态类若只是勉强排第一但未达到其0.95高相似度门限，不能让它
                # 挡住原24 SKU；排除动态类后再按冻结的0.80/0.15检索一次。
                dynamic_classes = registered_dynamic
                if not found and model_class in dynamic_classes:
                    original_match = retrieval_match_onnx(
                        crop,
                        self._library,
                        match_group,
                        excluded_classes=dynamic_classes,
                    )
                    (
                        original_class,
                        original_top1,
                        original_top2_class,
                        original_top2,
                        original_margin,
                    ) = original_match
                    original_product = self._products.get(original_class)
                    if (
                        original_product is not None
                        and original_top1 >= self.SIMILARITY_THRESHOLD
                        and original_margin >= self.MARGIN_THRESHOLD
                    ):
                        model_class = original_class
                        top1 = original_top1
                        top2_class = original_top2_class
                        top2 = original_top2
                        margin = original_margin
                        product = original_product
                        similarity_threshold = self.SIMILARITY_THRESHOLD
                        margin_threshold = self.MARGIN_THRESHOLD
                        similarity_ok = True
                        margin_ok = True
                        found = True

                # 新商品的包装形态可能被YOLO分到相邻大类（如圆瓶被判为罐装）。
                # 仅当主分组拒识时，才在含“在线注册类”的其他分组中补做检索；
                # 原24 SKU不会走这条宽松路径，避免破坏原开放集边界。
                if not found:
                    fallback_candidates = []
                    for candidate_group in PACKAGE_NAMES:
                        if candidate_group == package_type:
                            continue
                        if not self._library.registered_classes.get(candidate_group):
                            continue
                        candidate = retrieval_match_onnx(
                            crop,
                            self._library,
                            candidate_group,
                        )
                        candidate_class, candidate_top1, candidate_top2_class, candidate_top2, candidate_margin = candidate
                        if candidate_class not in self._library.registered_classes[candidate_group]:
                            continue
                        candidate_product = self._products.get(candidate_class)
                        candidate_sim_threshold, candidate_margin_threshold = self._library.thresholds_for(
                            candidate_group,
                            candidate_class,
                            self.SIMILARITY_THRESHOLD,
                            self.MARGIN_THRESHOLD,
                        )
                        if (
                            candidate_product is not None
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
                                    candidate_product,
                                    candidate_sim_threshold,
                                    candidate_margin_threshold,
                                )
                            )
                    if fallback_candidates:
                        (
                            top1,
                            margin,
                            match_group,
                            model_class,
                            top2_class,
                            top2,
                            product,
                            similarity_threshold,
                            margin_threshold,
                        ) = max(fallback_candidates, key=lambda item: (item[0], item[1]))
                        similarity_ok = True
                        margin_ok = True
                        found = True
                if product is None:
                    reason = "商品库无记录"
                elif not margin_ok:
                    reason = f"类别间隔 {margin:.2f} < {margin_threshold:.2f}"
                elif not similarity_ok:
                    reason = f"相似度 {top1:.2f} < {similarity_threshold:.2f}"
                else:
                    reason = ""

                if not found:
                    with self._unknown_lock:
                        if (
                            self._frame_sequence
                            - self._last_unknown_capture[package_type]
                            >= self.UNKNOWN_CAPTURE_INTERVAL
                        ):
                            self._unknown_crops[package_type].append(
                                (
                                    self._frame_sequence,
                                    _compact_registration_crop(crop),
                                )
                            )
                            self._last_unknown_capture[package_type] = self._frame_sequence
                detections.append(
                    {
                        "box": [x1, y1, x2, y2],
                        "yolo_class": class_id,
                        "package_type": match_group,
                        "detected_package_type": package_type,
                        "model_class": model_class,
                        "found": found,
                        "name": product["product_name"] if found else "未注册商品",
                        "price": float(product["unit_price"]) if found else None,
                        "det_confidence": float(box.conf.item()),
                        "similarity": float(top1),
                        "margin": float(margin),
                        "top2_class": top2_class,
                        "top2_similarity": float(top2),
                        "reason": reason,
                        "similarity_threshold": float(similarity_threshold),
                        "margin_threshold": float(margin_threshold),
                    }
                )

        static_detections: list[dict[str, Any]] = []
        dynamic_best: dict[str, dict[str, Any]] = {}
        for detection in detections:
            model_class = detection["model_class"]
            group = detection["package_type"]
            is_dynamic = (
                detection["found"]
                and model_class in self._library.registered_classes.get(group, set())
            )
            if not is_dynamic:
                static_detections.append(detection)
                continue
            previous = dynamic_best.get(model_class)
            score = (detection["similarity"], detection["det_confidence"])
            previous_score = (
                (previous["similarity"], previous["det_confidence"])
                if previous is not None
                else (-1.0, -1.0)
            )
            if score > previous_score:
                dynamic_best[model_class] = detection
        # 一个新商品可能同时被YOLO画出多个不同包装框，只保留检索最可信的一框。
        detections = static_detections + list(dynamic_best.values())
        for detection in detections:
            if detection["found"]:
                frame_counts[detection["model_class"]] += 1

        signature = tuple(sorted(frame_counts.items()))
        with self._cart_lock:
            self._signatures.append(signature)
            stable_signature = Counter(self._signatures).most_common(1)[0][0]
        cart = _summarize(dict(stable_signature), self._products)
        annotated = _annotate_frame(frame, detections, self._font)
        return annotated, detections, cart
