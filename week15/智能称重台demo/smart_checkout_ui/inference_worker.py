r"""推理工作线程：后台加载 ONNX 引擎 + 逐帧推理 + 滑动窗口购物车。

复用冻结方案：
  - onnx_engine.YoloOnnxDetector / OnnxFeatureLibrary / retrieval_match_onnx（ONNX GPU）
  - pipeline_demo.annotate_frame / expand_box / CLASS_NAMES / load_font（标注）
  - database.goods_dao.GoodsDao（商品库/结算）
阈值已冻结：sim >= 0.80 且 margin >= 0.15 才算已注册。
"""

from __future__ import annotations

import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from onnx_engine import OnnxFeatureLibrary, YoloOnnxDetector, retrieval_match_onnx  # noqa: E402
from pipeline_demo import CLASS_NAMES, annotate_frame, expand_box, load_font  # noqa: E402
from database.goods_dao import GoodsDao  # noqa: E402

ONNX_DIR = PROJECT_ROOT / "runs" / "onnx"
ONNX_DET = ONNX_DIR / "yolov8n_det.onnx"


class InferenceWorker(QThread):
    """输入源（摄像头/视频）→ ONNX 推理 → 标注帧 + 购物车，信号回主线程。"""

    frame_ready = Signal(object, object, object, float)  # annotated_bgr, detections, cart, fps
    error = Signal(str)
    input_ended = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # 输入源：int = 摄像头索引，str = 视频文件路径
        self.source: int | str = 0
        # 冻结的推理/阈值参数
        self.conf = 0.25
        self.iou = 0.45
        self.padding = 0.05
        self.min_box_size = 24
        self.similarity_threshold = 0.80
        self.margin_threshold = 0.15
        self.video_loop = True  # 视频播完自动循环

        self._running = False
        self._window: deque = deque(maxlen=25)  # 滑动窗口稳定购物车
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        # 未注册商品样本采集（用于注册新商品）
        self.recent_unknown_crops: dict[str, deque] = {
            group: deque(maxlen=60) for group in CLASS_NAMES.values()
        }
        self._frame_seq = 0
        self._reload_requested = False
        self._library = None

    # ------------------------------------------------------------------ API
    def set_source(self, source: int | str) -> None:
        with self._lock:
            self.source = source

    def reset_cart(self) -> None:
        with self._lock:
            self._window.clear()

    def request_stop(self) -> None:
        self._running = False

    def get_unknown_crops(self, group: str, max_n: int = 20) -> list:
        """取某大类最近采集的未注册商品裁剪图（用于注册）。"""
        with self._lock:
            return [crop for _, crop in list(self.recent_unknown_crops[group])[-max_n:]]

    def request_reload_library(self) -> None:
        """注册新商品后请求热重载特征库（下一帧生效）。"""
        self._reload_requested = True

    # ------------------------------------------------------------ 线程主体
    def run(self) -> None:
        try:
            yolo = YoloOnnxDetector(ONNX_DET)
            self._library = OnnxFeatureLibrary()
            dao = GoodsDao()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"模型/数据库加载失败：{exc}")
            return

        font = load_font(22)
        try:
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                self.error.emit(
                    f"无法打开输入源：{self.source}（摄像头可能被占用或视频路径错误）"
                )
                dao.close()
                return
            self._running = True
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[Qt] 输入源 {self.source} 已打开：{width}x{height}")

            last_time = time.time()
            fps = 0.0
            while self._running:
                # 注册新商品后热重载特征库（下一帧生效）
                if self._reload_requested:
                    self._reload_requested = False
                    self._library = OnnxFeatureLibrary()
                    print("[Qt] 特征库已热重载（新商品注册生效）")

                ok, frame = self._cap.read()
                if not ok:
                    if isinstance(self.source, str) and self.video_loop:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 视频循环
                        continue
                    break
                annotated, detections, cart = self._process_frame(
                    frame, width, height, yolo, self._library, dao, font
                )
                now = time.time()
                dt = max(now - last_time, 1e-6)
                fps = fps * 0.9 + (1.0 / dt) * 0.1
                last_time = now
                self.frame_ready.emit(annotated, detections, cart, fps)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"推理异常：{exc}")
        finally:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            dao.close()
            self.input_ended.emit()

    # ------------------------------------------------------------ 单帧推理
    def _process_frame(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
        yolo: YoloOnnxDetector,
        library: OnnxFeatureLibrary,
        dao: GoodsDao,
        font,
    ) -> tuple[np.ndarray, list[dict], dict]:
        results = yolo.predict(source=frame, conf=self.conf, iou=self.iou)[0]
        detections: list[dict] = []
        frame_counts: Counter[str] = Counter()
        self._frame_seq += 1

        if results.boxes is not None:
            for box in results.boxes:
                yolo_class = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                if x2 - x1 < self.min_box_size or y2 - y1 < self.min_box_size:
                    continue
                group = CLASS_NAMES[yolo_class]
                left, top, right, bottom = expand_box(box.xyxy[0], width, height, self.padding)
                crop = frame[top:bottom, left:right]
                if crop.size == 0:
                    continue

                model_class, sim_top1, top2_class, sim_top2, margin = retrieval_match_onnx(
                    crop, library, group
                )
                goods = dao.get_by_model_class(model_class)
                registered = goods is not None
                unknown = (sim_top1 < self.similarity_threshold) or (margin < self.margin_threshold)
                found = registered and not unknown
                if not found:
                    if not registered:
                        reason = "未在商品库"
                    elif margin < self.margin_threshold:
                        reason = f"间隔小({margin:.2f})"
                    else:
                        reason = f"相似度低({sim_top1:.2f})"
                    # 采集未注册样本（用于注册新商品）
                    self.recent_unknown_crops[group].append((self._frame_seq, crop.copy()))
                else:
                    reason = ""
                    frame_counts[model_class] += 1

                detections.append(
                    {
                        "box": [x1, y1, x2, y2],
                        "yolo_class": yolo_class,
                        "label": group,
                        "found": found,
                        "name": goods["product_name"] if found else None,
                        "price": float(goods["unit_price"]) if found else None,
                        "class_conf": round(sim_top1, 4),
                        "margin": round(margin, 4),
                        "top2_class": top2_class,
                        "top2_sim": round(sim_top2, 4),
                        "unknown_reason": reason,
                        "det_conf": round(confidence, 4),
                    }
                )

        # 滑动窗口稳定组合（避免单帧抖动）
        with self._lock:
            self._window.append(tuple(sorted(frame_counts.items())))
            stable = Counter(self._window).most_common(1)[0][0]
        cart = dao.summarize(dict(stable))

        # 底部状态条需要"本帧"与"累计组合"两组字段（与 pipeline_demo 一致）
        frame_summary = dao.summarize(dict(frame_counts))
        summary = {
            "frame_count": frame_summary["total_quantity"],
            "frame_amount": frame_summary["total_amount"],
            "total_quantity": cart["total_quantity"],
            "total_amount": cart["total_amount"],
        }
        annotated = annotate_frame(frame, detections, summary, font)
        return annotated, detections, cart
