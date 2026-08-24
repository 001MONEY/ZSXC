"""
================================================================================
YOLOv8n ONNX 推理封装
================================================================================
把 yolov8n.onnx 封装成一个可复用的检测器类，流程：
    图片(BGR) → letterbox 预处理 → onnxruntime 推理 → 后处理(NMS) → 检测结果

模型说明（yolov8n.onnx，COCO 80 类）：
    输入:  [1, 3, 640, 640]  float32，RGB，像素值归一化到 [0,1]
    输出:  [1, 84, 8400]
           84   = 4(框 cx,cy,w,h) + 80(类别得分)
           8400 = 640x640 下 3 个尺度共 8400 个候选框
================================================================================
"""

import cv2
import numpy as np
import onnxruntime as ort

# COCO 数据集的 80 个类别名（yolov8n 预训练权重用的就是它）
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


class YOLOv8nONNX:
    """基于 onnxruntime 的 YOLOv8n 目标检测器"""

    def __init__(self, model_path: str, conf_thres: float = 0.25,
                 iou_thres: float = 0.45, input_size: int = 640):
        # CPU 推理（跨平台、无需 GPU）；如需 GPU 换成 ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    # ------------------------------------------------------------------
    # 预处理：letterbox —— 保持宽高比缩放 + 四周灰色填充，凑成 640x640
    # 返回：blob[1,3,640,640]、缩放比例 ratio、填充偏移 (pad_left, pad_top)
    # ------------------------------------------------------------------
    def preprocess(self, img_bgr: np.ndarray):
        h, w = img_bgr.shape[:2]
        r = min(self.input_size / h, self.input_size / w)
        new_unpad = (round(w * r), round(h * r))

        dw, dh = self.input_size - new_unpad[0], self.input_size - new_unpad[1]
        left, top = dw // 2, dh // 2

        img = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, top, dh - top, left, dw - left,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # BGR -> RGB，HWC -> CHW，归一化 [0,1]，加 batch 维
        blob = img[:, :, ::-1].transpose(2, 0, 1)
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        blob = blob[None]
        return blob, r, (left, top)

    # ------------------------------------------------------------------
    # 后处理：解析 [1,84,8400] → 置信度过滤 → 坐标换算 → NMS → 映射回原图
    # ------------------------------------------------------------------
    def postprocess(self, output: np.ndarray, ratio: float, pad):
        preds = np.squeeze(output)              # (84, 8400) 或 (8400, 84)
        if preds.shape[0] == 84:
            preds = preds.T                     # 统一成 (8400, 84)

        boxes = preds[:, :4]                    # cx, cy, w, h（640 坐标系）
        class_scores = preds[:, 4:]             # 80 类得分

        class_ids = np.argmax(class_scores, axis=1)
        confs = class_scores[np.arange(len(class_scores)), class_ids]

        # 1) 置信度过滤
        keep = confs >= self.conf_thres
        boxes, class_ids, confs = boxes[keep], class_ids[keep], confs[keep]
        if len(boxes) == 0:
            return []

        # 2) cxcywh -> xyxy（640 坐标系）
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # 3) NMS 去重（同一目标只留一个框）
        indices = cv2.dnn.NMSBoxes(xyxy.tolist(), confs.tolist(),
                                   self.conf_thres, self.iou_thres)

        # 4) 把 640 坐标系的框映射回原图
        pad_left, pad_top = pad
        results = []
        for i in indices:
            i = int(i)
            cx, cy, w, h = boxes[i]
            bx1 = (cx - w / 2 - pad_left) / ratio
            by1 = (cy - h / 2 - pad_top) / ratio
            bx2 = (cx + w / 2 - pad_left) / ratio
            by2 = (cy + h / 2 - pad_top) / ratio
            results.append({
                "class_id": int(class_ids[i]),
                "class_name": CLASS_NAMES[int(class_ids[i])],
                "confidence": round(float(confs[i]), 4),
                "bbox": [round(float(bx1), 2), round(float(by1), 2),
                         round(float(bx2), 2), round(float(by2), 2)],
            })
        return results

    # ------------------------------------------------------------------
    # 对外主方法：输入 BGR 图片，输出检测结果列表
    # ------------------------------------------------------------------
    def detect(self, img_bgr: np.ndarray):
        blob, ratio, pad = self.preprocess(img_bgr)
        output = self.session.run([self.output_name], {self.input_name: blob})[0]
        return self.postprocess(output, ratio, pad)
