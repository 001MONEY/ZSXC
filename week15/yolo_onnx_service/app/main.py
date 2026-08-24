"""
================================================================================
YOLOv8n ONNX 推理服务 —— FastAPI 接口
================================================================================
对外提供：
    GET  /            Web 前端界面（上传图片可视化检测）
    GET  /docs         Swagger 接口文档
    GET  /health       健康检查（Docker 健康检查用它）
    GET  /api/info     服务信息
    POST /detect       上传图片 → 返回检测结果 JSON
    POST /annotate     上传图片 → 返回画好框的标注图
================================================================================
"""

import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .yolo_detector import CLASS_NAMES, YOLOv8nONNX

# 模型路径用环境变量控制，方便部署时换模型
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/yolov8n.onnx")

# Web 前端静态文件目录（和本文件同级）
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# 全局唯一的检测器（懒加载 + 进程内复用，避免每次请求都重新加载模型）
_detector = None


def get_detector() -> YOLOv8nONNX:
    global _detector
    if _detector is None:
        _detector = YOLOv8nONNX(MODEL_PATH)
    return _detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 服务启动时预加载模型，避免第一个请求卡顿
    get_detector()
    yield


app = FastAPI(title="YOLOv8n ONNX 推理服务",
              description="基于 ONNX Runtime 的目标检测服务（COCO 80 类）",
              version="1.0.0",
              lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index():
    """Web 前端界面"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/info")
def root():
    return {"service": "YOLOv8n ONNX 推理服务", "docs": "/docs", "health": "/health"}


# 托管前端静态资源（CSS/JS/图片等）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def health():
    return {"status": "ok",
            "model": os.path.basename(MODEL_PATH),
            "num_classes": len(CLASS_NAMES),
            "provider": get_detector().session.get_providers()}


async def _read_image(file: UploadFile) -> np.ndarray:
    content = await file.read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解析图片，请上传 jpg / png 等格式")
    return img


@app.post("/detect")
async def detect(file: UploadFile = File(...),
                 conf: float = 0.25,
                 iou: float = 0.45):
    """上传一张图片，返回所有检测框（类别、置信度、坐标）"""
    img = await _read_image(file)
    det = get_detector()
    det.conf_thres = conf
    det.iou_thres = iou
    results = det.detect(img)
    return {
        "filename": file.filename,
        "width": img.shape[1],
        "height": img.shape[0],
        "num_detections": len(results),
        "detections": results,
    }


@app.post("/annotate")
async def annotate(file: UploadFile = File(...),
                   conf: float = 0.25,
                   iou: float = 0.45):
    """上传一张图片，返回画好检测框的标注图（JPEG）"""
    img = await _read_image(file)
    det = get_detector()
    det.conf_thres = conf
    det.iou_thres = iou
    results = det.detect(img)

    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{r['class_name']} {r['confidence']:.2f}"
        cv2.putText(img, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    ok, buf = cv2.imencode(".jpg", img)
    return Response(content=buf.tobytes(), media_type="image/jpeg")
