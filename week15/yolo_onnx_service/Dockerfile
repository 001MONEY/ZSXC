# ==============================================================================
# YOLOv8n ONNX 推理服务镜像
# 思路：python 3.11 精简镜像 + onnxruntime 纯 CPU 推理，体积小、部署轻
# ==============================================================================
FROM python:3.11-slim

WORKDIR /app

# OpenCV(headless 版) 运行所需的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 先拷贝依赖文件再安装 —— 依赖没变时能复用 Docker 构建缓存，加快二次构建
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 拷贝应用代码和 ONNX 模型（模型打进镜像 = "封装"）
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

# 健康检查：容器是否真正"活"着
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
