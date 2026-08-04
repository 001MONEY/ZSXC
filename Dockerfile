# 基于本地已有的 PyTorch 官方镜像 (Python 3.11 + CUDA 12.6 + cuDNN 9)
# 已包含: torch / torchvision / CUDA 运行时
FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

# 安装 ultralytics (YOLO 全家桶: 检测/分类/分割/训练)
# 使用阿里云 PyPI 镜像加速国内下载
# --break-system-packages: 该镜像基于 Ubuntu 24.04 (PEP 668 外部管理环境)
RUN pip install --no-cache-dir --break-system-packages ultralytics -i https://mirrors.aliyun.com/pypi/simple/

# 常用开发/调试工具 + OpenCV 运行时依赖 (cv2 需要 libxcb/libgl 等)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    vim \
    htop \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# 进入 Jupyter 用的 notebook (pytorch 镜像已内置 jupyterlab)
# 工作目录设为挂载点 /workspace
WORKDIR /workspace

CMD ["/bin/bash"]
