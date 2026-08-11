import sys
# ultralytics 源码位于 week12 目录，加入模块搜索路径
sys.path.insert(0, r"d:\project\step1\week12\ultralytics-8.4.113")

from ultralytics import YOLO

if __name__=="__main__":
    # 方案一：从头训练
    # model = YOLO("yolov8n-seg.yaml")      # 仅网络结构

    # 方案二：迁移学习（推荐）
    # model = YOLO("yolov8n-seg.yaml").load("yolov8n-seg.pt")
    # 或直接加载预训练权重
    model = YOLO("yolov8s-seg.pt")

    # 方案三：断点续训
    # model = YOLO("yolov8s-seg.yaml").load("models/weights/last.pt")

    results = model.train(
        data=r"D:\project\step1\week13\Br35HDet\yolodataset\dataset.yaml",
        epochs=100,
        imgsz=512,
        batch=16,
        amp=False,   # 关掉 AMP 自动检查， yolov8n.pt检查
        project=r"D:\project\step1\week13\runs",   # 显式指定输出目录，不依赖运行目录
        name="seg_train",
        exist_ok=True,
    )