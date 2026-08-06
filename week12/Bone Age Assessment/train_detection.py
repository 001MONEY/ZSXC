# -*- coding: utf-8 -*-
"""
检测模型训练：YOLOv8 7类检测（迁移学习）

方案（两段式迁移学习，自定义类别数≠80 的标准姿势）：
    model = YOLO("yolov8n.yaml")        # 1) 用 yaml 重建网络 → 检测头按 7 类重新初始化
    model.load("yolov8n.pt")            # 2) 加载 COCO 预训练权重 → backbone/neck 直接复用

训练数据：datasets/detection_pre/（CLAHE 预处理版）
输出：runs/detect/{name}/ 下的 best.pt / last.pt 及指标

用法：
    python train_detection.py                        # 默认训练
    python train_detection.py --epochs 100 --batch 16 --imgsz 640
    python train_detection.py --smoke                # 快速冒烟测试（2 轮，验证流程）
    python train_detection.py --resume               # 从 runs/detect/{name}/weights/last.pt 恢复
    python train_detection.py --model yolov8s        # 换模型尺寸（n/s/m/l/x）
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

import config

DEFAULT_PT = config.WORKSPACE / "yolov8n.pt"          # COCO 预训练权重
DEFAULT_DATA = config.DETECTION_PRE / "data.yaml"     # 预处理后的检测数据集


def build_model(model_name: str, pretrained: Path, freeze_backbone: bool):
    """两段式构建：重建网络(按类别数初始化检测头) + 加载预训练权重。"""
    # 1) 用 yaml 重建网络（yolov8n.yaml 由 ultralytics 包内 cfg 解析）
    model = YOLO(f"{model_name}.yaml")
    # 2) 加载预训练权重（backbone/neck 复用，检测头随后由 train 按 nc 重新初始化）
    if pretrained.exists():
        model.load(str(pretrained))
        print(f"[OK] 已加载预训练权重: {pretrained}")
    else:
        print(f"[警告] 未找到预训练权重 {pretrained}，将从零训练")
    return model


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 骨龄检测模型训练（7 类）")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="数据集 data.yaml 路径")
    parser.add_argument("--model", default="yolov8n", help="模型尺寸: yolov8n/s/m/l/x")
    parser.add_argument("--pretrained", default=str(DEFAULT_PT), help="COCO 预训练权重路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--lr0", type=float, default=0.005, help="初始学习率（迁移学习用小值）")
    parser.add_argument("--freeze", type=int, default=10, help="冻结前 N 层（backbone）")
    parser.add_argument("--workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--name", default="bone7", help="实验名（runs/detect/{name}）")
    parser.add_argument("--project", default=str(config.BAA_DIR / "runs"), help="输出目录")
    parser.add_argument("--weights", default=None, help="从已有权重继续训练（.pt 路径），如 runs/bone7/weights/best.pt")
    parser.add_argument("--smoke", action="store_true", help="冒烟测试：只跑 2 轮验证流程")
    parser.add_argument("--resume", action="store_true", help="从上次中断处恢复")
    args = parser.parse_args()

    # 冒烟测试覆盖参数
    if args.smoke:
        args.epochs, args.freeze = 2, 0
        args.name = f"{args.name}_smoke"

    print("=" * 60)
    print(f"数据  : {args.data}")
    print(f"模型  : {args.model}  预训练: {args.pretrained}")
    print(f"训练  : epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} "
          f"lr0={args.lr0} freeze={args.freeze}")

    if args.weights:
        model = YOLO(args.weights)
        print(f"[OK] 从已有权重继续训练: {args.weights}")
    elif args.resume:
        last = Path(args.project) / "detect" / args.name / "weights" / "last.pt"
        if not last.exists():
            print(f"[错误] 未找到可恢复的权重: {last}")
            return
        model = YOLO(str(last))
        print(f"[OK] 恢复训练: {last}")
    else:
        model = build_model(args.model, Path(args.pretrained), args.freeze > 0)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        freeze=args.freeze,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=False,      # 权重已通过 .load() 显式加载，避免二次下载
        verbose=True,
    )

    # 训练结束后自动验证 best.pt，打印 mAP
    best = Path(args.project) / "detect" / args.name / "weights" / "best.pt"
    if best.exists():
        print("\n[OK] 训练完成，验证最佳模型 ...")
        val_model = YOLO(str(best))
        results = val_model.val(data=args.data, imgsz=args.imgsz, project=args.project,
                                name=f"{args.name}_val")
        print(f"[OK] best.pt 验证完成: {best}")


if __name__ == "__main__":
    main()
