r"""导出推理模型为 ONNX（冻结方案的部署优化，不改特征库/阈值）。

导出 5 个模型到 runs/onnx/：
  - yolov8n_det.onnx                YOLOv8 检测（输入 1x3x640x640）
  - {bag,bottle,box,cylinder}_resnet18_feat.onnx   特征提取（输入 1x3x224x224，输出 512 维）

用法：
  D:\project\step1\env\python.exe export_onnx.py            # 导出全部
  D:\project\step1\env\python.exe export_onnx.py --yolo     # 只导出检测
  D:\project\step1\env\python.exe export_onnx.py --feat     # 只导出特征模型

说明：
  - 特征模型导出时去掉 fc 层（model.fc = Identity），输出 512 维 L2 特征；
    输入是已经过 EnsureRGB+SquarePad+Resize+Normalize 的归一化张量，
    ONNX 推理侧需用完全相同的预处理（见 onnx_engine.py）。
  - 导出不改动 runs/features 特征库与阈值，冻结方案不受影响。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CLASSIFY_DIR = PROJECT_ROOT / "runs" / "classify"
DETECT_PT = PROJECT_ROOT / "runs" / "detect" / "smart_checkout_yolov8n" / "weights" / "best.pt"
ONNX_DIR = PROJECT_ROOT / "runs" / "onnx"
GROUPS = ("bag", "bottle", "box", "cylinder")
OPSET = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出推理模型为 ONNX。")
    parser.add_argument("--yolo", action="store_true", help="只导出 YOLO 检测模型。")
    parser.add_argument("--feat", action="store_true", help="只导出 ResNet 特征模型。")
    parser.add_argument("--device", default="cpu", help="导出设备：cpu（默认）或0。")
    return parser.parse_args()


def export_yolo(device: str) -> Path:
    """用 ultralytics 导出 YOLOv8 检测模型。"""
    sys.path.insert(0, str(PROJECT_ROOT / "third_party"))
    from ultralytics import YOLO

    if not DETECT_PT.is_file():
        raise FileNotFoundError(f"YOLO 权重不存在：{DETECT_PT}")

    model = YOLO(str(DETECT_PT))
    exported = model.export(
        format="onnx",
        imgsz=640,
        opset=OPSET,
        dynamic=False,
        simplify=False,
        device=device,
    )
    src = Path(exported)
    dst = ONNX_DIR / "yolov8n_det.onnx"
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    print(f"✅ YOLO 已导出：{dst}（{dst.stat().st_size / 1e6:.1f} MB）")
    return dst


def export_feature_model(group: str, device: str) -> Path:
    """导出单个 ResNet 特征提取模型（去掉 fc 分类头）。"""
    import torch

    from train_resnet_classifier import build_model

    checkpoint_path = CLASSIFY_DIR / f"{group}_resnet18" / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"分类模型不存在：{checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, _ = build_model(checkpoint["architecture"], len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.fc = torch.nn.Identity()  # 输出 512 维特征
    model.eval()

    img_size = int(checkpoint["img_size"])
    device_obj = torch.device(device if device.isdigit() else "cpu")
    if device_obj.type == "cpu":
        model.to("cpu")
    else:
        model.to(device_obj)

    dummy = torch.randn(1, 3, img_size, img_size, device=device_obj)
    dst = ONNX_DIR / f"{group}_resnet18_feat.onnx"
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(dst),
        input_names=["input"],
        output_names=["feature"],
        opset_version=OPSET,
        dynamic_axes=None,  # 固定 batch=1
    )
    print(f"✅ {group} 特征模型已导出：{dst}（{dst.stat().st_size / 1e6:.1f} MB，img_size={img_size}）")
    return dst


def main() -> None:
    args = parse_args()
    do_yolo = args.yolo or not args.feat
    do_feat = args.feat or not args.yolo

    if do_yolo:
        export_yolo(args.device)
    if do_feat:
        for group in GROUPS:
            export_feature_model(group, args.device)
    print("\n导出完成，目录：", ONNX_DIR)


if __name__ == "__main__":
    main()
