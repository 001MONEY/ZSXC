# -*- coding: utf-8 -*-
"""
Grad-CAM 热力图（可解释性）

对 X 光片跑流水线，对每块骨用其关节分类模型生成 Grad-CAM，
显示模型在做等级判断时关注图像的哪些区域（骨骺、生长板等）。

输出（output/gradcam/）:
  {stem}_overlay.png   整图：每块骨热力图叠加在其检测框区域
  {stem}_grid.png      13 骨网格：[裁剪原图 | 热力图 | 叠加]

用法:
    python gradcam.py --image 图.png [--sex boy|girl] [--joints Radius Ulna]
    python gradcam.py --demo        # 用 1526 演示
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import config
from pipeline import Pipeline, CROP_TF, PAD, DEVICE

OUT_DIR = config.BAA_DIR / "output" / "gradcam"
JOINT_LABEL = {"DIPFirst": "DIP-1", "MCPFirst": "MCP-1", "PIPFirst": "PIP-1",
               "DIP": "DIP-3/5", "MCP": "MCP-3/5", "PIP": "PIP-3/5",
               "MIP": "MIP-3/5"}


# ---------------------------------------------------------------- Grad-CAM
class GradCAM:
    """对 ResNet 系模型（layer4 最后 block 特征图）做 Grad-CAM。"""

    def __init__(self, model):
        self.model = model
        self.grads = None
        self.acts = None
        self._fh = model.layer4[-1].register_forward_hook(self._forward_hook)

    def _forward_hook(self, m, i, o):
        self.acts = o.detach()
        o.register_hook(self._backward_hook)

    def _backward_hook(self, g):
        self.grads = g.detach()

    def generate(self, tensor, class_idx=None):
        """返回与输入同尺寸的归一化热力图 (0~1, numpy)。"""
        self.model.eval()
        with torch.enable_grad():
            out = self.model(tensor)
            if class_idx is None:
                # 分类: argmax 类别; 序数: 最强 logit
                class_idx = int(out[0].argmax().item())
            self.model.zero_grad()
            out[0, class_idx].backward()
        acts = self.acts[0]                     # C,H,W
        grads = self.grads[0]
        weights = grads.mean(dim=(1, 2))        # C
        cam = (weights[:, None, None] * acts).sum(0).clamp(min=0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam[None, None], size=(tensor.shape[2], tensor.shape[3]),
                            mode="bilinear", align_corners=False)[0, 0]
        return cam.cpu().numpy()


# ---------------------------------------------------------------- 工具
def crop_and_tensor(img, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1 - PAD), max(0, y1 - PAD)
    x2, y2 = min(img.shape[1], x2 + PAD), min(img.shape[0], y2 + PAD)
    crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    t = CROP_TF(transforms_to_pil(crop)).unsqueeze(0).to(DEVICE)
    return crop, t, (y1, y2, x1, x2)


def transforms_to_pil(rgb):
    from PIL import Image
    return Image.fromarray(rgb)


def heat_overlay(img_bgr, cam, alpha=0.55):
    """把归一化热力图叠加到 BGR 图上（jet 配色）。"""
    h, w = img_bgr.shape[:2]
    heat = cv2.resize(cam, (w, h))
    heat = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, heat_color, alpha, 0)


# ---------------------------------------------------------------- 主流程
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--sex", default="boy", choices=["boy", "girl"])
    parser.add_argument("--joints", nargs="+", default=None,
                        help="只画指定关节（如 Radius Ulna MIP）")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        image_path = config.DETECTION_PRE / "images" / "train" / "1526.png"
    else:
        image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline(ordinal_all=True, calibrated=True)
    print(f"[OK] 流水线已加载，处理: {image_path}")

    # 原图（用与分类一致的输入：先 CLAHE）
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    proc = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    res = pipe.predict(str(image_path), sex=args.sex)
    print(f"[OK] 检出 {res['n_bones']}/13 骨，骨龄={res['bone_age_years']} 岁")

    cams = {}   # rid -> (crop_bgr, cam)
    for rid, info in res["bones"].items():
        joint = info["classifier"]
        if args.joints and joint not in args.joints:
            continue
        model = pipe.clfs[joint]          # clfs[joint] = 模型对象
        crop, t, _ = crop_and_tensor(proc, info["box"])
        cam = GradCAM(model).generate(t)
        cams[rid] = (crop, cam)
        print(f"  {rid:<7} 等级 {info['grade']}  热力图 OK")

    # 1) 整图叠加
    overlay = proc.copy()
    for rid, info in res["bones"].items():
        if rid not in cams:
            continue
        crop, cam = cams[rid]
        y1, y2, x1, x2 = crop_and_tensor(proc, info["box"])[2]
        region = overlay[y1:y2, x1:x2]
        overlay[y1:y2, x1:x2] = heat_overlay(region, cam)
    ov_path = OUT_DIR / f"{image_path.stem}_overlay.png"
    cv2.imwrite(str(ov_path), overlay)
    print(f"[OK] 整图叠加 -> {ov_path}")

    # 2) 网格对比 [裁剪 | 热力图 | 叠加]
    rids = [rid for rid in res["bones"] if rid in cams]
    if rids:
        cols = 3
        cell_h, cell_w = 224, 224
        grid = np.full((len(rids) * (cell_h + 30), cols * (cell_w + 10), 3),
                       255, np.uint8)
        for i, rid in enumerate(rids):
            crop, cam = cams[rid]
            crop_r = cv2.resize(crop, (cell_w, cell_h))
            heat = cv2.resize((cam * 255).astype(np.uint8), (cell_w, cell_h))
            heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            ov = heat_overlay(crop_r, cam)
            y0 = i * (cell_h + 30)
            for j, im in enumerate([crop_r, heat_color, ov]):
                x0 = j * (cell_w + 10)
                grid[y0:y0 + cell_h, x0:x0 + cell_w] = im
            cv2.putText(grid, rid, (8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 2)
        g_path = OUT_DIR / f"{image_path.stem}_grid.png"
        cv2.imwrite(str(g_path), grid)
        print(f"[OK] 网格图 -> {g_path}")

    print(f"[OK] 全部完成，输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
