# ============================================================
# 用训练好的肿瘤分割模型 (Br35HDet) 抠出肿瘤并保存查看
# 模型: runs/seg_train/weights/best.pt
# 输入: Br35HDet/yolodataset/images/val 验证集图片
# 输出: tumor_crop_out/ 下每个肿瘤的黑底抠图
# ============================================================
import sys
sys.path.insert(0, r"d:\project\step1\week12\ultralytics-8.4.113")

import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"d:\project\step1\week13\runs\seg_train\weights\best.pt"
VAL_DIR = r"d:\project\step1\week13\Br35HDet\yolodataset\images\val"
OUT_DIR = r"d:\project\step1\week13\tumor_crop_out"
MAX_IMGS = 10   # 处理前 10 张验证图


def crop_tumor(ori_img, mask_img, pts):
    """从原图按框裁剪，只保留掩码区域像素（肿瘤），放到黑底图上
    返回: (黑底抠图, 原图裁剪, 掩码)"""
    x1, y1, x2, y2 = pts
    crop_img = ori_img[y1:y2, x1:x2].copy()      # 原图对应区域
    crop_mask = mask_img[y1:y2, x1:x2]           # 掩码对应区域
    result = np.zeros_like(crop_img)             # 黑底图
    result[crop_mask > 0] = crop_img[crop_mask > 0]   # 只保留肿瘤像素
    return result, crop_img, crop_mask


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)

    val_imgs = sorted(glob.glob(os.path.join(VAL_DIR, "*.jpg")))[:MAX_IMGS]
    total_tumors = 0
    print(f"模型: {MODEL_PATH}")
    print(f"处理 {len(val_imgs)} 张验证图，输出到: {OUT_DIR}\n")

    for img_path in val_imgs:
        name = os.path.basename(img_path)
        results = model.predict(img_path, verbose=False)
        result = results[0]

        if result.masks is None or len(result.boxes) == 0:
            print(f"{name}: 未检测到肿瘤")
            continue

        bboxes = result.boxes.xyxy.cpu().numpy()
        confes = result.boxes.conf.cpu().numpy()
        maskes = result.masks.data.cpu().numpy()
        ori_img = result.orig_img

        print(f"{name}: 检测到 {len(bboxes)} 个肿瘤")
        for i, mask in enumerate(maskes):
            x1, y1, x2, y2 = map(int, bboxes[i])
            mask_img = (mask * 255).astype(np.uint8)
            mask_img = cv2.resize(mask_img, (ori_img.shape[1], ori_img.shape[0]))

            # 抠肿瘤（黑底）
            crop, crop_img, crop_mask = crop_tumor(ori_img, mask_img, (x1, y1, x2, y2))

            # 保存抠图
            stem = os.path.splitext(name)[0]
            save_path = os.path.join(OUT_DIR, f"{stem}_tumor{i}_conf{confes[i]:.2f}.jpg")
            cv2.imwrite(save_path, crop)
            print(f"  ✅ 已保存: {os.path.basename(save_path)}  (置信度 {confes[i]:.2f})")
            total_tumors += 1

    print(f"\n完成！共抠出 {total_tumors} 个肿瘤，全部保存在: {OUT_DIR}")


if __name__ == "__main__":
    main()