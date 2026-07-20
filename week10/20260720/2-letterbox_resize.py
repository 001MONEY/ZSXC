import cv2
import os

INPUT_DIR = r"E:\zs_kejian\AIcourse\07-YOLOV3\datas\xiaojinyu\frame_out"
OUTPUT_DIR = r"E:\zs_kejian\AIcourse\07-YOLOV3\datas\xiaojinyu\frame_out_416"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for name in os.listdir(INPUT_DIR):
    img = cv2.imread(os.path.join(INPUT_DIR, name))
    if img is None:
        continue
    h, w = img.shape[:2]

    # 1. 长边填充灰边 → 正方形
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    square = cv2.copyMakeBorder(img, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    # 2. 缩放成 416×416
    out = cv2.resize(square, (416, 416))

    cv2.imwrite(os.path.join(OUTPUT_DIR, name), out)

print(f"完成, 处理了 {len(os.listdir(INPUT_DIR))} 张图片")
print(f"输出目录: {OUTPUT_DIR}/")
