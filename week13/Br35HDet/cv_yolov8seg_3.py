# 1_convert_labelme_to_yolo_seg.py
import json
import sys
import cv2
from pathlib import Path
import shutil
import numpy as np

# Windows 控制台 UTF-8 支持（解决 emoji 打印报错）
sys.stdout.reconfigure(encoding='utf-8')

# cv2.imread 不支持中文路径，用 imdecode 替代
def cv2_imread(path):
    """读取图片，兼容中文路径"""
    img_bytes = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    return img

def convert_labelme_to_yolo_seg(
        labelme_dir: str,
        output_images_dir: str,
        output_labels_dir: str,
        class_name_to_id: dict = None,
        vis: bool = False,  # 生成带 polygon 的预览图（存到 ./vis/）
):
    """
    Convert all LabelMe (polygon) files to YOLOv8-seg flat format.

    Output structure:
      yolo_all/
        ├── images/   # all .jpg copied here
        └── labels/   # all .txt generated here (same name as .jpg)
    """
    labelme_dir = Path(labelme_dir)
    out_img = Path(output_images_dir)
    out_lbl = Path(output_labels_dir)
    out_img.mkdir(exist_ok=True, parents=True)
    out_lbl.mkdir(exist_ok=True, parents=True)

    if class_name_to_id is None:
        class_name_to_id = {"object": 0}
    print(f"✅ Using class mapping: {class_name_to_id}")

    # Optional vis dir
    vis_dir = None
    if vis:
        vis_dir = Path("./vis")
        vis_dir.mkdir(exist_ok=True)

    converted = 0
    for json_path in labelme_dir.glob("*.json"):
        try:
            # Load JSON
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Get image path & validate
            img_name = data.get("imagePath")
            if not img_name:
                print(f"[WARN] No imagePath in {json_path.name}, skipped.")
                continue
            img_path = labelme_dir / img_name
            if not img_path.exists():
                print(f"[WARN] Image not found: {img_path}, skipped.")
                continue

            # Read image (兼容中文路径)
            img = cv2_imread(str(img_path))
            if img is None:
                print(f"[WARN] Failed to load {img_path}, skipped.")
                continue
            h, w = img.shape[:2]

            # Copy image
            out_img_path = out_img / img_name
            out_img_path.parent.mkdir(exist_ok=True)
            shutil.copy2(img_path, out_img_path)

            # Process polygons
            lines = []
            for shape in data.get("shapes", []):
                if shape.get("shape_type") != "polygon":
                    continue
                label = shape["label"]
                if label not in class_name_to_id:
                    continue
                class_id = class_name_to_id[label]
                points = shape["points"]
                if not isinstance(points, list) or len(points) < 3:
                    continue

                # Normalize to [0,1]
                norm_pts = []
                for x, y in points:
                    norm_pts.extend([x / w, y / h])
                lines.append(f"{class_id} " + " ".join(map(str, norm_pts)))

            # Save .txt
            txt_path = out_lbl / f"{json_path.stem}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            # Visualize (optional)
            if vis_dir and lines:
                vis_img = img.copy()
                for line in lines:
                    parts = list(map(float, line.strip().split()))
                    pts = np.array(parts[1:]).reshape(-1, 2) * [w, h]
                    pts = pts.astype(int)
                    if len(pts) >= 3:
                        cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.imwrite(str(vis_dir / f"{json_path.stem}_seg.jpg"), vis_img)

            converted += 1

        except Exception as e:
            print(f"[ERROR] {json_path.name}: {e}")

    print(f"\n✅ Conversion done: {converted} files saved to {out_img.parent}")
    print(f"   images → {out_img}")
    print(f"   labels → {out_lbl}")
    if vis:
        print(f"   visualizations → {vis_dir}")


if __name__ == "__main__":
    # ======================
    # ✅ USER CONFIG —— 修改这里
    # ======================
    convert_labelme_to_yolo_seg(
        labelme_dir=r"D:\project\step1\week13\Br35HDet\Br35HDet\images_labelme",
        output_images_dir=r"D:\project\step1\week13\Br35HDet\yolo_all\images",
        output_labels_dir=r"D:\project\step1\week13\Br35HDet\yolo_all\labels",
        class_name_to_id={"tumor": 0},
        vis=False,
    )