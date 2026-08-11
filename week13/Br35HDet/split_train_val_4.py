# 2_split_yolo_train_val.py
import random
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')


def split_yolo_train_val(
        input_dir: str,
        output_dir: str,
        train_ratio: float = 0.8,
        seed: int = 42,
):
    """
    Split a flat YOLOv8-seg dataset (images/ + labels/) into train/val with images and labels nested directly under train and val.

    Input: yolo_all/ (with images/ and labels/)
    Output: yolo_split/ (with images/train, images/val, labels/train, labels/val subdirs + dataset.yaml)
    """
    input_dir = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(exist_ok=True)

    img_dir = input_dir / "images"
    lbl_dir = input_dir / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"Missing images/ or labels/ in {input_dir}")

    # Get all image names (with extension)
    img_files = [f for f in img_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    img_stems = [f.stem for f in img_files]

    # Match with .txt labels
    valid_pairs = []
    for stem in img_stems:
        txt_path = lbl_dir / f"{stem}.txt"
        if txt_path.exists():
            valid_pairs.append((stem, txt_path))

    print(f"✅ Found {len(valid_pairs)} valid image+label pairs.")

    # Shuffle & split
    random.seed(seed)
    random.shuffle(valid_pairs)
    n_train = int(len(valid_pairs) * train_ratio)
    train_pairs = valid_pairs[:n_train]
    val_pairs = valid_pairs[n_train:]

    # Setup output dirs
    train_img_out = out_root / "images" / "train"
    train_lbl_out = out_root / "labels" / "train"
    val_img_out = out_root / "images" / "val"
    val_lbl_out = out_root / "labels" / "val"
    for d in [train_img_out, train_lbl_out, val_img_out, val_lbl_out]:
        d.mkdir(exist_ok=True, parents=True)

    # Copy train
    print(f"\n🔄 Copying {len(train_pairs)} train files...")
    for stem, txt_path in train_pairs:
        # Find original image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"[WARN] Image missing for {stem}")
            continue
        # Copy
        shutil.copy2(img_path, train_img_out / img_path.name)
        shutil.copy2(txt_path, train_lbl_out / txt_path.name)

    # Copy val
    print(f"🔄 Copying {len(val_pairs)} val files...")
    for stem, txt_path in val_pairs:
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"[WARN] Image missing for {stem}")
            continue
        shutil.copy2(img_path, val_img_out / img_path.name)
        shutil.copy2(txt_path, val_lbl_out / txt_path.name)

    # Generate dataset.yaml
    # Infer classes from labels (or you can hardcode — safer)
    all_labels = list(lbl_dir.glob("*.txt"))
    classes = set()
    for txt in all_labels:
        with open(txt) as f:
            for line in f:
                if line.strip():
                    cls_id = int(line.strip().split()[0])
                    classes.add(cls_id)
    nc = len(classes)
    names = ["tumor"]


    # 生成标准 coco8-seg 格式的 yaml
    names_dict = "\n".join([f"  {i}: {name}" for i, name in enumerate(names)])
    yaml_content = f"""# YOLOv8-seg dataset (coco8-seg format)
path: {out_root.as_posix()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test:  # optional

# Classes
nc: {nc}
names:
{names_dict}
"""
    yaml_path = out_root / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n✅ Split done!")
    print(f"   Train: {len(train_pairs)} images, Val: {len(val_pairs)} images")
    print(f"   Classes: {names}")
    print(f"   Saved to: {out_root}")
    print(f"   dataset.yaml → {yaml_path}")


if __name__ == "__main__":
    root_dir = r"D:\project\step1\week13\Br35HDet"

    split_yolo_train_val(
        input_dir=root_dir+r"\yolo_all", 
        output_dir=root_dir+r"\yolodataset", 
        train_ratio=0.8,
        seed=42,
    )