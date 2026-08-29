r"""注册新商品：数据库加记录 + 特征库新增类（Qt 界面与命令行共用）。

注册流程：
  1) 数据库 products 加一行（GoodsDao.add_goods）
  2) 对真实场景裁剪图做清晰度过滤、感知哈希去重和时间分散采样，
     再提取 512 维特征并追加进对应大类的特征库
     （新增类或扩充样本），保存 runs/features/*.npy / *.json
  3) 为新增类建立多个姿态原型，兼顾正面、侧面、远近和横竖姿态
  4) 更新 products.feature_index 为新的类中心标识

注意：
  - 特征提取复用 ONNX 引擎（onnx_engine），不依赖 PT 模型。
  - 特征库被冻结方案引用，新增类后阈值（sim>=0.80 / margin>=0.15）保持不变，
    但新类样本数过少时类中心可能不稳定，建议注册时提供一段环绕商品拍摄的视频。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from onnx_engine import (  # noqa: E402
    OnnxFeatureLibrary,
    YoloOnnxDetector,
    preprocess_crop,
)
from database.goods_dao import GoodsDao  # noqa: E402

FEATURES_DIR = PROJECT_ROOT / "runs" / "features"
FEATURE_DIM = 512
LIBRARY_VERSION = 1
GROUPS = ("bag", "bottle", "box", "cylinder")
GROUP_TO_CLASS_ID = {"bag": 0, "bottle": 1, "box": 2, "cylinder": 3}
MIN_REGISTRATION_SAMPLES = 5
MAX_REGISTRATION_SAMPLES = 64
MAX_REGISTRATION_PROTOTYPES = 4
REGISTRATION_MIN_BLUR = 18.0
REGISTRATION_MIN_HASH_DISTANCE = 4
YOLO_ONNX_PATH = PROJECT_ROOT / "runs" / "onnx" / "yolov8n_det.onnx"


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


def _blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _difference_hash(image: np.ndarray, hash_size: int = 8) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def _uniform_indices(length: int, limit: int) -> list[int]:
    if length <= limit:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, limit, dtype=int).tolist()))


def select_registration_crops(
    crops_bgr: list[np.ndarray],
    max_samples: int = MAX_REGISTRATION_SAMPLES,
    min_blur: float = REGISTRATION_MIN_BLUR,
    min_hash_distance: int = REGISTRATION_MIN_HASH_DISTANCE,
) -> tuple[list[np.ndarray], dict[str, int]]:
    """筛选清晰、跨时间且外观不完全重复的注册样本。

    输入顺序视为采集时间顺序。先均匀限制候选规模，再按感知哈希去重；
    若严格去重后不足5张，会用时间分散的清晰样本补足，避免无法注册。
    """
    received = len(crops_bgr)
    valid: list[tuple[np.ndarray, float, int]] = []
    for crop in crops_bgr:
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 24:
            continue
        score = _blur_score(crop)
        if score < min_blur:
            continue
        valid.append((crop, score, _difference_hash(crop)))

    if not valid:
        raise ValueError("没有清晰且尺寸有效的注册样本，请重新拍摄商品。")

    # 至多检查160张，且覆盖整个采集时间，而不是只取末尾连续帧。
    candidate_indices = _uniform_indices(len(valid), max(max_samples * 2 + 32, 160))
    candidates = [valid[index] for index in candidate_indices]
    selected: list[tuple[np.ndarray, float, int]] = []
    for item in candidates:
        if all((item[2] ^ old[2]).bit_count() >= min_hash_distance for old in selected):
            selected.append(item)

    # 画面非常稳定时哈希会高度相似，仍按时间均匀补足最小样本数。
    if len(selected) < MIN_REGISTRATION_SAMPLES:
        selected_ids = {id(item[0]) for item in selected}
        for index in _uniform_indices(len(candidates), MIN_REGISTRATION_SAMPLES * 2):
            item = candidates[index]
            if id(item[0]) not in selected_ids:
                selected.append(item)
                selected_ids.add(id(item[0]))
            if len(selected) >= MIN_REGISTRATION_SAMPLES:
                break

    if len(selected) > max_samples:
        selected = [selected[index] for index in _uniform_indices(len(selected), max_samples)]

    selected_crops = [item[0] for item in selected]
    return selected_crops, {
        "received": received,
        "quality_valid": len(valid),
        "selected": len(selected_crops),
        "duplicates_or_excess_removed": max(0, len(valid) - len(selected_crops)),
    }


def collect_video_crops(
    video_path: str | Path,
    group: str,
    sample_fps: float = 3.0,
    confidence: float = 0.25,
    max_raw_samples: int = 160,
    allow_other_classes: bool = True,
) -> tuple[list[np.ndarray], dict[str, float | int | str]]:
    """从单商品补充视频均匀抽帧并提取主体。

    新商品在注册前可能被 YOLO 错分为相邻包装类型。注册视频明确要求只拍一件
    商品，因此默认在目标类别没有框时使用其他包装类别的最大框，但后续特征仍
    写入用户指定的真实包装组。
    """
    if group not in GROUPS:
        raise ValueError(f"未知包装大类：{group}")
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"补充视频不存在：{path}")
    if sample_fps <= 0:
        raise ValueError("sample_fps 必须大于0。")

    detector = YoloOnnxDetector(YOLO_ONNX_PATH)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开补充视频：{path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(source_fps / sample_fps, 1.0)
    next_sample = frame_step / 2.0
    frame_index = 0
    sampled_frames = 0
    detected_frames = 0
    detected_class_counts = {name: 0 for name in GROUPS}
    crops: list[np.ndarray] = []
    expected_class = GROUP_TO_CLASS_ID[group]
    try:
        while len(crops) < max_raw_samples:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index + 1e-9 < next_sample:
                frame_index += 1
                continue
            next_sample += frame_step
            sampled_frames += 1
            height, width = frame.shape[:2]
            result = detector.predict_with_rotation_fallback(
                frame,
                conf=confidence,
                iou=0.45,
            )[0]
            matches = []
            if result.boxes is not None:
                for box in result.boxes:
                    detected_class = int(box.cls.item())
                    x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
                    box_width = max(x2 - x1, 1.0)
                    box_height = max(y2 - y1, 1.0)
                    area = box_width * box_height
                    matches.append(
                        (
                            detected_class,
                            area,
                            float(box.conf.item()),
                            (x1, y1, x2, y2),
                        )
                    )
            if matches:
                expected_matches = [item for item in matches if item[0] == expected_class]
                candidates = expected_matches or (matches if allow_other_classes else [])
            else:
                candidates = []
            if candidates:
                # 注册视频应以单件商品为主；优先真实包装类型，否则选最大的主体框。
                detected_class, _, _, (x1, y1, x2, y2) = max(
                    candidates,
                    key=lambda item: (item[1], item[2]),
                )
                pad_x = (x2 - x1) * 0.05
                pad_y = (y2 - y1) * 0.05
                left = max(0, int(x1 - pad_x))
                top = max(0, int(y1 - pad_y))
                right = min(width, int(np.ceil(x2 + pad_x)))
                bottom = min(height, int(np.ceil(y2 + pad_y)))
                crop = frame[top:bottom, left:right]
                if crop.size and min(crop.shape[:2]) >= 24:
                    crops.append(crop.copy())
                    detected_frames += 1
                    detected_class_counts[GROUPS[detected_class]] += 1
            frame_index += 1
    finally:
        capture.release()

    return crops, {
        "video": str(path),
        "source_frames": frame_total,
        "source_fps": round(source_fps, 3),
        "sampled_frames": sampled_frames,
        "detected_frames": detected_frames,
        "detected_class_counts": detected_class_counts,
    }


def extract_features(group: str, crops_bgr: list[np.ndarray], library: OnnxFeatureLibrary) -> np.ndarray:
    """用 ONNX 特征模型提取一批裁剪图的 L2 归一化特征，返回 (N, 512)。"""
    features = []
    for crop in crops_bgr:
        if crop is None or crop.size == 0:
            continue
        blob = preprocess_crop(crop, library.img_size[group])
        feature = library.sessions[group].run(None, {library.input_names[group]: blob})[0][0]
        features.append(_l2_normalize(feature.astype(np.float32)))
    if not features:
        raise ValueError("没有有效的裁剪图可用于提取特征。")
    return np.stack(features)


def _build_prototypes(features: np.ndarray, max_prototypes: int) -> np.ndarray:
    """用确定性的球面聚类为一个 SKU 建立多个姿态原型。"""
    if len(features) == 0:
        raise ValueError("无法从空特征构建原型。")
    prototype_count = min(max_prototypes, max(1, int(np.ceil(len(features) / 12))))
    mean = _l2_normalize(features.mean(axis=0))
    seeds = [int(np.argmax(features @ mean))]
    while len(seeds) < prototype_count:
        similarities = features @ features[seeds].T
        nearest = similarities.max(axis=1)
        next_index = int(np.argmin(nearest))
        if next_index in seeds:
            break
        seeds.append(next_index)
    prototypes = features[seeds].copy()
    for _ in range(8):
        assignments = np.argmax(features @ prototypes.T, axis=1)
        updated = []
        for index in range(len(prototypes)):
            members = features[assignments == index]
            updated.append(_l2_normalize(members.mean(axis=0)) if len(members) else prototypes[index])
        new_prototypes = np.stack(updated).astype(np.float32)
        if np.allclose(new_prototypes, prototypes, atol=1e-5):
            prototypes = new_prototypes
            break
        prototypes = new_prototypes
    return prototypes


def add_sku_samples(group: str, sku: str, crops_bgr: list[np.ndarray]) -> dict[str, int]:
    """筛选并追加 SKU 特征，同时更新类中心和多姿态原型。"""
    if group not in GROUPS:
        raise ValueError(f"未知包装大类：{group}")
    if not crops_bgr:
        raise ValueError("未提供样本图像。")

    selected_crops, selection_stats = select_registration_crops(crops_bgr)
    library = OnnxFeatureLibrary()  # 复用 ONNX 会话
    new_features = extract_features(group, selected_crops, library)

    emb_path = FEATURES_DIR / f"{group}_embeddings.npy"
    labels_path = FEATURES_DIR / f"{group}_labels.json"
    centers_path = FEATURES_DIR / f"{group}_centers.npy"
    classes_path = FEATURES_DIR / f"{group}_classes.json"
    metadata_path = FEATURES_DIR / f"{group}_metadata.json"
    prototypes_path = FEATURES_DIR / f"{group}_prototypes.npy"
    prototype_labels_path = FEATURES_DIR / f"{group}_prototype_labels.json"

    embeddings = np.load(emb_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    classes = json.loads(classes_path.read_text(encoding="utf-8"))

    is_new_class = sku not in classes
    embeddings = np.vstack([embeddings, new_features])
    labels.extend([sku] * len(new_features))

    # 重算全部类中心（均值后 L2 归一化）。
    if is_new_class:
        classes.append(sku)
    class_to_idx = {name: index for index, name in enumerate(classes)}
    new_centers = np.zeros((len(classes), FEATURE_DIM), dtype=np.float32)
    for name in classes:
        indices = [i for i, label in enumerate(labels) if label == name]
        new_centers[class_to_idx[name]] = _l2_normalize(embeddings[indices].mean(axis=0))

    # 旧类默认沿用单中心；本次注册的目标类根据全部样本建立多个姿态原型。
    old_prototypes: dict[str, list[np.ndarray]] = {}
    if prototypes_path.is_file() and prototype_labels_path.is_file():
        loaded_prototypes = np.load(prototypes_path)
        loaded_labels = json.loads(prototype_labels_path.read_text(encoding="utf-8"))
        for label, prototype in zip(loaded_labels, loaded_prototypes):
            old_prototypes.setdefault(label, []).append(prototype)
    prototype_rows: list[np.ndarray] = []
    prototype_labels: list[str] = []
    for name in classes:
        indices = [i for i, label in enumerate(labels) if label == name]
        if name == sku:
            rows = _build_prototypes(embeddings[indices], MAX_REGISTRATION_PROTOTYPES)
        elif name in old_prototypes:
            rows = np.stack(old_prototypes[name])
        else:
            rows = new_centers[class_to_idx[name]][None, :]
        prototype_rows.extend(rows)
        prototype_labels.extend([name] * len(rows))

    np.save(emb_path, embeddings)
    labels_path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    np.save(centers_path, new_centers)
    classes_path.write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")
    np.save(prototypes_path, np.stack(prototype_rows).astype(np.float32))
    prototype_labels_path.write_text(
        json.dumps(prototype_labels, ensure_ascii=False), encoding="utf-8"
    )

    # 更新元数据中的样本数/类数。
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["samples"] = len(labels)
        metadata["num_classes"] = len(classes)
        metadata["augmented_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        metadata["registration_selection"] = selection_stats
        metadata["prototype_count"] = len(prototype_rows)
        registered_classes = list(metadata.get("registered_classes", []))
        if sku not in registered_classes:
            registered_classes.append(sku)
        metadata["registered_classes"] = registered_classes
        thresholds = dict(metadata.get("registration_thresholds", {}))
        # 在线注册类要求与所采样本高度相似（更严格的0.95），但由于基础模型
        # 从未训练过该类，不再要求与旧类拉开0.15，改用0.01的小间隔防止并列。
        thresholds[sku] = {"similarity": 0.95, "margin": 0.01}
        metadata["registration_thresholds"] = thresholds
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    action = "新增类" if is_new_class else "扩充样本"
    print(
        f"[注册] {group}/{sku} {action}：收到{selection_stats['received']}张，"
        f"筛选后+{len(new_features)}张（现共 {len(labels)} 个，{len(classes)} 类，"
        f"目标类原型{prototype_labels.count(sku)}个）"
    )
    return {
        **selection_stats,
        "samples_added": len(new_features),
        "target_prototypes": prototype_labels.count(sku),
    }


def suggest_sku(dao: GoodsDao, group: str) -> str:
    """按当前数据库最大编号建议下一个 SKU，如 bag07。"""
    max_num = 0
    for item in dao.list_all(active_only=False):
        code = item["sku_code"]
        if code.startswith(group) and code[len(group):].isdigit():
            max_num = max(max_num, int(code[len(group):]))
    return f"{group}{max_num + 1:02d}"


def suggest_model_class(group: str, sku: str, product_name: str = "") -> str:
    """建议分类名，如 BAG_07_new_snack。"""
    number = sku[len(group):] if sku.startswith(group) else sku
    stem = ""
    for char in product_name:
        if char.isascii() and (char.isalnum() or char == " "):
            stem += char
        elif stem and stem[-1] != "_":
            stem += "_"
    stem = stem.strip("_").replace(" ", "_").lower()
    if not stem:
        stem = "new"
    return f"{group.upper()}_{number}_{stem}"


def register_sku(
    group: str,
    sku: str,
    product_name: str,
    unit_price: float,
    crops_bgr: list[np.ndarray],
    model_class: str | None = None,
    dao: GoodsDao | None = None,
) -> dict:
    """完整注册：特征库 + 数据库 + feature_index；失败时恢复特征文件。"""
    close_dao = dao is None
    if dao is None:
        dao = GoodsDao()

    feature_files = list(FEATURES_DIR.glob(f"{group}_*"))
    database_added = False
    try:
        if model_class is None:
            model_class = suggest_model_class(group, sku, product_name)
        if any(item["sku_code"] == sku for item in dao.list_all(active_only=False)):
            raise ValueError(f"注册失败：SKU {sku} 已存在。")
        if any(item["model_class"] == model_class for item in dao.list_all(active_only=False)):
            raise ValueError(f"注册失败：检索分类名 {model_class} 已存在。")

        with tempfile.TemporaryDirectory(prefix=f"smart_checkout_{group}_") as backup_dir:
            backup = Path(backup_dir)
            for path in feature_files:
                shutil.copy2(path, backup / path.name)
            try:
                stats = add_sku_samples(group, model_class, crops_bgr)
                ok = dao.add_goods(
                    sku_code=sku,
                    product_name=product_name,
                    package_type=group,
                    unit_price=unit_price,
                    model_class=model_class,
                )
                if not ok:
                    raise ValueError(f"注册失败：SKU {sku} 可能已存在。")
                database_added = True

                classes = json.loads(
                    (FEATURES_DIR / f"{group}_classes.json").read_text(encoding="utf-8")
                )
                index = classes.index(model_class)
                marker = f"lib{LIBRARY_VERSION}_center{index}_{model_class}"
                if not dao.update_goods(sku, feature_index=marker):
                    raise RuntimeError("商品已写入，但 feature_index 更新失败。")
            except Exception:
                if database_added:
                    dao.delete_goods(sku, soft=False)
                    database_added = False
                # 删除本次新建文件，再从临时备份恢复注册前状态。
                for path in FEATURES_DIR.glob(f"{group}_*"):
                    path.unlink()
                for path in backup.iterdir():
                    shutil.copy2(path, FEATURES_DIR / path.name)
                raise

        return {
            "sku": sku,
            "model_class": model_class,
            "index": index,
            **stats,
        }
    finally:
        if close_dao:
            dao.close()
