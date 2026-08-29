"""安全移除答辩前临时注册的两个测试 SKU，并保留可恢复备份。"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.mysql_db import MySqlHelper  # noqa: E402


FEATURES_DIR = PROJECT_ROOT / "runs" / "features"
BACKUP_ROOT = PROJECT_ROOT / "work" / "sku_cleanup_backups"
TARGETS = (
    ("bottle", "bottle07", "BOTTLE_07_asm milktea"),
    ("cylinder", "cylinder07", "CYLINDER_07_cocacola sugar"),
)
FEATURE_SUFFIXES = (
    "embeddings.npy",
    "labels.json",
    "centers.npy",
    "classes.json",
    "metadata.json",
)


def _json_default(value: Any) -> str:
    return str(value)


def _atomic_write_json(path: Path, value: Any, *, indent: int | None = None) -> None:
    temporary = path.with_name(f"{path.name}.cleanup.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=indent), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_write_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f"{path.name}.cleanup.tmp")
    with temporary.open("wb") as file:
        np.save(file, value)
    os.replace(temporary, path)


def _prepare_group(group: str, target: str) -> dict[str, Any]:
    embeddings_path = FEATURES_DIR / f"{group}_embeddings.npy"
    labels_path = FEATURES_DIR / f"{group}_labels.json"
    centers_path = FEATURES_DIR / f"{group}_centers.npy"
    classes_path = FEATURES_DIR / f"{group}_classes.json"
    metadata_path = FEATURES_DIR / f"{group}_metadata.json"

    embeddings = np.load(embeddings_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    centers = np.load(centers_path)
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if embeddings.shape[0] != len(labels):
        raise RuntimeError(f"{group} embeddings 与 labels 数量不一致")
    if centers.shape[0] != len(classes):
        raise RuntimeError(f"{group} centers 与 classes 数量不一致")
    if classes.count(target) != 1:
        raise RuntimeError(f"{group} 目标类别不存在或重复：{target}")

    sample_mask = np.asarray(labels) != target
    removed_samples = int((~sample_mask).sum())
    if removed_samples <= 0:
        raise RuntimeError(f"{group} 目标类别没有样本：{target}")

    class_index = classes.index(target)
    new_embeddings = embeddings[sample_mask]
    new_labels = [label for label in labels if label != target]
    new_centers = np.delete(centers, class_index, axis=0)
    new_classes = [name for name in classes if name != target]
    new_metadata = dict(metadata)
    new_metadata["samples"] = len(new_labels)
    new_metadata["num_classes"] = len(new_classes)

    if new_embeddings.shape[0] != len(new_labels):
        raise RuntimeError(f"{group} 清理后 embeddings 与 labels 数量不一致")
    if new_centers.shape[0] != len(new_classes):
        raise RuntimeError(f"{group} 清理后 centers 与 classes 数量不一致")
    if not np.isfinite(new_embeddings).all() or not np.isfinite(new_centers).all():
        raise RuntimeError(f"{group} 清理后出现无效向量")

    return {
        "removed_samples": removed_samples,
        "embeddings_path": embeddings_path,
        "labels_path": labels_path,
        "centers_path": centers_path,
        "classes_path": classes_path,
        "metadata_path": metadata_path,
        "embeddings": new_embeddings,
        "labels": new_labels,
        "centers": new_centers,
        "classes": new_classes,
        "metadata": new_metadata,
    }


def main() -> int:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / timestamp
    backup_dir.mkdir(parents=True, exist_ok=False)

    helper = MySqlHelper()
    if not helper.connect() or helper.connection is None or helper.cursor is None:
        raise RuntimeError("无法连接 smart_checkout 数据库")

    backup_files: list[tuple[Path, Path]] = []
    prepared: dict[str, dict[str, Any]] = {}
    try:
        database_rows = []
        for group, sku, target in TARGETS:
            helper.cursor.execute(
                "SELECT * FROM products WHERE sku_code = %s AND model_class = %s",
                (sku, target),
            )
            row = helper.cursor.fetchone()
            if row is None:
                raise RuntimeError(f"数据库中找不到目标 SKU：{sku} / {target}")
            database_rows.append(row)
            prepared[group] = _prepare_group(group, target)

            for suffix in FEATURE_SUFFIXES:
                source = FEATURES_DIR / f"{group}_{suffix}"
                destination = backup_dir / source.name
                shutil.copy2(source, destination)
                backup_files.append((source, destination))

        (backup_dir / "database_rows.json").write_text(
            json.dumps(
                database_rows,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )

        for data in prepared.values():
            _atomic_write_npy(data["embeddings_path"], data["embeddings"])
            _atomic_write_json(data["labels_path"], data["labels"])
            _atomic_write_npy(data["centers_path"], data["centers"])
            _atomic_write_json(data["classes_path"], data["classes"])
            _atomic_write_json(data["metadata_path"], data["metadata"], indent=2)

        for _group, sku, target in TARGETS:
            affected = helper.cursor.execute(
                "DELETE FROM products WHERE sku_code = %s AND model_class = %s",
                (sku, target),
            )
            if affected != 1:
                raise RuntimeError(f"删除数据库记录失败：{sku} / {target}")
        helper.connection.commit()
    except Exception:
        helper.connection.rollback()
        for source, backup in backup_files:
            if backup.is_file():
                shutil.copy2(backup, source)
        raise
    finally:
        helper.close()

    print(f"[OK] 可恢复备份：{backup_dir}")
    for group, sku, target in TARGETS:
        data = prepared[group]
        print(
            f"[OK] 已删除 {sku} / {target}："
            f"{data['removed_samples']} 条向量，剩余 {len(data['classes'])} 类"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
