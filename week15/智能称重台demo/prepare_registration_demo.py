r"""把答辩演示恢复到“阿萨姆尚未注册”的 24-SKU 初始状态。

日常查看状态（只读）：
    D:\project\step1\env\python.exe prepare_registration_demo.py

录制前恢复初始状态：
    D:\project\step1\env\python.exe prepare_registration_demo.py --reset

脚本只会处理 bottle 特征库和固定演示 SKU ``bottle07``。执行重置前，
会先把当前 bottle 特征文件和数据库商品表快照保存到
``work/demo_state_backups/<时间戳>/``，因此录制失败后可以安全重复执行。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_DIR = PROJECT_ROOT / "runs" / "features"
BASELINE_DIR = PROJECT_ROOT / "work" / "registration_backups" / "20260828_100844"
BACKUP_ROOT = PROJECT_ROOT / "work" / "demo_state_backups"

DEMO_SKU = "bottle07"
DEMO_MODEL_CLASS = "BOTTLE_07_asm milktea"
REQUIRED_BASELINE_FILES = {
    "bottle_embeddings.npy",
    "bottle_labels.json",
    "bottle_centers.npy",
    "bottle_classes.json",
    "bottle_metadata.json",
    "bottle_stats.json",
}

sys.path.insert(0, str(PROJECT_ROOT))
from database.goods_dao import GoodsDao  # noqa: E402


def _json_default(value: Any) -> str | float:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat(sep=" ")
    return str(value)


def _bottle_files(directory: Path) -> list[Path]:
    resolved_dir = directory.resolve()
    files = sorted(directory.glob("bottle_*"))
    for path in files:
        if not path.is_file() or path.resolve().parent != resolved_dir:
            raise RuntimeError(f"发现非预期 bottle 路径，已停止：{path}")
    return files


def _read_feature_state(directory: Path) -> dict[str, Any]:
    embeddings_path = directory / "bottle_embeddings.npy"
    labels_path = directory / "bottle_labels.json"
    classes_path = directory / "bottle_classes.json"
    metadata_path = directory / "bottle_metadata.json"
    missing = [
        path.name
        for path in (embeddings_path, labels_path, classes_path, metadata_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"bottle 特征库缺少文件：{', '.join(missing)}")

    embeddings = np.load(embeddings_path, mmap_mode="r")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise ValueError(f"bottle_embeddings.npy 形状异常：{embeddings.shape}")
    if embeddings.shape[0] != len(labels):
        raise ValueError("bottle embeddings 与 labels 数量不一致")
    return {
        "samples": int(embeddings.shape[0]),
        "classes": list(classes),
        "registered_classes": list(metadata.get("registered_classes", [])),
        "has_prototypes": (
            (directory / "bottle_prototypes.npy").is_file()
            and (directory / "bottle_prototype_labels.json").is_file()
        ),
    }


def _validate_baseline() -> dict[str, Any]:
    if not BASELINE_DIR.is_dir():
        raise FileNotFoundError(f"找不到 24-SKU 基线备份：{BASELINE_DIR}")
    names = {path.name for path in _bottle_files(BASELINE_DIR)}
    missing = sorted(REQUIRED_BASELINE_FILES - names)
    if missing:
        raise FileNotFoundError(f"24-SKU 基线备份不完整：{', '.join(missing)}")
    state = _read_feature_state(BASELINE_DIR)
    if len(state["classes"]) != 6 or DEMO_MODEL_CLASS in state["classes"]:
        raise ValueError("指定备份不是预期的 6 类 bottle / 24-SKU 初始基线")
    return state


def _database_state(dao: GoodsDao) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    rows = dao.list_all(active_only=False)
    demo_row = next((row for row in rows if row.get("sku_code") == DEMO_SKU), None)
    return rows, demo_row


def _make_backup(dao: GoodsDao) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_dir = BACKUP_ROOT / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    for source in _bottle_files(FEATURES_DIR):
        shutil.copy2(source, backup_dir / source.name)
    rows, _ = _database_state(dao)
    (backup_dir / "database_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (backup_dir / "README.txt").write_text(
        "本目录由 prepare_registration_demo.py 在重置前自动创建。\n"
        "包含重置前的 bottle 特征文件和 products 全表快照。\n",
        encoding="utf-8",
    )
    return backup_dir


def _restore_feature_files(source_dir: Path) -> None:
    source_files = _bottle_files(source_dir)
    if not source_files:
        raise FileNotFoundError(f"恢复来源中没有 bottle 文件：{source_dir}")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    for current in _bottle_files(FEATURES_DIR):
        current.unlink()
    for source in source_files:
        shutil.copy2(source, FEATURES_DIR / source.name)


def _restore_demo_row(dao: GoodsDao, row: dict[str, Any] | None) -> None:
    if row is None or dao.get_by_sku_code(DEMO_SKU) is not None:
        return
    dao.add_goods(
        sku_code=str(row["sku_code"]),
        product_name=str(row["product_name"]),
        package_type=str(row["package_type"]),
        unit_price=float(row["unit_price"]),
        model_class=str(row["model_class"]),
        barcode=row.get("barcode"),
        unit_weight_g=float(row["unit_weight_g"]) if row.get("unit_weight_g") is not None else None,
        remark=row.get("remark"),
    )
    updates: dict[str, Any] = {}
    if row.get("feature_index") is not None:
        updates["feature_index"] = row["feature_index"]
    if not bool(row.get("is_active", 1)):
        updates["is_active"] = 0
    if updates:
        dao.update_goods(DEMO_SKU, **updates)


def _print_status(dao: GoodsDao) -> dict[str, Any]:
    feature_state = _read_feature_state(FEATURES_DIR)
    active_rows = dao.list_all(active_only=True)
    demo_row = dao.get_by_sku_code(DEMO_SKU)
    print(f"有效商品：{len(active_rows)} 条")
    print(
        "bottle 特征库："
        f"{len(feature_state['classes'])} 类 / {feature_state['samples']} 条特征 / "
        f"在线注册类 {len(feature_state['registered_classes'])} 个"
    )
    print(f"阿萨姆数据库记录：{'已注册' if demo_row else '未注册'}")
    print(f"阿萨姆特征类别：{'已写入' if DEMO_MODEL_CLASS in feature_state['classes'] else '未写入'}")
    return {
        "active_products": len(active_rows),
        "feature_state": feature_state,
        "demo_row": demo_row,
    }


def reset_demo() -> None:
    baseline = _validate_baseline()
    dao = GoodsDao()
    backup_dir: Path | None = None
    old_demo_row: dict[str, Any] | None = None
    try:
        if dao.helper.connection is None:
            raise ConnectionError("MySQL 未连接，不能安全同步数据库与特征库")
        _, old_demo_row = _database_state(dao)
        backup_dir = _make_backup(dao)
        try:
            _restore_feature_files(BASELINE_DIR)
            if old_demo_row is not None:
                dao.delete_goods(DEMO_SKU, soft=False)

            status = _print_status(dao)
            if status["active_products"] != 24:
                raise RuntimeError(f"重置后有效商品应为24条，实际为{status['active_products']}条")
            feature_state = status["feature_state"]
            if len(feature_state["classes"]) != 6 or DEMO_MODEL_CLASS in feature_state["classes"]:
                raise RuntimeError("重置后 bottle 特征库仍包含阿萨姆")
            if status["demo_row"] is not None:
                raise RuntimeError("重置后 MySQL 仍包含有效的 bottle07")
        except Exception:
            _restore_feature_files(backup_dir)
            _restore_demo_row(dao, old_demo_row)
            raise

        print(f"24-SKU 基线：{baseline['samples']} 条 bottle 特征 / 6 类")
        print(f"重置前快照：{backup_dir}")
        print("演示初始状态已就绪：第一次播放验证视频时，阿萨姆应显示为未注册商品。")
    finally:
        dao.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="准备阿萨姆在线注册答辩演示")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="备份当前状态并恢复到24-SKU、阿萨姆未注册的演示初始状态",
    )
    args = parser.parse_args()
    if args.reset:
        reset_demo()
        return

    dao = GoodsDao()
    try:
        _print_status(dao)
        print("这里只检查状态；录制前需要恢复时请加 --reset。")
    finally:
        dao.close()


if __name__ == "__main__":
    main()
