r"""注册功能闭环测试：备份特征库 → 注册测试商品 → 验证检索/数据库 → 恢复清理。"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
from onnx_engine import OnnxFeatureLibrary, retrieval_match_onnx  # noqa: E402
from feature_library_updater import register_sku, suggest_model_class, suggest_sku  # noqa: E402
from database.goods_dao import GoodsDao  # noqa: E402

FEATURES = PROJECT_ROOT / "runs" / "features"
TEST_SKU = "bag88test"
TEST_MODEL = "BAG_88_test_product"


def main() -> int:
    # 1) 备份 bag 特征库
    backup = Path(tempfile.mkdtemp(prefix="feat_backup_"))
    for f in FEATURES.glob("bag_*"):
        shutil.copy2(f, backup / f.name)
    print(f"[1] 特征库已备份：{backup}")

    # 2) 造样本（3 张 bag 图）
    src = Path("classification_dataset_from_videos") / "bag" / "val" / "BAG_01_kebike_chips"
    images = sorted(src.glob("*.jpg"))[:3]
    if not images:
        print("[FAIL] 无样本图片")
        return 1
    crops = [cv2.imread(str(p)) for p in images]
    print(f"[2] 样本：{len(crops)} 张")

    # 3) 注册
    dao = GoodsDao()
    print(f"[3] 建议SKU={suggest_sku(dao, 'bag')}  建议分类名={suggest_model_class('bag', 'bag07', '新薯片')}")
    result = register_sku("bag", TEST_SKU, "测试商品", 9.9, crops, model_class=TEST_MODEL, dao=dao)
    print(f"[3] 注册结果：{result}")

    # 4) 检索验证
    lib = OnnxFeatureLibrary()
    model_class, sim, top2, sim2, margin = retrieval_match_onnx(crops[0], lib, "bag")
    print(f"[4] 检索：{model_class}  sim={sim:.4f}  margin={margin:.4f}")
    assert model_class == TEST_MODEL, "检索未命中新注册类！"

    # 5) 数据库验证
    goods = dao.get_by_model_class(TEST_MODEL)
    print(f"[5] 数据库：{goods['product_name']} ¥{goods['unit_price']} index={goods['feature_index']}")
    assert goods is not None and goods["feature_index"] == f"lib1_center{result['index']}_{TEST_MODEL}"

    # 6) 清理：删数据库记录 + 恢复特征库
    dao.delete_goods(TEST_SKU, soft=False)
    dao.close()
    for f in FEATURES.glob("bag_*"):
        f.unlink()
    for f in backup.glob("bag_*"):
        shutil.copy2(f, FEATURES / f.name)
    print("[6] 已清理测试数据并恢复特征库")

    print("\n[OK] 注册闭环测试全部通过！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
