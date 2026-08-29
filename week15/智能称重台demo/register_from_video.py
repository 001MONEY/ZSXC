r"""从一段单商品多姿态视频注册 SKU，并立即验证开放集检索结果。

默认只做注册前诊断，不修改数据库和特征库；显式传入 --apply 才正式注册。

示例：
    D:\project\step1\env\python.exe register_from_video.py ^
      --video "video\asm milktea.mp4" --group bottle ^
      --sku bottle07 --model-class "BOTTLE_07_asm milktea" ^
      --name 阿萨姆奶茶 --price 3.0 --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from feature_library_updater import (  # noqa: E402
    GROUPS,
    collect_video_crops,
    register_sku,
    select_registration_crops,
)
from database.goods_dao import GoodsDao  # noqa: E402
from onnx_engine import OnnxFeatureLibrary, retrieval_match_onnx  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用多姿态视频注册新商品。")
    parser.add_argument("--video", required=True, help="注册视频路径。")
    parser.add_argument(
        "--verify-video",
        help="可选的独立单商品验证视频；多商品场景请使用 pipeline_demo.py 验证购物车。",
    )
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--sku", help="业务 SKU，例如 bottle07。")
    parser.add_argument("--model-class", help="检索类别，例如 BOTTLE_07_asm milktea。")
    parser.add_argument("--name", help="商品中文名称。")
    parser.add_argument("--price", type=float, help="商品单价。")
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--apply", action="store_true", help="正式写数据库和特征库。")
    return parser.parse_args()


def evaluate(
    video: str | Path,
    group: str,
    expected_class: str | None,
    sample_fps: float,
) -> dict:
    raw_crops, video_stats = collect_video_crops(
        video,
        group,
        sample_fps=sample_fps,
    )
    crops, selection_stats = select_registration_crops(raw_crops)
    library = OnnxFeatureLibrary()
    rows = []
    for crop in crops:
        top1_class, top1, top2_class, top2, margin = retrieval_match_onnx(
            crop,
            library,
            group,
        )
        similarity_threshold, margin_threshold = library.thresholds_for(
            group,
            top1_class,
        )
        rows.append(
            {
                "top1_class": top1_class,
                "similarity": float(top1),
                "top2_class": top2_class,
                "top2_similarity": float(top2),
                "margin": float(margin),
                "similarity_threshold": similarity_threshold,
                "margin_threshold": margin_threshold,
                "passes_open_set": bool(
                    top1 >= similarity_threshold and margin >= margin_threshold
                ),
                "expected": bool(expected_class and top1_class == expected_class),
            }
        )
    expected_hits = sum(row["expected"] for row in rows)
    accepted_hits = sum(
        row["expected"] and row["passes_open_set"] for row in rows
    )
    return {
        "video": video_stats,
        "selection": selection_stats,
        "evaluated": len(rows),
        "expected_top1": expected_hits,
        "expected_accepted": accepted_hits,
        "top1_rate": expected_hits / len(rows) if rows else 0.0,
        "accepted_rate": accepted_hits / len(rows) if rows else 0.0,
        "similarity_min": min((row["similarity"] for row in rows), default=0.0),
        "similarity_mean": sum(row["similarity"] for row in rows) / len(rows) if rows else 0.0,
        "margin_min": min((row["margin"] for row in rows), default=0.0),
        "margin_mean": sum(row["margin"] for row in rows) / len(rows) if rows else 0.0,
        "rows": rows,
    }


def create_persistent_backup(group: str) -> Path:
    """正式注册前保存可人工恢复的特征文件和数据库快照。"""
    backup = (
        PROJECT_ROOT
        / "work"
        / "registration_backups"
        / time.strftime("%Y%m%d_%H%M%S")
    )
    backup.mkdir(parents=True, exist_ok=False)
    for path in (PROJECT_ROOT / "runs" / "features").glob(f"{group}_*"):
        shutil.copy2(path, backup / path.name)
    dao = GoodsDao()
    try:
        rows = dao.list_all(active_only=False)
    finally:
        dao.close()
    (backup / "database_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return backup


def main() -> int:
    args = parse_args()
    if args.apply and not all((args.sku, args.model_class, args.name, args.price is not None)):
        raise ValueError("--apply 时必须同时提供 --sku、--model-class、--name 和 --price。")

    print("[1/3] 提取注册视频样本…")
    raw_crops, source_stats = collect_video_crops(
        args.video,
        args.group,
        sample_fps=args.sample_fps,
    )
    selected_crops, selection_stats = select_registration_crops(raw_crops)
    print(json.dumps({"video": source_stats, "selection": selection_stats}, ensure_ascii=False, indent=2))

    if not args.apply:
        print("\n[只读诊断] 未传 --apply，数据库和特征库均未修改。")
        before = evaluate(args.verify_video or args.video, args.group, None, args.sample_fps)
        print(json.dumps({key: value for key, value in before.items() if key != "rows"}, ensure_ascii=False, indent=2))
        return 0

    print("[2/3] 写入商品数据库、特征样本和多姿态原型…")
    backup_path = create_persistent_backup(args.group)
    print(f"注册前备份：{backup_path}")
    registration = register_sku(
        group=args.group,
        sku=args.sku,
        product_name=args.name,
        unit_price=args.price,
        crops_bgr=raw_crops,
        model_class=args.model_class,
    )

    print("[3/3] 重新加载特征库并立即验证…")
    evaluation = evaluate(
        args.verify_video or args.video,
        args.group,
        args.model_class,
        args.sample_fps,
    )
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "registration_source": source_stats,
        "backup": str(backup_path),
        "registration": registration,
        "verification": {key: value for key, value in evaluation.items() if key != "rows"},
    }
    report_dir = PROJECT_ROOT / "runs" / "registration"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{args.sku}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告：{report_path}")

    if evaluation["accepted_rate"] < 0.80:
        print("[FAIL] 注册后通过开放集阈值的命中率低于80%。")
        return 2
    print("[OK] 新商品注册后可立即通过开放集检索。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
