# -*- coding: utf-8 -*-
"""
RSNA 真实骨龄验证：跑完整流水线 → 与 RSNA 标签计算 MAE

数据：
  真值   : data/rsna_gt.csv（id, boneage_months, sex，从 RSNA train.csv 提取）
  图片   : datasets/detection_pre/（全部 881 张已 CLAHE 预处理，含 train/val）

流程：对每张图按真实性别跑 pipeline → 预测骨龄(年) → 转月 → vs 真实骨龄(月)
指标：MAE(月)、RMSE、分性别/分年龄段的 MAE

用法：
    python validate_rsna.py --limit 20        # 先小样本试跑
    python validate_rsna.py                   # 全量 881 张
    python validate_rsna.py --resume          # 断点续跑（跳过已有结果）
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

import config
from pipeline import Pipeline

GT_PATH = config.BAA_DIR / "data" / "rsna_gt.csv"
OUT_CSV = config.BAA_DIR / "data" / "rsna_results.csv"


def load_gt():
    gt = {}
    with open(GT_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt[int(r["id"])] = {"months": int(r["boneage_months"]), "sex": r["sex"]}
    return gt


def main():
    parser = argparse.ArgumentParser(description="RSNA 真实骨龄 MAE 验证")
    parser.add_argument("--limit", type=int, default=None, help="只验证前 N 张（试跑）")
    parser.add_argument("--resume", action="store_true", help="跳过已有结果（断点续跑）")
    args = parser.parse_args()

    gt = load_gt()
    print(f"真值条数: {len(gt)}")

    # 收集全部检测集图片（train + val = 881）
    img_paths = []
    for split in ("train", "val"):
        img_paths += sorted((config.DETECTION_PRE / "images" / split).glob("*.png"))
    img_paths.sort(key=lambda p: int(p.stem))
    print(f"图片总数: {len(img_paths)}")

    if args.limit:
        img_paths = img_paths[:args.limit]

    # 已有结果（续跑）
    done = {}
    if args.resume and OUT_CSV.exists():
        with open(OUT_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done[int(r["id"])] = r
        print(f"已有结果: {len(done)} 张，续跑跳过")

    pipe = Pipeline()

    results = list(done.values())
    t0 = time.time()
    for i, p in enumerate(img_paths):
        iid = int(p.stem)
        if iid in done:
            continue
        if iid not in gt:
            print(f"[跳过] {iid} 无真值")
            continue
        try:
            res = pipe.predict(p, sex=gt[iid]["sex"])
            pred_years = res["bone_age_years"]
            pred_months = round(pred_years * 12.0, 2) if pred_years is not None else None
            results.append({
                "id": iid, "sex": gt[iid]["sex"], "true_months": gt[iid]["months"],
                "pred_months": pred_months, "total_score": res["total_score"],
                "n_bones": res["n_bones"], "missing": ";".join(res["missing"]),
            })
        except Exception as e:
            print(f"[错误] {iid}: {e}")
            results.append({"id": iid, "sex": gt[iid]["sex"], "true_months": gt[iid]["months"],
                            "pred_months": None, "total_score": None,
                            "n_bones": 0, "missing": f"ERROR:{e}"})
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  进度 {i+1}/{len(img_paths)}  已用 {elapsed:.0f}s")
            _save(results)

    _save(results)
    _report(results)


def _save(results):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id", "sex", "true_months", "pred_months",
                                          "total_score", "n_bones", "missing"])
        w.writeheader()
        w.writerows(results)


def _report(results):
    print("\n" + "=" * 60)
    valid = [r for r in results if r["pred_months"] is not None]
    if not valid:
        print("无有效预测结果")
        return
    true = np.array([r["true_months"] for r in valid], dtype=float)
    pred = np.array([r["pred_months"] for r in valid], dtype=float)
    err = np.abs(pred - true)
    mae = err.mean()
    rmse = np.sqrt(((pred - true) ** 2).mean())
    print(f"有效样本: {len(valid)}/{len(results)}")
    print(f"MAE  = {mae:.2f} 月 ({mae/12:.2f} 岁)")
    print(f"RMSE = {rmse:.2f} 月")
    print(f"误差分位: P50={np.percentile(err,50):.1f}  P90={np.percentile(err,90):.1f}  最大={err.max():.1f}")
    print()
    for sex in ("male", "female"):
        sub = [r for r in valid if r["sex"] == sex]
        if not sub:
            continue
        e = np.abs(np.array([r["pred_months"] for r in sub]) -
                   np.array([r["true_months"] for r in sub], dtype=float))
        print(f"  {sex}: n={len(sub)}  MAE={e.mean():.2f} 月")
    print()
    # 分年龄段
    print("分年龄段 MAE:")
    for lo in range(5, 18):
        sub = [r for r in valid if lo * 12 <= r["true_months"] < (lo + 1) * 12]
        if sub:
            e = np.abs(np.array([r["pred_months"] for r in sub]) -
                       np.array([r["true_months"] for r in sub], dtype=float))
            print(f"  {lo}-{lo+1}岁: n={len(sub):>3}  MAE={e.mean():6.2f} 月")


if __name__ == "__main__":
    main()
