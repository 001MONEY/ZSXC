# -*- coding: utf-8 -*-
"""
全量验证集正式评估（真实泛化效果）

适用范围: 6-19 岁（排除学龄前）——训练集无 0-5 岁样本，低龄为外推区，不计入正式指标。

方案 A（严格独立，推荐报告值）:
  训练: 881 张 handbone 训练图特征（features_train_ordinal.csv）
  测试: 全部 1425 张 RSNA 验证图中 6 岁+ 部分（features_val_ordinal.csv），模型从未见过
  拟合: GradientBoosting（与 calibrate.py 相同超参数），imputer 仅用训练集统计

方案 B（生产模型参考）:
  加载现有 models/bone_age_regressor.pkl（2306 张合并训练，含部分验证图）
  对 6 岁+ 验证图预测 —— 有轻微数据泄漏，偏乐观，仅作对照

用法：
    python eval_full_val.py            # 正式评估（6岁+，默认）
    python eval_full_val.py --min-age 0   # 含全部年龄段
    python eval_full_val.py --save     # 保存预测 CSV
"""
import argparse
import csv
from pathlib import Path

import numpy as np

BAA = Path(__file__).parent
RUS_13 = ["Radius", "Ulna", "MCP-1", "MCP-3", "MCP-5",
          "PIP-1", "PIP-3", "PIP-5", "MIP-3", "MIP-5",
          "DIP-1", "DIP-3", "DIP-5"]
FEAT_TR = BAA / "data" / "features_train_ordinal.csv"
FEAT_VA = BAA / "data" / "features_val_ordinal.csv"
VAL_LABELS = BAA / "data" / "rsna_val_labels.csv"


def load_features(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    ids = [int(r["id"]) for r in rows]
    true = np.array([float(r["true_months"]) for r in rows])
    X = np.array([[float(r[b]) if r[b] not in ("", "None") else np.nan for b in RUS_13]
                  for r in rows])
    return ids, X, true


def load_sex_map():
    m = {}
    with open(VAL_LABELS, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            m[int(r["Image ID"])] = "male" if r["male"].upper() == "TRUE" else "female"
    return m


def report(name, pred, y, ids, sex_map):
    err = np.abs(pred - y)
    mae = err.mean()
    rmse = np.sqrt(((pred - y) ** 2).mean())
    corr = np.corrcoef(pred, y)[0, 1]
    print("=" * 62)
    print(f"[方案] {name}")
    print(f"样本数: {len(y)}   MAE={mae:.2f} 月 ({mae/12:.2f} 岁)   RMSE={rmse:.2f} 月")
    print(f"误差分位: P50={np.percentile(err,50):.1f}  P90={np.percentile(err,90):.1f}  最大={err.max():.1f}")
    print(f"预测 vs 真实 相关: {corr:.3f}")
    print("分性别:")
    for sex in ("male", "female"):
        idx = [i for i, sid in enumerate(ids) if sex_map.get(sid) == sex]
        if idx:
            print(f"  {sex}: n={len(idx):>4}  MAE={err[idx].mean():6.2f} 月")
    print("分年龄段 MAE:")
    for lo in range(0, 19):
        idx = [i for i in range(len(y)) if lo * 12 <= y[i] < (lo + 1) * 12]
        if idx:
            print(f"  {lo:2d}-{lo+1}岁: n={len(idx):>3}  MAE={err[idx].mean():6.2f} 月")
    return mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="保存验证集预测 CSV")
    parser.add_argument("--min-age", type=int, default=72,
                        help="最小月龄（默认 72=排除学龄前 0-6 岁）")
    args = parser.parse_args()

    ids_tr, X_tr, y_tr = load_features(FEAT_TR)
    ids_va, X_va, y_va = load_features(FEAT_VA)
    sex_map = load_sex_map()
    print(f"训练特征: {X_tr.shape}   验证特征: {X_va.shape}")

    # 适用范围过滤（排除学龄前）
    keep = y_va >= args.min_age
    n_drop = (~keep).sum()
    if n_drop:
        print(f"[过滤] 排除 {args.min_age/12:.0f} 岁以下 {n_drop} 张，评估范围 n={keep.sum()}")
    ids_va, X_va, y_va = (np.array(ids_va)[keep], X_va[keep], y_va[keep])

    # ---- 方案 A：严格独立（仅 881 训练图拟合） ----
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import GradientBoostingRegressor
    imp = SimpleImputer(strategy="mean").fit(X_tr)      # 只用训练集统计
    X_tr_i, X_va_i = imp.transform(X_tr), imp.transform(X_va)
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                      learning_rate=0.05, random_state=42)
    model.fit(X_tr_i, y_tr)
    pred_a = model.predict(X_va_i).clip(0, 240)
    mae_a = report(f"A: 严格独立 (881训练 -> {len(y_va)}张测试)", pred_a, y_va, ids_va, sex_map)

    # ---- 方案 B：生产模型对照（2306 合并训练，有轻微泄漏） ----
    import joblib
    pkl = joblib.load(BAA / "models" / "bone_age_regressor.pkl")
    X_va_b = pkl["imputer"].transform(X_va)
    pred_b = pkl["model"].predict(X_va_b).clip(0, 240)
    mae_b = report("B: 生产模型对照 (2306合并训练, 偏乐观)", pred_b, y_va, ids_va, sex_map)

    print("=" * 62)
    print(f"结论: 严格独立 MAE = {mae_a:.2f} 月 ({mae_a/12:.2f} 岁)   "
          f"生产模型对照 MAE = {mae_b:.2f} 月")

    if args.save:
        out = BAA / "data" / "full_val_predictions.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["id", "true_months", "pred_A_strict", "pred_B_prod",
                        "abs_err_A", "abs_err_B", "sex"])
            for i, sid in enumerate(ids_va):
                w.writerow([sid, y_va[i], round(pred_a[i], 1), round(pred_b[i], 1),
                            round(abs(pred_a[i] - y_va[i]), 1),
                            round(abs(pred_b[i] - y_va[i]), 1),
                            sex_map.get(sid, "")])
        print(f"[OK] 验证集预测已保存 -> {out}")


if __name__ == "__main__":
    main()
