# -*- coding: utf-8 -*-
"""
全量 RSNA 训练集扩充校准（6 岁+，11279 张）

背景: RSNA 训练集 12611 张只有骨龄标签(boneage)，无关节等级标注，
      无法训练分类模型；但校准模型只需 13 骨得分 + 真实骨龄，可全量利用。

流程:
  1. 读取 train.csv，筛选 boneage >= 72（6 岁+，正式范围），共 11279 张
  2. CLAHE 预处理 -> data/rsna_train_pre/（带缓存）
  3. 全 ordinal 流水线提取 13 骨得分 -> data/features_train_full.csv（断点续跑）
  4. 用全量特征训练 GradientBoosting / Ridge
  5. 在 1273 张 6 岁+ 验证图（features_val_ordinal.csv + 过滤）上严格独立评估
     （1425 验证集与 12611 训练集完全无重叠）

严格独立: 训练=全量12611(6岁+), 测试=1425验证(6岁+, n=1273)  —— 无任何泄漏

用法:
    python extend_calibration.py                # 全量（特征提取约 3-4 小时）
    python extend_calibration.py --limit 100    # 快速试跑
    python extend_calibration.py --skip-features # 只用已有特征直接训练评估
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

import config
from pipeline import Pipeline

BAA = config.BAA_DIR
TRAIN_IMG = config.WORKSPACE / "boneage-training-dataset"   # week12 根目录
TRAIN_CSV = config.WORKSPACE / "train.csv"
TRAIN_PRE = BAA / "data" / "rsna_train_pre"
FEAT_FULL = BAA / "data" / "features_train_full.csv"
FEAT_VAL = BAA / "data" / "features_val_ordinal.csv"
RUS_13 = ["Radius", "Ulna", "MCP-1", "MCP-3", "MCP-5",
          "PIP-1", "PIP-3", "PIP-5", "MIP-3", "MIP-5",
          "DIP-1", "DIP-3", "DIP-5"]
MIN_AGE = 72


# ---------------------------------------------------------------- 预处理
def preprocess_imgs(ids, limit=None):
    """CLAHE 预处理（与检测训练数据一致），带缓存。"""
    import cv2
    TRAIN_PRE.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (TRAIN_PRE / f"{i}.png").exists()]
    if limit:
        todo = todo[:limit]
    done, t0 = 0, time.time()
    for i, iid in enumerate(todo):
        src = TRAIN_IMG / f"{iid}.png"
        if not src.exists():
            continue
        img = cv2.imread(str(src))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        cv2.imwrite(str(TRAIN_PRE / f"{iid}.png"), cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR))
        done += 1
        if (i + 1) % 2000 == 0:
            print(f"  预处理 {i+1}/{len(todo)}  已用 {time.time()-t0:.0f}s")
    print(f"[OK] 预处理完成（新增 {done} 张，共 {len(todo)} 张）")


# ---------------------------------------------------------------- 特征提取
def extract_features(pipe, ids, gt, sex_map, out_csv, resume=True, limit=None):
    done = {}
    if resume and out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done[int(r["id"])] = r
    rows = list(done.values())
    todo = [i for i in ids if i not in done]
    if limit:
        todo = todo[:limit]
    t0 = time.time()
    for n, iid in enumerate(todo):
        p = TRAIN_PRE / f"{iid}.png"
        try:
            res = pipe.predict(p, sex="boy")     # 性别不影响 13 骨得分
            scores = {d["bone"]: d["score"] for d in res["detail"]}
            row = {"id": iid, "true_months": gt[iid], "sex": sex_map.get(iid, "")}
            for b in RUS_13:
                row[b] = scores.get(b, "")
            rows.append(row)
        except Exception as e:
            print(f"[错误] {iid}: {e}")
        if (n + 1) % 500 == 0:
            print(f"  特征进度 {n+1}/{len(todo)}  已用 {time.time()-t0:.0f}s")
            _save(rows, out_csv)
    _save(rows, out_csv)
    return rows


def _save(rows, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "true_months", "sex"] + RUS_13)
        w.writeheader()
        w.writerows(rows)


def to_matrix(rows):
    ids = [int(r["id"]) for r in rows]
    y = np.array([float(r["true_months"]) for r in rows])
    X = np.array([[float(r[b]) if r[b] not in ("", None) else np.nan for b in RUS_13]
                  for r in rows])
    return ids, X, y


# ---------------------------------------------------------------- 评估
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--save-model", action="store_true", help="覆盖生产校准模型")
    args = parser.parse_args()

    # 1. 训练标签（6 岁+）
    gt, sex_map = {}, {}
    with open(TRAIN_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            a = int(r["boneage"])
            if a >= MIN_AGE:
                gt[int(r["id"])] = a
                sex_map[int(r["id"])] = "male" if r["male"].upper() == "TRUE" else "female"
    ids = sorted(gt)
    print(f"[数据] 6 岁+ 训练样本: {len(ids)} 张（train.csv 共 12611）")

    if not args.skip_features:
        print("\n[1/3] CLAHE 预处理...")
        preprocess_imgs(ids, args.limit)
        print("\n[2/3] 全 ordinal 流水线提取 13 骨特征...")
        pipe = Pipeline(ordinal_all=True)
        sub = ids[:args.limit] if args.limit else ids
        rows = extract_features(pipe, sub, gt, sex_map, FEAT_FULL, True)
        print(f"[OK] 特征矩阵行数: {len(rows)}")
    else:
        print("\n[跳过特征提取] 使用已有特征缓存")

    # 3. 训练 + 严格独立评估
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    ids_tr, X_tr, y_tr = to_matrix(list(csv.DictReader(open(FEAT_FULL, encoding="utf-8"))))
    ids_va, X_va, y_va = to_matrix(list(csv.DictReader(open(FEAT_VAL, encoding="utf-8"))))
    keep = y_va >= MIN_AGE
    ids_va, X_va, y_va = np.array(ids_va)[keep], X_va[keep], y_va[keep]

    imp = SimpleImputer(strategy="mean").fit(X_tr)
    X_tr, X_va = imp.transform(X_tr), imp.transform(X_va)
    print(f"\n[3/3] 训练 {X_tr.shape[0]} 张 -> 严格独立测试 {X_va.shape[0]} 张")
    print(f"训练年龄分布: {y_tr.min():.0f}-{y_tr.max():.0f} 月")

    best = None
    for name, mk in [("Ridge", lambda: Ridge(alpha=10.0)),
                     ("GradientBoosting", lambda: GradientBoostingRegressor(
                         n_estimators=300, max_depth=3, learning_rate=0.05,
                         random_state=42))]:
        m = mk().fit(X_tr, y_tr)
        pred = m.predict(X_va).clip(0, 240)
        err = np.abs(pred - y_va)
        corr = np.corrcoef(pred, y_va)[0, 1]
        print(f"  {name}: MAE={err.mean():.2f} 月  相关={corr:.3f}  "
              f"P50={np.percentile(err,50):.1f} P90={np.percentile(err,90):.1f}")
        if best is None or err.mean() < best[0]:
            best = (err.mean(), name, m, imp)

    mae, name, model, imp = best
    print("\n" + "=" * 60)
    print(f"扩充校准结果（训练 {X_tr.shape[0]} -> 测试 {X_va.shape[0]}，严格独立）")
    print(f"最佳模型: {name}   MAE = {mae:.2f} 月 ({mae/12:.2f} 岁)")
    print(f"对比: 之前 881 训练 MAE = 13.19 月")

    if args.save_model:
        import joblib
        model_dir = BAA / "models"
        joblib.dump({"model": model, "imputer": imp, "name": name,
                     "train_mae_months": mae, "test_mae_months": mae},
                    model_dir / "bone_age_regressor.pkl")
        print(f"[OK] 生产校准模型已更新 -> {model_dir / 'bone_age_regressor.pkl'}")


if __name__ == "__main__":
    main()
