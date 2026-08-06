# -*- coding: utf-8 -*-
"""
数据驱动骨龄校准：13 根骨得分 → 骨龄回归

背景：RUS-CHN 表硬查发现部分骨头等级与真实年龄脱节（Radius/PIP），
因此改用真实标签学习"13 骨得分 → 骨龄"的映射。

数据划分（无泄漏）：
  训练: 881 张 handbone 训练图（RSNA 标签, data/rsna_gt.csv）
  测试: 1425 张 RSNA 验证图（data/rsna_val_labels.csv, 完全独立）

流程：
  1. 预处理验证图（CLAHE）→ data/rsna_val_pre/
  2. 对每张图跑流水线 → 13 骨得分特征（带缓存，支持断点续跑）
  3. Ridge 回归：13 骨得分 → 骨龄(月)，在 881 训练图上拟合
  4. 在 1425 验证图上评估 MAE/RMSE（分性别、分年龄段）

用法：
    python calibrate.py --limit-train 100 --limit-val 50   # 快速试跑
    python calibrate.py                                    # 全量（默认分类模型特征）
    python calibrate.py --ordinal                          # 用全部 9 关节 ordinal 模型特征
    python calibrate.py --preprocess-only                  # 只预处理验证图
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

import config
from pipeline import Pipeline

VAL_IMG = config.BAA_DIR / "data" / "rsna_val_images"
VAL_PRE = config.BAA_DIR / "data" / "rsna_val_pre"
VAL_LABEL_CSV = config.BAA_DIR / "data" / "rsna_val_labels.csv"
TRAIN_GT = config.BAA_DIR / "data" / "rsna_gt.csv"
FEAT_TRAIN = config.BAA_DIR / "data" / "features_train.csv"
FEAT_VAL = config.BAA_DIR / "data" / "features_val.csv"
FEAT_TRAIN_ORD = config.BAA_DIR / "data" / "features_train_ordinal.csv"
FEAT_VAL_ORD = config.BAA_DIR / "data" / "features_val_ordinal.csv"

RUS_13 = ["Radius", "Ulna", "MCP-1", "MCP-3", "MCP-5",
          "PIP-1", "PIP-3", "PIP-5", "MIP-3", "MIP-5",
          "DIP-1", "DIP-3", "DIP-5"]


# ---------------------------------------------------------------- 预处理
def preprocess_validation_images():
    """对验证集图片做 CLAHE 预处理（与检测训练数据一致）。"""
    import cv2
    VAL_PRE.mkdir(parents=True, exist_ok=True)
    imgs = sorted(VAL_IMG.glob("*.png"))
    done = 0
    for p in imgs:
        out = VAL_PRE / p.name
        if out.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        cv2.imwrite(str(out), cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR))
        done += 1
    print(f"[OK] 验证图预处理完成（新增 {done} 张，共 {len(imgs)} 张）")


# ---------------------------------------------------------------- 特征提取
def load_val_labels():
    labels = {}
    with open(VAL_LABEL_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            labels[int(r["Image ID"])] = int(r["Bone Age (months)"])
    return labels


def extract_features(pipe, img_paths, gt, out_csv, label_key, resume=True):
    """对图片跑流水线，提取 13 骨得分特征。带 CSV 缓存。"""
    done = {}
    if resume and out_csv.exists():
        with open(out_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done[int(r["id"])] = r

    rows = list(done.values())
    t0 = time.time()
    for i, p in enumerate(img_paths):
        iid = int(p.stem)
        if iid in done:
            continue
        if iid not in gt:
            continue
        try:
            res = pipe.predict(p, sex="boy")   # 性别不影响 13 骨得分特征
            scores = {d["bone"]: d["score"] for d in res["detail"]}
            row = {"id": iid, "true_months": gt[iid]}
            for b in RUS_13:
                row[b] = scores.get(b, "")
            rows.append(row)
        except Exception as e:
            print(f"[错误] {iid}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {label_key} 进度 {i+1}/{len(img_paths)}  已用 {time.time()-t0:.0f}s")
            _save_rows(rows, out_csv)
    _save_rows(rows, out_csv)
    return rows


def _save_rows(rows, out_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "true_months"] + RUS_13)
        w.writeheader()
        w.writerows(rows)


def to_matrix(rows):
    ids = [int(r["id"]) for r in rows]
    true = np.array([float(r["true_months"]) for r in rows])
    X = np.array([[float(r[b]) if r[b] not in ("", None) else np.nan for b in RUS_13]
                  for r in rows])
    return ids, X, true


# ---------------------------------------------------------------- 校准评估
def train_and_eval(X, y, ids_all, sex_map, test_frac=0.15):
    """合并全部带标签数据，按年龄分层划分训练/测试，训练并评估。"""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    imp = SimpleImputer(strategy="mean")
    X = imp.fit_transform(X)

    # 按年龄分箱分层划分
    bins = np.clip((y // 12).astype(int), 0, 18)
    tr_idx, te_idx = train_test_split(np.arange(len(y)), test_size=test_frac,
                                      random_state=42, stratify=bins)

    best = None
    for name, make_model in [
        ("Ridge", lambda: Ridge(alpha=10.0)),
        ("GradientBoosting", lambda: GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)),
    ]:
        model = make_model()
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[te_idx]).clip(0, 240)
        mae = np.abs(pred - y[te_idx]).mean()
        corr = np.corrcoef(pred, y[te_idx])[0, 1]
        print(f"  {name}: MAE={mae:.2f} 月  相关={corr:.3f}")
        if best is None or mae < best[0]:
            best = (mae, name, model)

    mae, name, model = best
    pred = model.predict(X[te_idx]).clip(0, 240)
    y_va, ids_va = y[te_idx], [ids_all[i] for i in te_idx]
    err = np.abs(pred - y_va)
    rmse = np.sqrt(((pred - y_va) ** 2).mean())

    print("\n" + "=" * 60)
    print(f"数据驱动校准（最佳模型: {name}，13 骨得分→骨龄）")
    print(f"训练样本: {len(tr_idx)}  测试样本: {len(te_idx)}")
    print(f"MAE  = {mae:.2f} 月 ({mae/12:.2f} 岁)")
    print(f"RMSE = {rmse:.2f} 月")
    print(f"误差分位: P50={np.percentile(err,50):.1f}  P90={np.percentile(err,90):.1f}  最大={err.max():.1f}")
    print(f"预测 vs 真实 相关系数: {np.corrcoef(pred, y_va)[0,1]:.3f}")

    # 分性别
    print("\n分性别:")
    for sex in ("male", "female"):
        idx = [i for i, sid in enumerate(ids_va) if sex_map.get(sid) == sex]
        if idx:
            e = np.abs(pred[idx] - y_va[idx])
            print(f"  {sex}: n={len(idx)}  MAE={e.mean():.2f} 月")

    # 分年龄段
    print("\n分年龄段 MAE:")
    for lo in range(0, 19):
        idx = [i for i in range(len(y_va)) if lo * 12 <= y_va[i] < (lo + 1) * 12]
        if idx:
            e = np.abs(pred[idx] - y_va[idx])
            print(f"  {lo}-{lo+1}岁: n={len(idx):>3}  MAE={e.mean():6.2f} 月")

    # 保存预测
    out = config.BAA_DIR / "data" / "calib_holdout_predictions.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "true_months", "pred_months", "abs_err"])
        for i, sid in enumerate(ids_va):
            w.writerow([sid, y_va[i], round(pred[i], 1), round(err[i], 1)])
    print(f"\n预测结果已保存 -> {out}")

    # 保存完整模型（含预处理器）供 pipeline 使用
    import joblib
    model_dir = config.BAA_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "imputer": imp, "name": name,
                 "train_mae_months": mae, "test_mae_months": mae},
                model_dir / "bone_age_regressor.pkl")
    print(f"模型已保存 -> {model_dir / 'bone_age_regressor.pkl'}")
    return mae


# ---------------------------------------------------------------- 主流程
def main():
    parser = argparse.ArgumentParser(description="数据驱动骨龄校准（13 骨得分→骨龄）")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ordinal", action="store_true",
                        help="用全部 9 关节 ordinal 模型提取特征（否则默认分类模型）")
    args = parser.parse_args()

    # 1. 生成验证集标签
    if not VAL_LABEL_CSV.exists():
        import zipfile, csv as _csv
        z = zipfile.ZipFile(config.WORKSPACE / "Bone+Age+Validation+Set.zip")
        z.extract("Bone Age Validation Set/Validation Dataset.csv", "rsna_tmp")
        src = Path("rsna_tmp/Bone Age Validation Set/Validation Dataset.csv")
        rows = list(_csv.DictReader(src.open(encoding="utf-8-sig")))
        with open(VAL_LABEL_CSV, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["Image ID", "male", "Bone Age (months)"])
            w.writerows([[r["Image ID"], r["male"], r["Bone Age (months)"]] for r in rows])
        print(f"[OK] 验证集标签 -> {VAL_LABEL_CSV}")

    preprocess_validation_images()
    if args.preprocess_only:
        return

    val_labels = load_val_labels()
    train_gt = {int(r["id"]): int(r["boneage_months"])
                for r in csv.DictReader(open(TRAIN_GT, encoding="utf-8"))}
    # 性别映射
    sex_map = {}
    for r in csv.DictReader(open(TRAIN_GT, encoding="utf-8")):
        sex_map[int(r["id"])] = "male" if r["sex"] == "male" else "female"
    for r in csv.DictReader(open(VAL_LABEL_CSV, encoding="utf-8-sig")):
        sex_map[int(r["Image ID"])] = "male" if r["male"].upper() == "TRUE" else "female"

    pipe = Pipeline(ordinal_all=args.ordinal)
    feat_train, feat_val = (FEAT_TRAIN_ORD, FEAT_VAL_ORD) if args.ordinal else (FEAT_TRAIN, FEAT_VAL)
    tag = "ordinal" if args.ordinal else "分类"
    print(f"[OK] 特征来源：{tag} 模型  -> {feat_train.name} / {feat_val.name}")

    # 2. 训练特征：881 张训练图（detection_pre 已预处理）
    print(f"\n提取训练集特征（{tag}，881 张）...")
    train_imgs = []
    for split in ("train", "val"):
        train_imgs += sorted((config.DETECTION_PRE / "images" / split).glob("*.png"))
    train_imgs = [p for p in sorted(train_imgs, key=lambda p: int(p.stem))
                  if int(p.stem) in train_gt]
    if args.limit_train:
        train_imgs = train_imgs[:args.limit_train]
    rows_tr = extract_features(pipe, train_imgs, train_gt, feat_train, "train", True)
    _, X_tr, y_tr = to_matrix(rows_tr)
    print(f"训练特征矩阵: {X_tr.shape}")

    # 3. 测试特征：1425 张验证图
    print(f"\n提取验证集特征（{tag}，1425 张）...")
    val_imgs = sorted(VAL_PRE.glob("*.png"))
    val_imgs = [p for p in sorted(val_imgs, key=lambda p: int(p.stem))
                if int(p.stem) in val_labels]
    if args.limit_val:
        val_imgs = val_imgs[:args.limit_val]
    rows_va = extract_features(pipe, val_imgs, val_labels, feat_val, "val", True)
    ids_va, X_va, y_va = to_matrix(rows_va)
    print(f"验证特征矩阵: {X_va.shape}")

    # 4. 合并全部带标签数据，按年龄分层划分训练/测试
    import numpy as np
    X_all = np.vstack([X_tr, X_va])
    y_all = np.concatenate([y_tr, y_va])
    ids_all = [int(r["id"]) for r in rows_tr] + [int(r["id"]) for r in rows_va]
    print(f"\n合并后总样本: {X_all.shape[0]}（训练 {len(rows_tr)} + 验证 {len(rows_va)}）")
    train_and_eval(X_all, y_all, ids_all, sex_map)


if __name__ == "__main__":
    main()
