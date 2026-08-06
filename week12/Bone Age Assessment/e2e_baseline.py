# -*- coding: utf-8 -*-
"""
端到端骨龄回归对比实验（论文实验表）

ResNet18 直接回归：X 光片 -> 骨龄(月)。与两阶段方案（检测+分类+校准）对比，
使用完全相同的训练/测试划分（6 岁+）：
  训练: 11279 张 RSNA 训练图（CLAHE 预处理, data/rsna_train_pre/）
  测试: 1273 张 6 岁+ 验证图（data/rsna_val_pre/ + rsna_val_labels.csv）

损失: SmoothL1（Huber），对离群鲁棒；AdamW + 余弦退火 + 早停(验证MAE)

对比目标: 两阶段方案严格独立 MAE = 12.83 月

用法:
    python e2e_baseline.py                 # 全量训练 (~25 epoch, 约20-30分钟)
    python e2e_baseline.py --smoke         # 冒烟: 1 epoch, 200 张
    python e2e_baseline.py --epochs 30 --batch 64
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import config

BAA = config.BAA_DIR
TRAIN_PRE = BAA / "data" / "rsna_train_pre"
VAL_PRE = BAA / "data" / "rsna_val_pre"
TRAIN_CSV = config.WORKSPACE / "train.csv"
VAL_CSV = BAA / "data" / "rsna_val_labels.csv"
MIN_AGE = 72
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TF_TRAIN = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
TF_VAL = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class BoneAgeDS(Dataset):
    def __init__(self, img_dir, labels, tf, ids):
        self.paths = [img_dir / f"{i}.png" for i in ids]
        self.labels = [labels[i] for i in ids]
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(str(self.paths[i]))          # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tf(Image.fromarray(img)), torch.tensor(self.labels[i], dtype=torch.float32)


def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 1)             # 回归头：直接输出骨龄(月)
    return m


def filter_broken(img_dir, ids):
    """预检：过滤 cv2.imread 无法读取的损坏图片。"""
    good, bad = [], 0
    for iid in ids:
        if cv2.imread(str(img_dir / f"{iid}.png")) is None:
            bad += 1
            print(f"  [跳过损坏] {iid}")
        else:
            good.append(iid)
    if bad:
        print(f"[预检] 剔除 {bad} 张损坏图，剩余 {len(good)}")
    return good


def evaluate(model, loader, device):
    """返回 (MAE, RMSE, 相关, pred, true)"""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).cpu().numpy().ravel().clip(0, 240)
            preds.extend(pred)
            trues.extend(y.numpy())
    p, t = np.array(preds), np.array(trues)
    e = np.abs(p - t)
    return e.mean(), np.sqrt((e ** 2).mean()), np.corrcoef(p, t)[0, 1], p, t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    # ---------- 数据 ----------
    train_labels, sex_map = {}, {}
    with open(TRAIN_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if int(r["boneage"]) >= MIN_AGE:
                train_labels[int(r["id"])] = float(r["boneage"])
    val_labels = {}
    with open(VAL_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            val_labels[int(r["Image ID"])] = float(r["Bone Age (months)"])
            sex_map[int(r["Image ID"])] = "male" if r["male"].upper() == "TRUE" else "female"

    train_ids = sorted(train_labels)
    val_ids = [i for i in sorted(val_labels) if val_labels[i] >= MIN_AGE]
    print(f"[数据] 训练 {len(train_ids)}  测试(6岁+) {len(val_ids)}")
    print("[预检] 扫描训练图（剔除损坏）...")
    train_ids = filter_broken(TRAIN_PRE, train_ids)
    val_ids = filter_broken(VAL_PRE, val_ids)
    if args.smoke:
        train_ids = train_ids[:200]

    train_ds = BoneAgeDS(TRAIN_PRE, train_labels, TF_TRAIN, train_ids)
    val_ds = BoneAgeDS(VAL_PRE, val_labels, TF_VAL, val_ids)
    tr_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4,
                       persistent_workers=True)
    va_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4,
                       persistent_workers=True)

    # ---------- 模型 ----------
    model = build_model().to(DEVICE)
    loss_fn = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_mae, best_state, best_epoch, bad = 1e9, None, -1, 0
    print(f"[训练] device={DEVICE} epochs={args.epochs} batch={args.batch} lr={args.lr}")
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in tr_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x).squeeze(1), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        sched.step()
        mae, rmse, corr, _, _ = evaluate(model, va_ld, DEVICE)
        mark = ""
        if mae < best_mae:
            best_mae, best_state, best_epoch, bad = mae, {k: v.clone() for k, v in model.state_dict().items()}, ep, 0
            mark = " *best"
        else:
            bad += 1
        print(f"  E{ep:2d}/{args.epochs} train={tot/n:.4f} val_MAE={mae:.2f} "
              f"rmse={rmse:.2f} corr={corr:.3f} {time.time()-t0:.0f}s{mark}")
        if bad >= args.patience:
            print(f"[早停] {args.patience} 轮未提升")
            break

    # ---------- 最终评估 ----------
    model.load_state_dict(best_state)
    mae, rmse, corr, pred, true = evaluate(model, va_ld, DEVICE)
    err = np.abs(pred - true)
    print("\n" + "=" * 60)
    print(f"端到端 ResNet18 回归（best epoch {best_epoch}）")
    print(f"MAE = {mae:.2f} 月 ({mae/12:.2f} 岁)   RMSE = {rmse:.2f}  相关 = {corr:.3f}")
    print(f"P50 = {np.percentile(err,50):.1f}  P90 = {np.percentile(err,90):.1f}")
    print("分性别:")
    for s in ("male", "female"):
        idx = [i for i, vid in enumerate(val_ids) if sex_map.get(vid) == s]
        if idx:
            print(f"  {s}: n={len(idx)}  MAE={err[idx].mean():.2f}")
    print("分年龄段:")
    for lo in range(6, 19):
        idx = [i for i in range(len(true)) if lo * 12 <= true[i] < (lo + 1) * 12]
        if idx:
            print(f"  {lo}-{lo+1}岁: n={len(idx):>3}  MAE={err[idx].mean():6.2f}")
    print(f"\n对比: 两阶段方案 MAE = 12.83 月")

    # 保存模型与预测
    out_dir = BAA / "models" / "e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "mae": mae, "epoch": best_epoch},
               out_dir / "resnet18_boneage.pt")
    with open(BAA / "data" / "e2e_val_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "true_months", "pred_months", "abs_err", "sex"])
        for i, vid in enumerate(val_ids):
            w.writerow([vid, round(true[i], 1), round(pred[i], 1), round(err[i], 1),
                        sex_map.get(vid, "")])
    print(f"[OK] 模型 -> {out_dir / 'resnet18_boneage.pt'}")
    print(f"[OK] 预测 -> {BAA / 'data' / 'e2e_val_predictions.csv'}")


if __name__ == "__main__":
    main()
