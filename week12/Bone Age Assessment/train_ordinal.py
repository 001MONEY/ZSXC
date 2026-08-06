# -*- coding: utf-8 -*-
"""
序数回归训练：9 个关节的发育等级预测（CORN 条件序数回归）

为什么用序数回归：
  骨成熟是连续过程，相邻等级（如 5 级 vs 6 级）外观极相似，普通分类把"判错1级"
  和"判错5级"同等惩罚，不合理。CORN 把任务分解为 K-1 个二分类"等级是否 > i"，
  - 损失按等级距离自然加权（越界的错误惩罚越重）
  - 对极端类别不平衡更鲁棒（每个阈值独立加权）
  - 天然输出可解释的累积概率

模型：ResNet18 骨干 + 序数头（K-1 个 sigmoid 输出）
指标：等级 MAE（主指标，越小越好）、精确 acc、±1 级 acc

用法：
    python train_ordinal.py                              # 训练 Ulna（问题关节）
    python train_ordinal.py --joints Ulna Radius         # 指定关节
    python train_ordinal.py --joints DIP MCP MIP PIP     # 训练全部
    python train_ordinal.py --smoke                      # 冒烟测试
"""
import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

import config

MODELS_DIR = config.BAA_DIR / "models" / "classification"
RUNS_DIR = config.BAA_DIR / "runs" / "classification"

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
VAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ---------------------------------------------------------------- 序数模型
def grade_list_of(dataset) -> list:
    """ImageFolder 类别名是字符串（字典序），必须转成真实等级值并数字排序。
    例：classes=['1','10','11','2',...] → grade_list=[1,2,...,12]"""
    return sorted(int(c) for c in dataset.classes)


class OrdinalHead(nn.Module):
    """CORN 序数头：输出 K-1 个 logit，第 i 个经 sigmoid 后表示 P(等级 > grade_list[i])。
    注意：阈值顺序按 grade_list（数字升序）排列，与 ImageFolder 的字符串顺序无关。"""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes - 1)

    def forward(self, x):
        return self.fc(x)   # 返回 logits（配合 BCEWithLogitsLoss，AMP 安全）


def build_ordinal_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = OrdinalHead(model.fc.in_features, num_classes)
    return model


def make_targets(labels: torch.Tensor, grade_list: list) -> torch.Tensor:
    """labels: [B] ImageFolder 索引 → target [B, K-1]，t[b,i]=1 if 等级 > grade_list[i]"""
    grades = torch.tensor([grade_list[i] for i in labels.tolist()], device=labels.device)
    thresholds = torch.tensor(grade_list[:-1], device=labels.device)   # K-1 个阈值等级
    return (grades.unsqueeze(1) > thresholds.unsqueeze(0)).float()    # [B, K-1]


def make_threshold_weights(labels: torch.Tensor, grade_list: list) -> torch.Tensor:
    """每个阈值独立类别加权：w = N/(2*n_pos) 或 N/(2*n_neg)，返回 [B, K-1] 权重矩阵"""
    n = len(labels)
    targets = make_targets(labels, grade_list)              # [B, K-1]
    n_pos = targets.sum(0).clamp(min=1)                     # [K-1]
    n_neg = (n - targets.sum(0)).clamp(min=1)               # [K-1]
    pos_w = n / (2.0 * n_pos)
    neg_w = n / (2.0 * n_neg)
    return torch.where(targets == 1, pos_w, neg_w)          # [B, K-1]


def predict_grade(logits: torch.Tensor, grade_list: list) -> torch.Tensor:
    """logits: [B, K-1]（等级>grade_list[i] 的对数几率）→ 预测等级值 [B]
    软版本：等级 ≈ grade_list[0] + Σ P(>i)，对相邻混淆更稳健"""
    probs = torch.sigmoid(logits)
    idx = probs.sum(1).round().clamp(0, len(grade_list) - 1).long()
    return torch.tensor(grade_list, device=logits.device)[idx]


# ---------------------------------------------------------------- 评估
def evaluate(model, loader, device, grade_list):
    model.eval()
    pred_grades, true_grades = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred_grades.extend(predict_grade(logits, grade_list).cpu().tolist())
            true_grades.extend([grade_list[i] for i in y.tolist()])
    pred_grades, true_grades = np.array(pred_grades), np.array(true_grades)
    mae = np.abs(pred_grades - true_grades).mean()
    acc = accuracy_score(true_grades, pred_grades)
    acc1 = (np.abs(pred_grades - true_grades) <= 1).mean()
    return mae, acc, acc1, true_grades, pred_grades


def plot_confusion(labels, preds, classes, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title} (MAE={np.abs(labels - preds).mean():.3f})")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train_loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="val_loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history["epoch"], history["val_mae"], label="val_MAE", color="C2")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("grade MAE")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- 训练
def train_joint(joint, args, device):
    print("\n" + "=" * 60)
    print(f"训练关节: {joint}（序数回归）")
    root = config.CLASSIFICATION_PRE / joint
    if not (root / "train").exists():
        print(f"[警告] 缺少 {root}，跳过")
        return None

    train_ds = datasets.ImageFolder(root / "train", transform=TRAIN_TF)
    val_ds = datasets.ImageFolder(root / "val", transform=VAL_TF)
    num_classes = len(train_ds.classes)
    grade_list = grade_list_of(train_ds)   # 真实等级值（数字升序），如 [1,2,...,12]
    print(f"  等级数: {num_classes}  train: {len(train_ds)}  val: {len(val_ds)}")
    print(f"  等级值: {grade_list}")

    model = build_ordinal_model(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    run_dir = RUNS_DIR / f"{joint}_ordinal"
    run_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_mae, best_epoch, patience = 1e9, 0, 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_mae": []}
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"{joint} E{epoch}/{args.epochs}", ncols=80)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            targets = make_targets(y, grade_list)
            weights = make_threshold_weights(y, grade_list)
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(x)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        scheduler.step()

        val_mae, val_acc, val_acc1, _, _ = evaluate(model, val_loader, device, grade_list)
        # val_loss 用未加权的 BCE 计算便于对比
        val_loss = _val_plain_bce(model, val_loader, device, grade_list)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss / n)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        tag = ""
        if val_mae < best_mae:
            best_mae, best_epoch, patience = val_mae, epoch, 0
            torch.save({"state_dict": model.state_dict(), "classes": train_ds.classes,
                        "grade_list": grade_list, "model": "resnet18_ordinal",
                        "val_mae": float(val_mae)},
                       MODELS_DIR / f"{joint}_ordinal_best.pt")
            tag = " *best"
        else:
            patience += 1
        print(f"  E{epoch:>3}: train_loss={train_loss/n:.4f} val_loss={val_loss:.4f} "
              f"val_MAE={val_mae:.4f} acc={val_acc:.4f} ±1acc={val_acc1:.4f}{tag}")
        if patience >= args.patience:
            print(f"  [早停] {args.patience} 轮未提升，提前停止")
            break

    # 用最佳权重做最终评估 + 出图
    best_ckpt = torch.load(MODELS_DIR / f"{joint}_ordinal_best.pt",
                           map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["state_dict"])
    final_mae, final_acc, final_acc1, labels, preds = evaluate(model, val_loader, device, grade_list)
    plot_confusion(labels, preds, grade_list, run_dir / "confusion.png", "Ordinal CM")
    plot_history(history, run_dir / "history.png")

    print(f"  [OK] {joint}: 最佳 val_MAE={final_mae:.4f} acc={final_acc:.4f} "
          f"±1acc={final_acc1:.4f}（epoch {best_epoch}）耗时 {time.time()-t0:.0f}s")
    return {"joint": joint, "num_classes": num_classes, "train": len(train_ds), "val": len(val_ds),
            "best_mae": round(final_mae, 4), "best_acc": round(final_acc, 4),
            "acc1": round(final_acc1, 4), "best_epoch": best_epoch}


def _val_plain_bce(model, loader, device, grade_list):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, make_targets(y, grade_list))
            total += loss.item() * x.size(0)
            n += x.size(0)
    return total / n


# ---------------------------------------------------------------- 主流程
def main():
    parser = argparse.ArgumentParser(description="序数回归训练（CORN，等级预测）")
    parser.add_argument("--joints", nargs="+", default=["Ulna"], help="关节列表（默认 Ulna）")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="冒烟测试：1 关节 1 轮")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}  序数回归(CORN)  epochs={args.epochs} batch={args.batch} lr={args.lr}")

    joints = args.joints
    if args.smoke:
        joints, args.epochs = ["Ulna"], 1

    summary = []
    for j in joints:
        if j not in config.JOINT_TYPES:
            print(f"[警告] 未知关节类型: {j}，跳过")
            continue
        r = train_joint(j, args, device)
        if r:
            summary.append(r)

    if summary:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = MODELS_DIR / "summary_ordinal.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print("\n" + "=" * 60)
        print(f"汇总 -> {csv_path}")
        for r in summary:
            print(f"  {r['joint']:<10} MAE={r['best_mae']:.4f}  acc={r['best_acc']:.4f}  "
                  f"±1acc={r['acc1']:.4f} (E{r['best_epoch']})")
        avg = sum(r["best_mae"] for r in summary) / len(summary)
        print(f"  平均等级 MAE = {avg:.4f}")
    print("\n[OK] 全部完成！")


if __name__ == "__main__":
    main()
