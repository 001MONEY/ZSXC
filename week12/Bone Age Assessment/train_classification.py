# -*- coding: utf-8 -*-
"""
分类模型训练：9 个关节类型的发育等级分类（迁移学习）

数据：datasets/classification_pre/{关节}/train|val/{等级}/
模型：torchvision MobileNetV3-Small（默认）/ ResNet18，ImageNet 预训练
策略：
  - 全量微调（小学习率，CosineAnnealing 衰减）
  - CrossEntropyLoss 类别加权（缓解等级样本不平衡）
  - 早停 + 保存 val_acc 最佳权重
  - 自动生成混淆矩阵与训练曲线

用法：
    python train_classification.py                          # 训练全部 9 个关节
    python train_classification.py --joints DIP Radius      # 只训练指定关节
    python train_classification.py --model resnet18         # 换模型
    python train_classification.py --epochs 60 --batch 32
    python train_classification.py --smoke                  # 冒烟测试（1 关节 1 轮）
"""
import argparse
import csv
import time
from collections import Counter

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

# 训练增强：轻度旋转 ±15°（方案第二阶段建议），其余为通用增强
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


# ---------------------------------------------------------------- 模型
def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "mobilenetv3_small":
        # 注意：需要联网下载权重（download.pytorch.org 国内常不可达）
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"未知模型: {name}（可选 mobilenetv3_small / mobilenet_v2 / resnet18 / resnet34）")
    return model


def make_class_weights(targets):
    """按样本数反比生成类别权重：w_i = total / (n_class * count_i)"""
    counts = Counter(targets)
    total = len(targets)
    n = len(counts)
    return torch.tensor([total / (n * counts[i]) for i in range(n)], dtype=torch.float32)


# ---------------------------------------------------------------- 评估/绘图
def evaluate(model, loader, device, loss_fn, grade_list=None):
    """grade_list: 真实等级值（数字升序）。传入则额外计算等级 MAE 与 ±1 级准确率。
    注意：ImageFolder 类别是字典序，必须用 grade_list 把 label 索引映射为真实等级。"""
    model.eval()
    preds, labels, total_loss, n = [], [], 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += loss_fn(out, y).item() * x.size(0)
            preds.extend(out.argmax(1).cpu().tolist())
            labels.extend(y.cpu().tolist())
            n += x.size(0)
    acc = accuracy_score(labels, preds)
    mae, acc1 = None, None
    if grade_list is not None:
        true_grades = np.array([grade_list[i] for i in labels])
        pred_grades = np.array([grade_list[i] for i in preds])
        mae = float(np.abs(pred_grades - true_grades).mean())
        acc1 = float((np.abs(pred_grades - true_grades) <= 1).mean())
    return acc, total_loss / n, labels, preds, mae, acc1


def plot_confusion(labels, preds, classes, save_path):
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
    ax.set_title(f"Confusion Matrix (acc={accuracy_score(labels, preds):.4f})")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train_loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="val_loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history["epoch"], history["val_acc"], label="val_acc", color="C2")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("acc"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- 单关节训练
def train_joint(joint, args, device):
    print("\n" + "=" * 60)
    print(f"训练关节: {joint}")
    root = config.CLASSIFICATION_PRE / joint
    if not (root / "train").exists():
        print(f"[警告] 缺少 {root}，跳过")
        return None

    train_ds = datasets.ImageFolder(root / "train", transform=TRAIN_TF)
    val_ds = datasets.ImageFolder(root / "val", transform=VAL_TF)
    num_classes = len(train_ds.classes)
    print(f"  等级数: {num_classes}  train: {len(train_ds)}  val: {len(val_ds)}")

    model = build_model(args.model, num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=make_class_weights(train_ds.targets).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # 真实等级值（数字升序），ImageFolder 类别是字典序需转换
    grade_list = sorted(int(c) for c in train_ds.classes)
    print(f"  等级值: {grade_list}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    run_dir = RUNS_DIR / joint
    run_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_acc, best_epoch, patience = 0.0, 0, 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"{joint} E{epoch}/{args.epochs}", ncols=80)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                out = model(x)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        scheduler.step()

        val_acc, val_loss, _, _, val_mae, val_acc1 = evaluate(model, val_loader, device, loss_fn, grade_list)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss / n)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        tag = ""
        if val_acc > best_acc:
            best_acc, best_epoch, patience = val_acc, epoch, 0
            torch.save({"state_dict": model.state_dict(), "classes": train_ds.classes,
                        "grade_list": grade_list, "model": args.model, "val_acc": val_acc},
                       MODELS_DIR / f"{joint}_best.pt")
            tag = " ★best"
        else:
            patience += 1
        print(f"  E{epoch:>3}: train_loss={train_loss/n:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} 等级MAE={val_mae:.4f} ±1acc={val_acc1:.4f}{tag}")
        if patience >= args.patience:
            print(f"  [早停] {args.patience} 轮未提升，提前停止")
            break

    # 用最佳权重做最终评估 + 出图
    best_ckpt = torch.load(MODELS_DIR / f"{joint}_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["state_dict"])
    final_acc, _, labels, preds, final_mae, final_acc1 = evaluate(model, val_loader, device, loss_fn, grade_list)
    plot_confusion(labels, preds, grade_list, run_dir / "confusion.png")
    plot_history(history, run_dir / "history.png")

    print(f"  [OK] {joint}: 最佳 val_acc={final_acc:.4f} 等级MAE={final_mae:.4f} "
          f"±1acc={final_acc1:.4f}（epoch {best_epoch}）耗时 {time.time()-t0:.0f}s")
    return {"joint": joint, "num_classes": num_classes, "train": len(train_ds), "val": len(val_ds),
            "best_acc": round(final_acc, 4), "best_mae": round(final_mae, 4),
            "acc1": round(final_acc1, 4), "best_epoch": best_epoch, "model": args.model}


# ---------------------------------------------------------------- 主流程
def main():
    parser = argparse.ArgumentParser(description="9 个关节的发育等级分类训练")
    parser.add_argument("--joints", nargs="+", default=None, help="只训练指定关节，如 DIP Radius")
    parser.add_argument("--model", default="resnet18",
                        choices=["mobilenetv3_small", "mobilenet_v2", "resnet18", "resnet34"],
                        help="分类模型（resnet18/mobilenet_v2 权重已本地缓存，mobilenetv3_small 需联网）")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15, help="早停轮数")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="冒烟测试：1 关节 1 轮")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}  模型: {args.model}  epochs={args.epochs} batch={args.batch} lr={args.lr}")

    joints = args.joints or config.JOINT_TYPES
    if args.smoke:
        joints, args.epochs = ["DIP"], 1

    summary = []
    for j in joints:
        if j not in config.JOINT_TYPES:
            print(f"[警告] 未知关节类型: {j}，跳过")
            continue
        r = train_joint(j, args, device)
        if r:
            summary.append(r)

    # 汇总表
    if summary:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = MODELS_DIR / "summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print("\n" + "=" * 60)
        print(f"汇总 -> {csv_path}")
        for r in summary:
            print(f"  {r['joint']:<10} 等级{r['num_classes']:>2}  acc={r['best_acc']:.4f} (E{r['best_epoch']})")
        # 平均精度
        avg = sum(r["best_acc"] for r in summary) / len(summary)
        print(f"  平均 val_acc = {avg:.4f}")
    print("\n[OK] 全部完成！")


if __name__ == "__main__":
    main()
