# -*- coding: utf-8 -*-
"""
分类/序数模型评估：计算真实的等级指标

重要：ImageFolder 的类别是字符串字典序（'1','10','11','2'...），
必须用 grade_list（数字升序的真实等级值）把 label 索引映射为等级，
否则等级 MAE 等指标全是错的。

支持评估两种模型：
  - 普通分类: models/classification/{关节}_best.pt （含 grade_list，无则从数据推断）
  - 序数回归: models/classification/{关节}_ordinal_best.pt

用法：
    python evaluate_classification.py                       # 评估全部关节分类模型
    python evaluate_classification.py --ordinal             # 评估序数模型
    python evaluate_classification.py --joints Ulna DIP
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import config

VAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

MODELS_DIR = config.BAA_DIR / "models" / "classification"


def load_model(joint, ordinal, device):
    name = f"{joint}_ordinal_best.pt" if ordinal else f"{joint}_best.pt"
    path = MODELS_DIR / name
    if not path.exists():
        return None, None, None
    ckpt = torch.load(path, map_location=device, weights_only=True)
    grade_list = ckpt.get("grade_list")
    num_classes = len(ckpt["classes"]) if "classes" in ckpt else (len(grade_list) if grade_list else None)

    if ordinal:
        from train_ordinal import OrdinalHead
        model = models.resnet18()
        model.fc = OrdinalHead(model.fc.in_features, num_classes)
    else:
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), grade_list, ckpt


def predict(model, loader, device, ordinal, grade_list):
    model.eval()
    pred_grades, true_grades = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if ordinal:
                probs = torch.sigmoid(out)
                idx = probs.sum(1).round().clamp(0, len(grade_list) - 1).long()
                preds = torch.tensor(grade_list, device=device)[idx]
            else:
                preds = torch.tensor([grade_list[i] for i in out.argmax(1).tolist()], device=device)
            pred_grades.extend(preds.cpu().tolist())
            true_grades.extend([grade_list[i] for i in y.tolist()])
    return np.array(pred_grades), np.array(true_grades)


def main():
    parser = argparse.ArgumentParser(description="评估分类/序数模型（真实等级指标）")
    parser.add_argument("--joints", nargs="+", default=None, help="关节列表（默认全部）")
    parser.add_argument("--ordinal", action="store_true", help="评估序数回归模型")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = "序数回归" if args.ordinal else "普通分类"
    print(f"设备: {device}  评估模式: {tag}")

    joints = args.joints or config.JOINT_TYPES
    results = []
    for joint in joints:
        if joint not in config.JOINT_TYPES:
            continue
        model, grade_list, _ = load_model(joint, args.ordinal, device)
        if model is None:
            print(f"  [跳过] {joint}: 模型不存在")
            continue
        if grade_list is None:
            # 从数据推断 grade_list
            root = config.CLASSIFICATION_PRE / joint
            ds = datasets.ImageFolder(root / "val")
            grade_list = sorted(int(c) for c in ds.classes)

        root = config.CLASSIFICATION_PRE / joint
        ds = datasets.ImageFolder(root / "val", transform=VAL_TF)
        dl = DataLoader(ds, batch_size=64, num_workers=2)
        pred, true = predict(model, dl, device, args.ordinal, grade_list)

        mae = float(np.abs(pred - true).mean())
        acc = float(accuracy_score(true, pred))
        acc1 = float((np.abs(pred - true) <= 1).mean())
        results.append({"joint": joint, "grade_mae": round(mae, 4),
                        "acc": round(acc, 4), "acc1": round(acc1, 4), "val": len(true)})
        print(f"  {joint:<10} 等级MAE={mae:.4f}  精确acc={acc:.4f}  ±1acc={acc1:.4f}  (val={len(true)})")

    if results:
        out = MODELS_DIR / ("summary_eval_ordinal.csv" if args.ordinal else "summary_eval.csv")
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        avg = sum(r["grade_mae"] for r in results) / len(results)
        avg_acc1 = sum(r["acc1"] for r in results) / len(results)
        print(f"\n平均等级MAE={avg:.4f}  平均±1acc={avg_acc1:.4f}")
        print(f"结果已保存 -> {out}")


if __name__ == "__main__":
    main()
