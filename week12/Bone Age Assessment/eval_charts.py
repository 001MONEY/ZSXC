# -*- coding: utf-8 -*-
"""
误差分析图表生成（论文用）

数据源: data/full_val_predictions.csv（1425 张验证图，方案 A 严格独立 + 方案 B 生产对照）
输出  : output/eval_charts/
  - mae_by_age.png        分年龄段 MAE 柱状图（A/B 对比）
  - scatter_pred_true.png 预测 vs 真实 散点图（方案 A，含 y=x）
  - error_dist.png        误差分布直方图（方案 A/B）
  - mae_by_age_sex.png    分年龄段 x 性别 MAE

用法:
    python eval_charts.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BAA = Path(__file__).parent
SRC = BAA / "data" / "full_val_predictions.csv"
OUT = BAA / "output" / "eval_charts"


def load():
    rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
    y = np.array([float(r["true_months"]) for r in rows])
    keep = y >= 72                      # 正式评估范围：排除学龄前（6 岁以下）
    y = y[keep]
    pa = np.array([float(r["pred_A_strict"]) for r in rows])[keep]
    pb = np.array([float(r["pred_B_prod"]) for r in rows])[keep]
    sex = np.array([r["sex"] for r in rows])[keep]
    ea = np.abs(pa - y)
    eb = np.abs(pb - y)
    return y, pa, pb, sex, ea, eb


def chart1(y, ea, eb):
    """分年龄段 MAE 柱状图（A/B 对比）"""
    ages, ma, mb, ns = [], [], [], []
    for lo in range(6, 19):
        m = (y >= lo * 12) & (y < (lo + 1) * 12)
        if m.sum():
            ages.append(f"{lo}-{lo+1}")
            ma.append(ea[m].mean())
            mb.append(eb[m].mean())
            ns.append(m.sum())
    x = np.arange(len(ages))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w / 2, ma, w, label="严格独立", color="#4C72B0")
    ax.bar(x + w / 2, mb, w, label="生产对照", color="#DD8452")
    for i, n in enumerate(ns):
        ax.text(i - w / 2, ma[i] + 0.3, f"n={n}", ha="center", fontsize=7, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(ages)
    ax.set_xlabel("年龄段（岁）")
    ax.set_ylabel("MAE（月）")
    ax.set_title("分年龄段骨龄误差 MAE（6-19 岁，n=1273）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "mae_by_age.png", dpi=150)
    plt.close(fig)


def chart2(y, pa):
    """预测 vs 真实散点图（方案 A）"""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y / 12, pa / 12, s=6, alpha=0.35, color="#4C72B0")
    lim = [0.4, 19.5]
    ax.plot(lim, lim, "r--", lw=1.2, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("真实骨龄（岁）")
    ax.set_ylabel("预测骨龄（岁）")
    ax.set_title("预测 vs 真实骨龄（方案 A 严格独立）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "scatter_pred_true.png", dpi=150)
    plt.close(fig)


def chart3(ea, eb):
    """误差分布直方图"""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, ea.max() + 2, 2)
    ax.hist(ea, bins=bins, alpha=0.6, label="严格独立", color="#4C72B0")
    ax.hist(eb, bins=bins, alpha=0.5, label="生产对照", color="#DD8452")
    ax.axvline(ea.mean(), color="#4C72B0", ls="--", lw=1.2,
               label=f"A 均值 {ea.mean():.1f} 月")
    ax.axvline(eb.mean(), color="#DD8452", ls="--", lw=1.2,
               label=f"B 均值 {eb.mean():.1f} 月")
    ax.set_xlabel("绝对误差（月）")
    ax.set_ylabel("样本数")
    ax.set_title("骨龄误差分布")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "error_dist.png", dpi=150)
    plt.close(fig)


def chart4(y, ea, sex):
    """分年龄段 x 性别 MAE"""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for s, c, lab in [("male", "#4C72B0", "男"), ("female", "#C44E52", "女")]:
        ages, ma, ns = [], [], []
        for lo in range(6, 19):
            m = (y >= lo * 12) & (y < (lo + 1) * 12) & (sex == s)
            if m.sum():
                ages.append(f"{lo}-{lo+1}")
                ma.append(ea[m].mean())
                ns.append(m.sum())
        ax.plot(ages, ma, "o-", color=c, label=f"{lab}（总 n={sum(ns)}）")
    ax.set_xlabel("年龄段（岁）")
    ax.set_ylabel("MAE（月）")
    ax.set_title("分年龄段 x 性别 MAE（方案 A 严格独立）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "mae_by_age_sex.png", dpi=150)
    plt.close(fig)


def summary_table(y, ea, eb, sex):
    """汇总统计表（CSV + 打印）"""
    print("=" * 62)
    print("误差分析汇总（6-19 岁, n=1273）")
    print(f"  严格独立 A: MAE={ea.mean():.2f} 月  RMSE={np.sqrt((ea**2).mean()):.2f}  "
          f"P50={np.percentile(ea,50):.1f}  P90={np.percentile(ea,90):.1f}")
    print(f"  生产对照 B: MAE={eb.mean():.2f} 月  RMSE={np.sqrt((eb**2).mean()):.2f}  "
          f"P50={np.percentile(eb,50):.1f}  P90={np.percentile(eb,90):.1f}")
    print("  分性别:")
    for s, lab in [("male", "男"), ("female", "女")]:
        m = sex == s
        print(f"    {lab}: n={m.sum():>4}  MAE={ea[m].mean():6.2f} 月")
    print("  分年龄段:")
    rows = []
    for lo in range(6, 19):
        m = (y >= lo * 12) & (y < (lo + 1) * 12)
        if m.sum():
            rows.append([f"{lo}-{lo+1}岁", m.sum(),
                         round(ea[m].mean(), 2), round(eb[m].mean(), 2)])
            print(f"    {lo:2d}-{lo+1}岁: n={m.sum():>3}  A MAE={ea[m].mean():6.2f}  "
                  f"B MAE={eb[m].mean():6.2f}")
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["age_group", "n", "mae_A_strict", "mae_B_prod"])
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y, pa, pb, sex, ea, eb = load()
    chart1(y, ea, eb)
    chart2(y, pa)
    chart3(ea, eb)
    chart4(y, ea, sex)
    summary_table(y, ea, eb, sex)
    print(f"[OK] 图表已保存 -> {OUT}")


if __name__ == "__main__":
    main()
