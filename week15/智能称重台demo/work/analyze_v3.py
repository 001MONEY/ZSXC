"""分析 retrieval_v3 中星巴克的检索排名。"""

import csv

rows = list(csv.DictReader(open(r"runs\pipeline\retrieval_v3\frame_records.csv", encoding="utf-8-sig")))
print("总记录数:", len(rows))

# 星巴克相关记录（top1或top2是starbucks）
print("\n=== 含 starbucks 的记录（前20） ===")
n = 0
for r in rows:
    if "starbucks" in r["model_class"] or "starbucks" in r.get("top2_class", ""):
        n += 1
        if n <= 20:
            print(
                f"frame={r['frame']} top1={r['model_class']}({r['class_conf']}) "
                f"margin={r['margin']} top2={r['top2_class']}({r['top2_sim']}) found={r['found']}"
            )
print("含starbucks记录总数:", n)

# 视频里最常出现的瓶装商品（星巴克在真实视频中应高频出现）
print("\n=== 各model_class出现次数（bottle类） ===")
from collections import Counter

cnt = Counter(r["model_class"] for r in rows if r["group"] == "bottle")
for mc, c in cnt.most_common(10):
    print(f"  {mc}: {c}")
