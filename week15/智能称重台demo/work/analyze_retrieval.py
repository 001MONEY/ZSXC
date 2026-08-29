"""分析 retrieval_v2 中星巴克等商品的检索得分。"""

import csv
from collections import Counter

rows = list(csv.DictReader(open(r"runs\pipeline\retrieval_v2\frame_records.csv", encoding="utf-8-sig")))
print("总记录数:", len(rows))

# 查看 BOTTLE_06 相关记录（星巴克）和 found=False 的记录
print("\n=== BOTTLE_06_starbucks 记录 ===")
b06 = [r for r in rows if r["model_class"] == "BOTTLE_06_starbucks" or "starbucks" in r.get("top2_class", "")]
print(f"数量: {len(b06)}")
for r in b06[:10]:
    print(f"  frame={r['frame']} pred={r['model_class']} sim={r['class_conf']} margin={r['margin']} "
          f"top2={r['top2_class']}({r['top2_sim']}) found={r['found']}")

print("\n=== 未注册(found=False) 记录统计 ===")
unknown = [r for r in rows if r["found"] == "False"]
print(f"未注册条数: {len(unknown)}")
cnt = Counter((r["model_class"], r["unknown_reason"]) for r in unknown)
for (mc, reason), n in cnt.most_common(15):
    print(f"  {mc} x{n}  原因={reason}")

print("\n=== 各 model_class 的 sim/margin 分布 ===")
from collections import defaultdict
stats = defaultdict(list)
for r in rows:
    stats[r["model_class"]].append((float(r["class_conf"]), float(r["margin"] if r["margin"] else 0)))
for mc, vals in sorted(stats.items()):
    sims = [v[0] for v in vals]
    ms = [v[1] for v in vals]
    print(f"  {mc:<38} n={len(vals):>4} sim[{min(sims):.3f},{max(sims):.3f}] margin[{min(ms):.3f},{max(ms):.3f}]")
