"""分析视频推理的置信度分布，找出被0.85阈值误判的商品。"""

import csv
from collections import defaultdict

path = r"runs\pipeline\threshold_check\frame_records.csv"
rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
print("总记录数:", len(rows))

stats = defaultdict(list)
for r in rows:
    stats[r["model_class"]].append(float(r["class_conf"]))

print()
print(f"{'model_class':<35}{'次数':>5}{'最低':>8}{'最高':>8}{'平均':>8}")
print("-" * 70)
for mc, confs in sorted(stats.items()):
    print(f"{mc:<35}{len(confs):>5}{min(confs):>8.3f}{max(confs):>8.3f}{sum(confs)/len(confs):>8.3f}")

print()
low = [r for r in rows if float(r["class_conf"]) < 0.85]
print(f"置信度<0.85 的记录数: {len(low)}")
for r in low[:15]:
    print(f"  frame={r['frame']} {r['model_class']} conf={r['class_conf']} found={r['found']}")
