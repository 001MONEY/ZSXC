"""
VisA 以图搜图 — 主入口
"""
import os
from config import MODEL_NAME, FEAT_FILE, CATEGORIES
from dataset import collect_samples, print_stats
from model import FeatureExtractor
from feature_lib import build, load, search, evaluate
from visualize import show_search, show_batch_results


def main():
    print("=" * 70)
    print("VisA 数据集 — 以图搜图")
    print("=" * 70)

    # ---- Step 1: 收集样本 ----
    print("\n[1] 收集 VisA 样本")
    print("-" * 50)
    samples = collect_samples()
    print_stats(samples)

    # ---- Step 2: 创建特征提取器 ----
    print("\n[2] 创建特征提取器")
    print("-" * 50)
    extractor = FeatureExtractor(model_name=MODEL_NAME)

    # ---- Step 3: 制作 / 加载特征库 ----
    print("\n[3] 特征库")
    print("-" * 50)
    if os.path.exists(FEAT_FILE):
        print(f"  特征库已存在，直接加载 {FEAT_FILE}")
        paths, features = load(FEAT_FILE)
    else:
        paths, features = build(extractor, samples, FEAT_FILE)

    # ---- Step 4: 评估准确率 ----
    print("\n[4] 评估检索准确率")
    print("-" * 50)
    top1, top5 = evaluate(paths, features)
    print(f"\n  📊 {MODEL_NAME} 在 VisA 上的检索结果:")
    print(f"     Top-1 准确率: {top1:.2f}%")
    print(f"     Top-5 准确率: {top5:.2f}%")

    if top1 >= 75:
        print(f"  [OK] 超过 75% 要求！")
    elif top1 >= 70:
        print(f"  [!] 接近 70%，建议优化")
    else:
        print(f"  [x] 需要优化")

    # ---- Step 5: 检索演示 ----
    print("\n[5] 检索演示")
    print("-" * 50)

    # 前 3 类各选一张 Normal 做查询
    query_list = []
    result_list = []
    for cat in CATEGORIES[:3]:
        cat_indices = [i for i, p in enumerate(paths)
                       if p.startswith(f"{cat}/Normal/")]
        if not cat_indices:
            continue
        idx = cat_indices[5] if len(cat_indices) > 5 else cat_indices[0]
        q_path = paths[idx]
        q_feat = features[idx]

        results = search(q_feat, paths, features, top_k=5)
        query_list.append(q_path)
        result_list.append(results)

    show_batch_results(query_list, result_list)

    # ---- Step 6: 可视化 ----
    print("\n[6] 可视化展示")
    print("-" * 50)
    # 用 candle/Normal 第一张做可视化
    for p in paths:
        if p.startswith("candle/Normal/"):
            idx = paths.index(p)
            results = search(features[idx], paths, features)
            show_search(p, results)
            break

    print(f"\n{'=' * 70}")
    print("VisA 以图搜图完成！")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
