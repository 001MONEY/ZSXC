"""
VisA 以图搜图 — 数据集遍历
"""
import os
from config import VISA_ROOT, CATEGORIES, SUBSETS, MAX_PER_CLASS


def collect_samples(root=VISA_ROOT, categories=CATEGORIES,
                    subsets=SUBSETS, max_per_class=MAX_PER_CLASS):
    """
    遍历 VisA 数据集，收集所有图片路径。

    返回:
        samples: [(img_full_path, class_name, subset_name), ...]
    """
    samples = []

    for cat in categories:
        cat_path = os.path.join(root, cat, 'Data', 'Images')
        if not os.path.exists(cat_path):
            print(f"  跳过 {cat}: 路径不存在")
            continue

        for subset in subsets:
            subset_path = os.path.join(cat_path, subset)
            if not os.path.exists(subset_path):
                continue

            img_files = sorted(os.listdir(subset_path))
            for img_name in img_files[:max_per_class]:
                img_path = os.path.join(subset_path, img_name)
                samples.append((img_path, cat, subset))

    return samples


def print_stats(samples):
    """打印数据集统计信息"""
    from collections import defaultdict
    cat_counts = defaultdict(int)
    subset_counts = defaultdict(int)

    for _, cat, subset in samples:
        cat_counts[cat] += 1
        subset_counts[subset] += 1

    print(f"\n数据集统计:")
    print(f"  总图片数: {len(samples)}")
    print(f"  类别数:   {len(cat_counts)}")
    print(f"  子集分布: ", dict(subset_counts))
    print()
    for cat in sorted(cat_counts):
        print(f"    {cat:15s}: {cat_counts[cat]:3d} 张")


if __name__ == '__main__':
    samples = collect_samples()
    print_stats(samples)
