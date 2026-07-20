"""
VisA 以图搜图 — 可视化
"""
import matplotlib.pyplot as plt
from PIL import Image
import os
from config import VISA_ROOT

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _resolve_path(rel_path, root=VISA_ROOT):
    """将相对路径 (cat/subset/filename) 转为绝对路径"""
    parts = rel_path.split('/')
    return os.path.join(root, parts[0], 'Data', 'Images', parts[1], parts[2])


def show_search(query_path, results, visa_root=VISA_ROOT):
    """
    展示查询图片 + Top-5 检索结果。

    参数:
        query_path: 查询图片的相对路径 (cat/subset/filename)
        results: feature_lib.search() 的返回结果 [(path, dist, conf), ...]
    """
    query_cat = query_path.split('/')[0]
    actual_path = _resolve_path(query_path, visa_root)

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))

    # ---- 先全部隐藏，再选择性显示 ----
    for row in range(2):
        for col in range(5):
            axes[row, col].axis('off')

    # ---- 查询图片 (左上) ----
    query_img = Image.open(actual_path).convert('RGB')
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title(f"查询: {query_cat}", fontsize=12, fontweight='bold')

    # ---- Top-5 标题 (中上) ----
    axes[0, 1].text(0.5, 0.5, "Top-5 检索结果", ha='center', va='center',
                    fontsize=14, fontweight='bold')

    # ---- 检索结果 ----
    positions = [(0, 3), (0, 4), (1, 0), (1, 1), (1, 2)]
    for i, (path, dist, conf) in enumerate(results):
        real_path = _resolve_path(path, visa_root)
        img = Image.open(real_path).convert('RGB')

        res_cat = path.split('/')[0]
        is_correct = (res_cat == query_cat)
        color = 'green' if is_correct else 'red'
        title = f"#{i+1} {res_cat}\nd={dist:.4f}\n置信度: {conf:.2%}"

        row, col = positions[i]
        axes[row, col].imshow(img)
        axes[row, col].set_title(title, color=color, fontsize=9)

    status = '全部正确' if all(r[0].split('/')[0] == query_cat for r in results) else '部分错误'
    plt.suptitle(f"VisA 以图搜图 — 查询: {query_cat}  ({status})",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def show_batch_results(query_paths, results_list, visa_root=VISA_ROOT):
    """
    批量展示多个查询结果（文本形式）。

    参数:
        query_paths: 查询路径列表
        results_list: search() 结果列表
    """
    for query_path, results in zip(query_paths, results_list):
        q_cat = query_path.split('/')[0]
        print(f"\n  查询: {query_path}")
        print(f"  {'排名':>4s} | {'检索结果':25s} | {'距离':>8s} | {'置信度':>8s} | {'结果'}")
        for i, (p, d, conf) in enumerate(results):
            res_cat = p.split('/')[0]
            ok = 'Yes' if res_cat == q_cat else 'No'
            print(f"  {i+1:4d} | {p:25s} | {d:.4f}  | {conf:.2%}   | {ok}")
