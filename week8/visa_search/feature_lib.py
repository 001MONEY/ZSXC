"""
VisA 以图搜图 — 特征库 (构建/加载/检索/评估)

与讲义对齐:
  - 特征存储格式: img_path#[feat1, feat2, ...]  (JSON列表)
  - 读取使用 json.loads
"""
import json
import numpy as np
import os
from config import FEAT_FILE


# ============================================================
# 构建特征库
# ============================================================
def build(extractor, samples, save_path=FEAT_FILE):
    """
    遍历样本，提取特征并保存到文件。

    格式 (与讲义一致): 图片路径#[特征值列表]
    返回: (paths_list, features_matrix)
    """
    paths = []
    features = []

    print(f"\n正在制作特征库 ({len(samples)} 张图片)...")

    for i, (img_path, cat, subset) in enumerate(samples):
        feat = extractor.extract(img_path)
        rel_path = f"{cat}/{subset}/{os.path.basename(img_path)}"
        paths.append(rel_path)
        features.append(feat)

        if (i + 1) % 100 == 0:
            print(f"  已处理: {i+1}/{len(samples)}")

    # 保存到文件 (JSON列表格式，与讲义一致)
    with open(save_path, 'w', encoding='utf-8') as f:
        for path, feat in zip(paths, features):
            feat_list = feat.tolist()
            f.write(f"{path}#{feat_list}\n")

    print(f"  特征库保存到 {save_path}")
    print(f"  共 {len(paths)} 条, 维度 {len(features[0])}")

    return paths, np.array(features)


# ============================================================
# 加载特征库
# ============================================================
def load(feat_file=FEAT_FILE):
    """从文件加载特征库 (与讲义一致，使用 json.loads)"""
    paths = []
    features = []

    with open(feat_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 与讲义一致: split("#") 后 json.loads 解析列表
            img_path, feat_str = line.split('#', 1)
            feat = np.array(json.loads(feat_str))
            paths.append(img_path)
            features.append(feat)

    print(f"  加载特征库: {len(paths)} 条, 维度 {len(features[0])}")
    return paths, np.array(features)


# ============================================================
# 相似度检索
# ============================================================
def cosine_confidence(query_feat, features):
    """
    计算查询特征与库中所有特征的余弦相似度作为置信度。

    余弦相似度范围 [-1, 1]，对 CNN 特征通常 > 0，裁剪到 [0, 1] 作为置信度。
    """
    # L2 归一化
    q_norm = query_feat / (np.linalg.norm(query_feat) + 1e-8)
    f_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    sims = (f_norm @ q_norm).flatten()
    return np.clip(sims, 0, 1)


def search(query_feat, paths, features, top_k=5):
    """
    以图搜图 — 欧氏距离 Top-K 检索 + 余弦相似度置信度。

    参数:
        query_feat: 查询特征向量 (numpy 数组)
        paths: 特征库路径列表
        features: 特征库特征矩阵 (N, D)
        top_k: 返回前 k 个结果
    返回:
        [(路径, 距离, 置信度), ...]
    """
    query = query_feat.reshape(1, -1)
    distances = np.linalg.norm(features - query, axis=1)
    confs = cosine_confidence(query_feat, features)

    sorted_idx = np.argsort(distances)[:top_k]
    results = []
    for i in sorted_idx:
        results.append((paths[i], distances[i], float(confs[i])))
    return results


# ============================================================
# 准确率评估
# ============================================================
def evaluate(paths, features, top_k=5):
    """
    评估检索准确率 (排除自身)。

    返回: (top1_accuracy, top5_accuracy)
    """
    top1_correct = 0
    top5_correct = 0
    total = len(paths)

    for i in range(total):
        query_cat = paths[i].split('/')[0]
        query_feat = features[i]

        distances = np.linalg.norm(features - query_feat, axis=1)
        distances[i] = float('inf')      # 排除自身
        sorted_idx = np.argsort(distances)

        # Top-1
        if paths[sorted_idx[0]].split('/')[0] == query_cat:
            top1_correct += 1

        # Top-5
        top5_cats = [paths[idx].split('/')[0] for idx in sorted_idx[:top_k]]
        if query_cat in top5_cats:
            top5_correct += 1

    top1 = top1_correct / total * 100
    top5 = top5_correct / total * 100
    return top1, top5
