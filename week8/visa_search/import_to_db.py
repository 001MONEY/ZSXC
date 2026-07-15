"""
VisA 数据集 → 导入 MySQL（含特征向量）

将 VisA 数据集的图片信息 + 特征向量导入到 MySQL 数据库。
特征向量由 MobileNetV2 模型提取（1280 维），以 BLOB 二进制格式存储。
"""
import os
import sys
import numpy as np
import pymysql

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import VISA_ROOT, CATEGORIES, SUBSETS, DEVICE
from model import FeatureExtractor

# ============================================================
# 数据库配置（根据实际情况修改）
# ============================================================
config = {
    "host": "localhost",            # MySQL 服务器地址
    "port": 3306,                   # 端口号（默认 3306）
    "user": "root",                 # 用户名  ← 修改这里
    "password": "8888",             # 密码    ← 修改这里
    "database": "visa",             # 数据库名 ← 修改这里
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor
}

FEAT_DIM = 1280                      # MobileNetV2 特征维度


# ============================================================
# 1. 创建数据库和表
# ============================================================
def create_database():
    """创建数据库 visa（如果不存在）"""
    conn = pymysql.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        charset=config["charset"],
    )
    with conn.cursor() as cursor:
        cursor.execute("CREATE DATABASE IF NOT EXISTS visa DEFAULT CHARACTER SET utf8mb4")
        print("  数据库 visa 已就绪")
    conn.close()


def create_table():
    """创建图片信息表（含 features 字段）"""
    conn = pymysql.connect(**config)
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id          INT AUTO_INCREMENT PRIMARY KEY,
                category    VARCHAR(50)  NOT NULL COMMENT '类别 (candle/capsules/...)',
                subset      VARCHAR(10)  NOT NULL COMMENT '子集 (Normal/Anomaly)',
                filename    VARCHAR(100) NOT NULL COMMENT '图片文件名',
                full_path   VARCHAR(300) NOT NULL COMMENT '图片绝对路径',
                features    BLOB                  COMMENT '1280 维特征向量 (binary)',
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '导入时间',

                INDEX idx_category (category),
                INDEX idx_subset (subset)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        print("  表 images 已就绪（含 features 字段）")
    conn.close()


# ============================================================
# 2. 遍历 VisA 并插入数据（含特征提取）
# ============================================================
def import_images():
    """
    遍历 VisA 数据集，提取特征并写入 MySQL。

    流程:
      1. 加载 MobileNetV2 特征提取器
      2. 清空旧数据
      3. 遍历每个类别/子集，提取每张图片的特征
      4. 将图片信息 + 特征向量存入数据库

    每次运行前会清空旧数据，避免重复导入。
    """
    # ---- 加载特征提取器 ----
    print("  加载特征提取模型...")
    extractor = FeatureExtractor(model_name='mobilenet_v2')

    conn = pymysql.connect(**config)
    cursor = conn.cursor()

    # 清空旧数据
    cursor.execute("TRUNCATE TABLE images")
    print("  已清空旧数据")

    # 遍历数据集
    total = 0
    errors = 0
    for cat in CATEGORIES:
        cat_path = os.path.join(VISA_ROOT, cat, 'Data', 'Images')
        if not os.path.exists(cat_path):
            print(f"  跳过 {cat}: 路径不存在")
            continue

        for subset in SUBSETS:
            subset_path = os.path.join(cat_path, subset)
            if not os.path.exists(subset_path):
                continue

            for img_name in sorted(os.listdir(subset_path)):
                full_path = os.path.join(subset_path, img_name)

                # 提取特征向量
                try:
                    feat = extractor.extract(full_path)  # numpy 1280-dim
                    feat_bytes = feat.tobytes()          # → binary for BLOB
                except Exception as e:
                    print(f"  [WARN] 特征提取失败: {full_path} → {e}")
                    feat_bytes = None
                    errors += 1

                sql = """INSERT INTO images (category, subset, filename, full_path, features)
                         VALUES (%s, %s, %s, %s, %s)"""
                cursor.execute(sql, (cat, subset, img_name, full_path, feat_bytes))
                total += 1

                if total % 200 == 0:
                    print(f"  已处理: {total} 张...")

    conn.commit()
    print(f"\n  共导入 {total} 条记录")
    if errors:
        print(f"  [WARN] {errors} 张图片特征提取失败")

    # 打印统计
    cursor.execute("""
        SELECT category, subset, COUNT(*) as cnt
        FROM images
        GROUP BY category, subset
        ORDER BY category, subset
    """)
    print(f"\n  导入统计:")
    for row in cursor.fetchall():
        print(f"    {row['category']:15s} / {row['subset']:7s}: {row['cnt']:4d} 张")

    cursor.close()
    conn.close()
    return total


# ============================================================
# 3. 查询示例
# ============================================================
def query_examples():
    """展示几种常用查询"""
    conn = pymysql.connect(**config)
    cursor = conn.cursor()

    print(f"\n{'=' * 60}")
    print("查询示例")
    print("=" * 60)

    # 查询所有类别
    cursor.execute("SELECT DISTINCT category FROM images ORDER BY category")
    cats = [row['category'] for row in cursor.fetchall()]
    print(f"\n  所有类别 ({len(cats)}): {', '.join(cats)}")

    # 查询 Normal/Anomaly 数量
    cursor.execute("""
        SELECT subset, COUNT(*) as cnt
        FROM images
        GROUP BY subset
    """)
    for row in cursor.fetchall():
        print(f"  {row['subset']:7s}: {row['cnt']} 张")

    # 查询某个类别的前 5 张
    cursor.execute("""
        SELECT category, subset, filename
        FROM images
        WHERE category = 'candle' AND subset = 'Normal'
        LIMIT 5
    """)
    print(f"\n  candle/Normal 前 5 张:")
    for row in cursor.fetchall():
        print(f"    {row['filename']}")

    cursor.close()
    conn.close()


# ============================================================
# 4. 从数据库加载特征（代替文本文件方式）
# ============================================================
def load_features_from_db(limit=None):
    """
    从 MySQL 加载图片路径和特征向量。

    参数:
        limit: 限制加载条数（None=全部）
    返回:
        (paths_list, features_matrix)
            paths:   ['cat/subset/fname', ...]
            features: np.ndarray shape (N, 1280)
    """
    conn = pymysql.connect(**config)
    cursor = conn.cursor()

    sql = "SELECT category, subset, filename, features FROM images WHERE features IS NOT NULL"
    if limit:
        sql += f" LIMIT {limit}"
    cursor.execute(sql)

    paths = []
    feats = []
    for row in cursor.fetchall():
        rel_path = f"{row['category']}/{row['subset']}/{row['filename']}"
        feat = np.frombuffer(row['features'], dtype=np.float64)
        paths.append(rel_path)
        feats.append(feat)

    cursor.close()
    conn.close()

    print(f"  从数据库加载: {len(paths)} 条特征, 维度 {feats[0].shape[0]}")
    return paths, np.array(feats)


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("VisA 数据集 → MySQL 导入")
    print("=" * 60)

    # Step 1: 创建数据库
    print("\n[1] 创建数据库")
    print("-" * 40)
    create_database()

    # Step 2: 创建表
    print("\n[2] 创建表")
    print("-" * 40)
    create_table()

    # Step 3: 导入数据
    print("\n[3] 导入图片信息")
    print("-" * 40)
    total = import_images()

    # Step 4: 查询示例
    query_examples()

    print(f"\n{'=' * 60}")
    print(f"导入完成！共 {total} 条记录")
    print("提示: 加载特征用 load_features_from_db()")
    print("=" * 60)


if __name__ == '__main__':
    main()
