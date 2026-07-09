"""
VisA 以图搜图 — Flask Web 应用
===============================
功能: 在浏览器中浏览图片、点击搜索、上传图片搜索
"""
import os
import sys
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import io

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, FEAT_FILE, VISA_ROOT
from model import FeatureExtractor
from feature_lib import load, search

# ============================================================
# 初始化
# ============================================================
app = Flask(__name__)

print("正在加载模型...")
extractor = FeatureExtractor(model_name='mobilenet_v2')

print("正在加载特征库...")
paths, features = load(FEAT_FILE)

# 从特征库路径中构建 类别→文件列表 的索引
# 路径格式: cat/subset/filename
from collections import defaultdict
image_index = defaultdict(list)   # "cat/subset" -> [full_path, ...]
for p in paths:
    parts = p.split('/')
    cat, subset, fname = parts[0], parts[1], parts[2]
    VISA_ROOT = VISA_ROOT  # 从 config 导入
    full_path = os.path.join(VISA_ROOT, cat, 'Data', 'Images', subset, fname)
    image_index[f"{cat}/{subset}"].append(full_path)

categories = sorted(set(p.split('/')[0] for p in paths))

print(f"  已加载 {len(paths)} 条特征, {len(categories)} 个类别")
print(f"  Flask 服务启动: http://127.0.0.1:5000")


# ============================================================
# 路由
# ============================================================

@app.route('/')
def index():
    return render_template('index.html', categories=categories)


@app.route('/api/images')
def api_images():
    """返回某类别/子集的图片列表"""
    category = request.args.get('category', 'candle')
    subset = request.args.get('subset', 'Normal')
    key = f"{category}/{subset}"

    imgs = []
    for full_path in image_index.get(key, []):
        imgs.append({
            'full_path': full_path,
            'filename': os.path.basename(full_path),
        })
    return jsonify({'images': imgs})


@app.route('/image/<path:img_path>')
def serve_image(img_path):
    """提供图片文件"""
    if os.path.exists(img_path):
        return send_file(img_path)
    return '', 404


@app.route('/api/search', methods=['POST'])
def api_search():
    """以图搜图: 传入图片路径, 返回 Top-5 结果"""
    data = request.get_json()
    img_path = data['image_path']

    if not os.path.exists(img_path):
        return jsonify({'error': '图片不存在'}), 404

    # 提取特征
    query_feat = extractor.extract(img_path)

    # 检索
    results = search(query_feat, paths, features, top_k=5)

    # 构造返回
    # 从路径中提取类别: .../visa_root/category/Data/Images/...
    rel_path = os.path.relpath(img_path, VISA_ROOT)
    query_cat = rel_path.split(os.sep)[0]
    query_filename = os.path.basename(img_path)

    result_list = []
    for p, d in results:
        res_cat = p.split('/')[0]
        result_list.append({
            'path': _resolve_path(p),
            'category': res_cat,
            'distance': float(d),
            'correct': res_cat == query_cat,
        })

    return jsonify({
        'query_url': f"/image/{img_path}",
        'query_name': query_filename,
        'results': result_list,
    })


@app.route('/api/search_upload', methods=['POST'])
def api_search_upload():
    """以图搜图: 上传图片, 返回 Top-5 结果"""
    if 'image' not in request.files:
        return jsonify({'error': '未上传图片'}), 400

    file = request.files['image']
    img_bytes = file.read()

    # 保存到临时文件
    temp_dir = 'temp_uploads'
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename or 'query.jpg')
    with open(temp_path, 'wb') as f:
        f.write(img_bytes)

    # 提取特征
    query_feat = extractor.extract(temp_path)

    # 检索
    results = search(query_feat, paths, features, top_k=5)

    # 构造返回
    result_list = []
    for p, d in results:
        res_cat = p.split('/')[0]
        result_list.append({
            'path': _resolve_path(p),
            'category': res_cat,
            'distance': float(d),
            'correct': False,   # 上传图片无真实标签
        })

    return jsonify({
        'query_url': f"/image/{temp_path}",
        'query_name': file.filename or 'upload',
        'results': result_list,
    })


def _resolve_path(rel_path):
    """将 rel_path (cat/subset/fname) 转为绝对路径"""
    parts = rel_path.split('/')
    return os.path.join(VISA_ROOT, parts[0], 'Data', 'Images', parts[1], parts[2])


# ============================================================
# 启动
# ============================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
