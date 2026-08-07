# -*- coding: utf-8 -*-
"""
Qt 界面推理后端（本地服务）

Qt 界面通过 HTTP 调用本服务完成骨龄评估，避免每次启动 Python 进程。
模型常驻内存，推理快。

接口:
    POST /predict  body: {"image": "图片绝对路径", "sex": "boy|girl"}
    返回 JSON: {bone_age_years, bone_age_months, n_bones, missing,
                method, vis_path, detail:[{bone, grade, score}]}

启动:
    python qt_server.py [--port 8765]
"""
import argparse
import json
from pathlib import Path

import cv2
from flask import Flask, jsonify, request

from pipeline import Pipeline

app = Flask(__name__)
PIPE = Pipeline(ordinal_all=True, calibrated=True)
VIS_DIR = Path(__file__).resolve().parent / "output" / "qt_vis"
VIS_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    img_path = data.get("image")
    sex = data.get("sex", "boy")
    if not img_path or not Path(img_path).exists():
        return jsonify({"error": f"图片不存在: {img_path}"}), 400
    try:
        res = PIPE.predict(img_path, sex=sex, do_preprocess=True)
    except Exception as e:                      # noqa: BLE001
        return jsonify({"error": str(e)}), 500

    vis = PIPE.visualize(res)
    vis_path = VIS_DIR / f"{Path(img_path).stem}_vis.png"
    cv2.imwrite(str(vis_path), vis)

    return jsonify({
        "bone_age_years": res["bone_age_years"],
        "bone_age_months": res["bone_age_months"],
        "n_bones": res["n_bones"],
        "missing": res["missing"],
        "method": res["method"],
        "vis_path": str(vis_path),
        "detail": [{"bone": d["bone"], "grade": d["grade"], "score": d["score"]}
                   for d in res["detail"]],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        # 不启动服务，直接测一次推理（验证模型/服务可用）
        import tempfile
        from pipeline import Pipeline as P
        pipe = P(ordinal_all=True, calibrated=True)
        r = pipe.predict(str(Path(__file__).resolve().parent
                             / "datasets/detection_pre/images/train/1526.png"),
                         sex="boy", do_preprocess=False)
        print(f"[SMOKE] {r['bone_age_years']} 岁, {r['n_bones']}/13 骨")
    else:
        app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
