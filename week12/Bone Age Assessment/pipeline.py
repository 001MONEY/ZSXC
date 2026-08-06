# -*- coding: utf-8 -*-
"""
端到端流水线：X 光片 → 预处理 → 检测 → 13 骨过滤 → 分类 → 计分 → 骨龄

模块：
  检测   : runs/bone7_ft/weights/best.pt（YOLOv8n，7 类）
  过滤   : filter_bones.filter_13_bones（RUS 13 骨挑选）
  分类   : models/classification/{关节}_best.pt 或 {关节}_ordinal_best.pt
           （默认：Ulna 用序数模型，其余用分类模型；可用 --ordinal-all 切换）
  计分   : scoring.py（注意当前为占位表，需替换官方 RUS 数据）

用法：
    python pipeline.py --image <path> [--sex boy|girl]
    python pipeline.py --demo                    # 在验证集上批量跑并可视化
    python pipeline.py --demo --n 6
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

import config
import scoring
from filter_bones import filter_13_bones
from train_ordinal import OrdinalHead, predict_grade

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
PAD = 6          # 裁剪时向外扩的像素
CONF_THRESH = 0.25


# ---------------------------------------------------------------- 模型加载
class Pipeline:
    def __init__(self, det_weights=None, ordinal_all=False, calibrated=False):
        from ultralytics import YOLO
        det_weights = det_weights or (config.BAA_DIR / "runs" / "bone7_ft" / "weights" / "best.pt")
        self.detector = YOLO(str(det_weights))
        self.calibrated = calibrated
        self.calib = None
        if calibrated:
            import joblib
            pkl = config.BAA_DIR / "models" / "bone_age_regressor.pkl"
            if not pkl.exists():
                raise FileNotFoundError(f"缺少校准模型: {pkl}（先运行 calibrate.py）")
            self.calib = joblib.load(pkl)
            print(f"[OK] 已加载数据驱动骨龄模型: {self.calib['name']} (测试MAE={self.calib['test_mae_months']:.1f}月)")
        self.clfs = {}
        self.grade_lists = {}
        self.ordinals = {}
        for joint in config.JOINT_TYPES:
            ordinal = ordinal_all or joint == "Ulna"      # 默认 Ulna 用序数模型
            self.clfs[joint], self.grade_lists[joint], self.ordinals[joint] = self._load_clf(joint, ordinal)

    def _load_clf(self, joint, ordinal):
        path = config.BAA_DIR / "models" / "classification" / (
            f"{joint}_ordinal_best.pt" if ordinal else f"{joint}_best.pt")
        if not path.exists():
            raise FileNotFoundError(f"缺少分类模型: {path}")
        ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
        gl = ckpt.get("grade_list")
        if gl is None:
            root = config.CLASSIFICATION_PRE / joint
            gl = sorted(int(c) for c in datasets.ImageFolder(root / "val").classes)
        n = len(gl)
        if ordinal:
            m = models.resnet18()
            m.fc = OrdinalHead(m.fc.in_features, n)
        else:
            m = models.resnet18()
            m.fc = nn.Linear(m.fc.in_features, n)
        m.load_state_dict(ckpt["state_dict"])
        m.to(DEVICE).eval()
        return m, gl, ordinal

    def _classify_crop(self, joint, img, box):
        m, gl, ordinal = self.clfs[joint], self.grade_lists[joint], self.ordinals[joint]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1 - PAD), max(0, y1 - PAD)
        x2, y2 = min(img.shape[1], x2 + PAD), min(img.shape[0], y2 + PAD)
        crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        t = CROP_TF(transforms.ToPILImage()(crop)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = m(t)
            if ordinal:
                return int(predict_grade(out, gl).item())
            return int(gl[out.argmax(1).item()])

    def predict(self, image_path, sex="boy", require_all=False):
        """完整流水线：输入 X 光片路径 → 结果 dict"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")

        r = self.detector.predict(str(image_path), conf=CONF_THRESH,
                                  imgsz=640, verbose=False)[0]
        dets = [(self.detector.names[int(b.cls)], float(b.conf), b.xyxy[0].tolist())
                for b in r.boxes]
        bones, missing = filter_13_bones(dets)

        grades = {}
        for rid, info in bones.items():
            grades[rid] = self._classify_crop(info["classifier"], img, info["box"])

        if self.calibrated:
            # 数据驱动校准：13 骨得分 → 骨龄(月)
            import scoring as _sc
            tables = _sc.load_tables()
            feat = []
            for bone in _sc.RUS_13:
                g = grades.get(bone)
                feat.append(_sc.grade_to_score(bone, g, tables) if g is not None else None)
            X = self.calib["imputer"].transform([feat])[0].reshape(1, -1)
            months = float(self.calib["model"].predict(X)[0])
            result = {"sex": sex, "total_score": None, "bone_age_years": round(months / 12.0, 2),
                      "bone_age_months": round(months, 1), "bone_age_range": (None, None),
                      "detail": [{"bone": b, "grade": grades.get(b), "score": f}
                                  for b, f in zip(_sc.RUS_13, feat)],
                      "missing": [b for b in _sc.RUS_13 if b not in grades],
                      "method": "calibrated"}
        else:
            result = scoring.summarize(grades, sex=sex, require_all=require_all)
            result["method"] = "rus-chn"
            if result["bone_age_years"] is not None:
                result["bone_age_months"] = round(result["bone_age_years"] * 12.0, 1)
        result["image"] = str(image_path)
        result["n_bones"] = len(bones)
        result["detections"] = dets
        result["bones"] = {rid: {**info, "grade": grades.get(rid)} for rid, info in bones.items()}
        return result

    # ---- 可视化 ----
    def visualize(self, result, img=None):
        if img is None:
            img = cv2.imread(result["image"])
        out = img.copy()
        colors = {1: (0, 200, 0), 2: (200, 200, 0), 3: (0, 150, 255),
                  4: (255, 0, 200), 5: (0, 0, 255)}
        for rid, info in result["bones"].items():
            x1, y1, x2, y2 = [int(v) for v in info["box"]]
            c = colors.get(info["finger"], (0, 255, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), c, 3)
            cv2.putText(out, f"{rid} g{info['grade']}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        txt = (f"Sex={result['sex']}  Method={result.get('method','?')}  "
               f"BoneAge={result['bone_age_years']}y"
               + (f"({result['bone_age_months']}mo)" if result.get('bone_age_months') else "")
               + (f"  RUS={result['total_score']}" if result.get('total_score') is not None else "")
               + f"  missing={len(result['missing'])}")
        cv2.putText(out, txt, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return out


# ---------------------------------------------------------------- 主流程
def main():
    parser = argparse.ArgumentParser(description="骨龄端到端流水线")
    parser.add_argument("--image", type=str, help="单张 X 光片路径")
    parser.add_argument("--sex", default="boy", choices=["boy", "girl"])
    parser.add_argument("--ordinal-all", action="store_true", help="全部关节用序数模型（已默认）")
    parser.add_argument("--use-ce", action="store_true", help="改用普通分类模型（对比实验用）")
    parser.add_argument("--calibrated", action="store_true", help="用数据驱动校准模型（推荐）")
    parser.add_argument("--demo", action="store_true", help="验证集批量演示")
    parser.add_argument("--n", type=int, default=4, help="演示图片数")
    parser.add_argument("--save", type=str, help="单图结果保存路径")
    args = parser.parse_args()

    # 默认全部关节用 ordinal（等级 MAE 更低，且与校准模型特征一致）
    pipe = Pipeline(ordinal_all=not args.use_ce, calibrated=args.calibrated)

    if args.image:
        res = pipe.predict(args.image, sex=args.sex)
        print(f"图片: {res['image']}")
        print(f"检出骨头: {res['n_bones']}/13  缺失: {res['missing']}")
        print("13 骨等级:")
        for rid in scoring.RUS_13:
            if rid in res["bones"]:
                b = res["bones"][rid]
                print(f"  {rid:<7} 等级 {b['grade']:<3} 得分 {res['detail'] and next((d['score'] for d in res['detail'] if d['bone']==rid), '-')}")
        print(f"RUS 总分: {res['total_score']}   骨龄({res['sex']}): {res['bone_age_years']} 岁")
        if args.save:
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), pipe.visualize(res))
            print(f"可视化已保存: {save_path}")
        return

    if args.demo:
        out_dir = config.BAA_DIR / "output" / "pipeline_demo"
        out_dir.mkdir(parents=True, exist_ok=True)
        imgs = sorted((config.DETECTION_PRE / "images" / "val").glob("*.png"))[:args.n]
        for p in imgs:
            res = pipe.predict(p, sex=args.sex)
            vis = pipe.visualize(res)
            fname = out_dir / f"{p.stem}.jpg"
            cv2.imwrite(str(fname), vis)
            print(f"[OK] {p.stem}: 13骨={res['n_bones']} RUS={res['total_score']} "
                  f"骨龄={res['bone_age_years']}y -> {fname}")
        return

    print("请指定 --image <path> 或 --demo")


if __name__ == "__main__":
    main()
