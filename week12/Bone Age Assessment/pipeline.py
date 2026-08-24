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
import time
from functools import wraps
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


# ---------------------------------------------------------------- 模块计时
# 用装饰器统计每个模块的推理耗时，结果累加到全局 TIMINGS（毫秒）。
# 用法：在模块方法上加 @timed("模块名")；
#       单张图片推理前调用 reset_timing()，推理后调用 report_timing() 打印耗时表。
TIMINGS = {}     # {模块名: {"ms": 累计毫秒, "n": 调用次数}}


def timed(name=None):
    """装饰器：统计被装饰函数的累计推理耗时（毫秒），累加到全局 TIMINGS。"""
    def deco(fn):
        key = name or fn.__qualname__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            dt = (time.perf_counter() - t0) * 1000.0
            rec = TIMINGS.setdefault(key, {"ms": 0.0, "n": 0})
            rec["ms"] += dt
            rec["n"] += 1
            return result

        return wrapper
    return deco


def reset_timing():
    """清空计时统计，用于开始统计单张（或一批）图片。"""
    TIMINGS.clear()


def snapshot_timing():
    """返回 {模块名: 毫秒}，不清空统计。"""
    return {k: round(v["ms"], 2) for k, v in TIMINGS.items()}


def report_timing():
    """打印各模块耗时统计表，返回 {模块名: 毫秒}。"""
    total = sum(v["ms"] for v in TIMINGS.values())
    print(f"\n{'模块':<14}{'总耗时(ms)':>12}{'次数':>8}{'平均(ms)':>12}{'占比':>10}")
    print("-" * 60)
    for k, v in sorted(TIMINGS.items(), key=lambda kv: -kv[1]["ms"]):
        avg = v["ms"] / v["n"] if v["n"] else 0.0
        pct = v["ms"] / total * 100 if total else 0.0
        print(f"{k:<14}{v['ms']:>12.1f}{v['n']:>8d}{avg:>12.2f}{pct:>9.1f}%")
    print("-" * 60)
    print(f"{'合计':<14}{total:>12.1f}")
    return {k: round(v["ms"], 2) for k, v in TIMINGS.items()}


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

    # ---- 各模块（用 @timed 统计推理耗时） ----
    @timed("1 预处理")
    def _preprocess(self, img):
        """CLAHE 预处理：灰度 → 中值滤波 → CLAHE → BGR。"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    @timed("2 检测")
    def _detect(self, img):
        """YOLO 目标检测 → (类别, 置信度, box) 列表。"""
        r = self.detector.predict(img, conf=CONF_THRESH, imgsz=640, verbose=False)[0]
        return [(self.detector.names[int(b.cls)], float(b.conf), b.xyxy[0].tolist())
                for b in r.boxes]

    @timed("3 过滤")
    def _filter(self, dets):
        """从检测结果中挑选 RUS 13 骨。"""
        return filter_13_bones(dets)

    @timed("5 计分")
    def _score(self, grades, sex, require_all):
        """13 骨等级 → RUS 总分/骨龄（或数据驱动校准骨龄）。"""
        if self.calibrated:
            import scoring as _sc
            tables = _sc.load_tables()
            feat = []
            for bone in _sc.RUS_13:
                g = grades.get(bone)
                feat.append(_sc.grade_to_score(bone, g, tables) if g is not None else None)
            X = self.calib["imputer"].transform([feat])[0].reshape(1, -1)
            months = float(self.calib["model"].predict(X)[0])
            return {"sex": sex, "total_score": None,
                    "bone_age_years": round(months / 12.0, 2),
                    "bone_age_months": round(months, 1),
                    "bone_age_range": (None, None),
                    "detail": [{"bone": b, "grade": grades.get(b), "score": f}
                               for b, f in zip(_sc.RUS_13, feat)],
                    "missing": [b for b in _sc.RUS_13 if b not in grades],
                    "method": "calibrated"}
        result = scoring.summarize(grades, sex=sex, require_all=require_all)
        result["method"] = "rus-chn"
        if result["bone_age_years"] is not None:
            result["bone_age_months"] = round(result["bone_age_years"] * 12.0, 1)
        return result

    @timed("4 分类(13骨累计)")
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

    def predict(self, image_path, sex="boy", require_all=False, do_preprocess=False):
        """完整流水线：输入 X 光片路径 → 结果 dict

        do_preprocess: 对原始 X 光片自动做 CLAHE 预处理（灰度+中值滤波+CLAHE），
        与训练数据一致。用户上传原始片时应设为 True（模型在预处理图上训练）。
        注意：已预处理过的图（detection_pre / rsna_val_pre）不要重复处理，保持 False。

        各模块耗时由 @timed 装饰器统计，随结果写入 result["timings_ms"]；
        单张图片想得到纯净的耗时，请在调用前 reset_timing()。
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        if do_preprocess:
            img = self._preprocess(img)

        dets = self._detect(img)
        bones, missing = self._filter(dets)

        grades = {}
        for rid, info in bones.items():
            grades[rid] = self._classify_crop(info["classifier"], img, info["box"])

        result = self._score(grades, sex=sex, require_all=require_all)
        result["image"] = str(image_path)
        result["n_bones"] = len(bones)
        result["detections"] = dets
        result["bones"] = {rid: {**info, "grade": grades.get(rid)} for rid, info in bones.items()}
        result["timings_ms"] = snapshot_timing()
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
    parser.add_argument("--do-preprocess", action="store_true", help="输入为原始X光片时自动CLAHE预处理")
    parser.add_argument("--demo", action="store_true", help="验证集批量演示")
    parser.add_argument("--n", type=int, default=4, help="演示图片数")
    parser.add_argument("--save", type=str, help="单图结果保存路径")
    args = parser.parse_args()

    # 默认全部关节用 ordinal（等级 MAE 更低，且与校准模型特征一致）
    pipe = Pipeline(ordinal_all=not args.use_ce, calibrated=args.calibrated)

    if args.image:
        reset_timing()   # 只统计这一张图
        t0 = time.perf_counter()
        res = pipe.predict(args.image, sex=args.sex, do_preprocess=args.do_preprocess)
        wall = (time.perf_counter() - t0) * 1000.0
        report_timing()
        print(f"\n整条流水线实际耗时: {wall:.1f} ms")
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
            reset_timing()   # 逐张清零，统计每张图的模块耗时
            res = pipe.predict(p, sex=args.sex)
            vis = pipe.visualize(res)
            fname = out_dir / f"{p.stem}.jpg"
            cv2.imwrite(str(fname), vis)
            total = sum(res["timings_ms"].values())
            print(f"[OK] {p.stem}: 13骨={res['n_bones']} RUS={res['total_score']} "
                  f"骨龄={res['bone_age_years']}y 推理耗时={total:.1f}ms -> {fname}")
        return

    print("请指定 --image <path> 或 --demo")


if __name__ == "__main__":
    main()
