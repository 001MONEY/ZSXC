# -*- coding: utf-8 -*-
"""
位置过滤：从 7 类检测结果中挑选 RUS 计分需要的 13 根骨头

输入：检测结果（类别 + 置信度 + 框）
输出：13 根 RUS 骨头的 {rus_id, 检测框, 手指号, 分类模型关节}

RUS 13 根骨头（TW3 计分标准）：
  桡骨(Radius)、尺骨(Ulna)、
  第 1/3/5 掌骨、第 1/3/5 近节指骨、第 3/5 中节指骨、第 1/3/5 远节指骨

算法：
  1. Radius / Ulna 直接取置信度最高者
  2. 手指锚点：MCPFirst（拇指，已知）+ 4 个 MCP
  3. 朝向判定：MCPFirst 在 MCP 群哪一侧 → 决定手指 2~5 的 x 排序方向
  4. 指骨分配：各节指骨按 x 排序，与锚点列做【保序最近匹配】（可容忍漏检/中间缺失）
  5. 挑选 RUS 13 根并映射到对应分类模型关节

用法：
    python filter_bones.py --demo             # 跑检测+过滤，保存可视化到 output/filter_demo/
    python filter_bones.py --demo --n 6       # 指定张数
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config

# ---------------------------------------------------------------- RUS 骨头定义
# (rus_id, 手指, 检测类别, 分类模型关节)
RUS_BONES = [
    ("Radius", None, "Radius", "Radius"),
    ("Ulna",   None, "Ulna",   "Ulna"),
    ("MCP-1",  1, "MCPFirst",      "MCPFirst"),
    ("MCP-3",  3, "MCP",           "MCP"),
    ("MCP-5",  5, "MCP",           "MCP"),
    ("PIP-1",  1, "ProximalPhalanx", "PIPFirst"),
    ("PIP-3",  3, "ProximalPhalanx", "PIP"),
    ("PIP-5",  5, "ProximalPhalanx", "PIP"),
    ("MIP-3",  3, "MiddlePhalanx",   "MIP"),
    ("MIP-5",  5, "MiddlePhalanx",   "MIP"),
    ("DIP-1",  1, "DistalPhalanx",   "DIPFirst"),
    ("DIP-3",  3, "DistalPhalanx",   "DIP"),
    ("DIP-5",  5, "DistalPhalanx",   "DIP"),
]
# 各节指骨的类别名、期望数量、是否含拇指
SEGMENTS = [
    ("ProximalPhalanx", 5, True),
    ("MiddlePhalanx",   4, False),
    ("DistalPhalanx",   5, True),
]


def centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def match_order_preserving(items_x, anchor_x):
    """保序最近匹配：items 按 x 升序，anchor 按 x 升序，返回每个 item 的 anchor 索引（严格递增）。
    可容忍漏检（中间缺失），即 item 数 <= anchor 数。用 DP 求最小 |x 差| 之和。
    若 item 数 > anchor 数（多余检测/误检），只匹配前 n 个（多出的多为远处误检）。"""
    m, n = len(items_x), len(anchor_x)
    if m == 0:
        return []
    if m > n:
        items_x = items_x[:n]       # 丢弃多余（通常为远处误检）
        m = n
    if m == n:
        return list(range(n))          # 数量相等时保序双射即按序对应
    INF = float("inf")
    dp = [[INF] * n for _ in range(m)]
    par = [[-1] * n for _ in range(m)]
    for j in range(n):
        dp[0][j] = abs(items_x[0] - anchor_x[j])
    for i in range(1, m):
        for j in range(i, n):
            best_k, best_c = -1, INF
            for k in range(i - 1, j):
                if dp[i - 1][k] < best_c:
                    best_c, best_k = dp[i - 1][k], k
            if best_k >= 0:
                dp[i][j] = best_c + abs(items_x[i] - anchor_x[j])
                par[i][j] = best_k
    j = min(range(m - 1, n), key=lambda j: dp[m - 1][j])
    result = [0] * m
    for i in range(m - 1, -1, -1):
        result[i] = j
        if i > 0:
            j = par[i][j]
    return result


# ---------------------------------------------------------------- 核心过滤
def filter_13_bones(detections):
    """
    detections: [(类别名, 置信度, box(x1,y1,x2,y2)), ...]
    返回: (bones, missing)
      bones: dict rus_id -> {box, conf, finger, det_cls, classifier}
      missing: 缺失的 rus_id 列表
    """
    groups = defaultdict(list)
    for name, conf, box in detections:
        groups[name].append((conf, box))

    def best_box(cls):
        lst = groups.get(cls, [])
        return max(lst, key=lambda t: t[0]) if lst else None

    bones, missing = {}, []

    # ---- 1. Radius / Ulna ----
    for cls in ("Radius", "Ulna"):
        hit = best_box(cls)
        if hit:
            bones[cls] = {"box": hit[1], "conf": hit[0], "finger": None,
                          "det_cls": cls, "classifier": cls}
        else:
            missing.append(cls)

    # ---- 2. 手指锚点 ----
    mcpf = best_box("MCPFirst")
    mcp_list = groups.get("MCP", [])
    if not mcp_list:
        missing += ["MCP-3", "MCP-5", "PIP-1", "PIP-3", "PIP-5",
                    "MIP-3", "MIP-5", "DIP-1", "DIP-3", "DIP-5"]
        return bones, missing

    mcp_sorted = sorted(mcp_list, key=lambda t: centroid(t[1])[0])   # 按 x 升序
    mcp_x = [centroid(t[1])[0] for t in mcp_sorted]

    if mcpf is not None:
        thumb_x = centroid(mcpf[1])[0]
        thumb_left = thumb_x < (mcp_x[0] + mcp_x[-1]) / 2.0
    else:
        thumb_left = True    # 缺 MCPFirst 时默认拇指在左，稍后尽力补救

    # 锚点列：[(finger, x)] 按 x 升序（用实际 MCP 数量，漏检时降级不崩溃）
    n_mcp = len(mcp_x)
    if thumb_left:
        anchors = [(1, centroid(mcpf[1])[0])] if mcpf else []
        anchors += [(i + 2, mcp_x[i]) for i in range(n_mcp)]          # 手指 2,3,4,5
    else:
        anchors = [(5 - i, mcp_x[i]) for i in range(n_mcp)]          # 手指 5,4,3,2
        if mcpf:
            anchors.append((1, centroid(mcpf[1])[0]))                # 拇指在最后
    anchor_x = [x for _, x in anchors]
    anchor_f = [f for f, _ in anchors]

    # ---- 3. 掌骨 ----
    # 注意：MCP 只对应手指 2~5（拇指掌骨是单独的 MCPFirst 类），
    # 不要用包含拇指的 anchor_f，否则会整体错位一格（4 个 MCP 分到 1,2,3,4）
    mcp_fingers = [2, 3, 4, 5] if thumb_left else [5, 4, 3, 2]
    for i, (conf, box) in enumerate(mcp_sorted):
        finger = mcp_fingers[i] if i < len(mcp_fingers) else None
        if finger in (3, 5):
            bones[f"MCP-{finger}"] = {"box": box, "conf": conf, "finger": finger,
                                      "det_cls": "MCP", "classifier": "MCP"}
    if mcpf:
        bones["MCP-1"] = {"box": mcpf[1], "conf": mcpf[0], "finger": 1,
                          "det_cls": "MCPFirst", "classifier": "MCPFirst"}
    # MCP 缺失追踪
    for rus_id in ("MCP-1", "MCP-3", "MCP-5"):
        if rus_id not in bones:
            missing.append(rus_id)

    # ---- 4. 各节指骨分配 ----
    for seg, n_exp, has_thumb in SEGMENTS:
        seg_list = groups.get(seg, [])
        if not seg_list:
            for rus_id, finger, det_cls, clf in RUS_BONES:
                if det_cls == seg and (rus_id.startswith("PIP-") or rus_id.startswith("DIP-")):
                    if rus_id not in bones:
                        missing.append(rus_id)
            continue
        seg_sorted = sorted(seg_list, key=lambda t: centroid(t[1])[0])
        seg_x = [centroid(t[1])[0] for t in seg_sorted]
        idxs = match_order_preserving(seg_x, anchor_x)
        # 注意：多余检测会被 match 截断，只遍历匹配到的那部分
        for i, (conf, box) in enumerate(seg_sorted[:len(idxs)]):
            finger = anchor_f[idxs[i]]
            rus_id = None
            if seg == "ProximalPhalanx":
                rus_id = f"PIP-{finger}"
            elif seg == "MiddlePhalanx":
                rus_id = f"MIP-{finger}"
            else:
                rus_id = f"DIP-{finger}"
            if rus_id in {r[0] for r in RUS_BONES}:
                bones[rus_id] = {"box": box, "conf": conf, "finger": finger,
                                 "det_cls": seg, "classifier": _classifier_of(rus_id)}
        # 记录缺失的 RUS 骨头
        for rus_id, finger, det_cls, clf in RUS_BONES:
            if det_cls == seg and rus_id not in bones:
                missing.append(rus_id)

    return bones, missing


def _classifier_of(rus_id):
    for r in RUS_BONES:
        if r[0] == rus_id:
            return r[3]
    return None


# ---------------------------------------------------------------- 可视化
def draw_bones(img, bones, missing=None):
    out = img.copy()
    colors = {1: (0, 200, 0), 2: (200, 200, 0), 3: (0, 150, 255),
              4: (255, 0, 200), 5: (0, 0, 255)}
    for rus_id, info in bones.items():
        x1, y1, x2, y2 = [int(v) for v in info["box"]]
        color = colors.get(info["finger"], (255, 255, 255)) if info["finger"] else (0, 255, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = f"{rus_id} ({info['classifier']})"
        cv2.putText(out, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
    if missing:
        txt = "missing: " + ",".join(missing)
        cv2.putText(out, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return out


# ---------------------------------------------------------------- 演示
def demo(n=4, conf_thresh=0.25):
    from ultralytics import YOLO

    det_model = YOLO(str(config.BAA_DIR / "runs" / "bone7_ft" / "weights" / "best.pt"))
    out_dir = config.BAA_DIR / "output" / "filter_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted((config.DETECTION_PRE / "images" / "val").glob("*.png"))[:n]
    for p in imgs:
        r = det_model.predict(str(p), conf=conf_thresh, imgsz=640, verbose=False)[0]
        dets = [(det_model.names[int(b.cls)], float(b.conf), b.xyxy[0].tolist())
                for b in r.boxes]
        bones, missing = filter_13_bones(dets)
        img = cv2.imread(str(p))
        vis = draw_bones(img, bones, missing)
        n_found = len(bones)
        cv2.putText(vis, f"{p.stem}  found={n_found}/13  missing={len(missing)}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        out = out_dir / f"{p.stem}.jpg"
        cv2.imwrite(str(out), vis)
        print(f"[OK] {p.stem}: 检出 {n_found}/13, 缺失 {missing if missing else '无'} -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="13 根 RUS 骨头位置过滤")
    parser.add_argument("--demo", action="store_true", help="跑检测+过滤并可视化")
    parser.add_argument("--n", type=int, default=4, help="演示图片数")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    args = parser.parse_args()
    if args.demo:
        demo(args.n, args.conf)
    else:
        print("请用 --demo 运行演示")
