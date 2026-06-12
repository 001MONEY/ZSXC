"""
第三题 - IOU 示意图
绘制 box 与 boxes 中每个矩形的交并比可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm

# 尝试设置中文字体，避免乱码
for font_name in ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'Noto Sans CJK SC']:
    try:
        fm.findfont(font_name, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue

def compute_iou(box, boxes):
    x1_inter = np.maximum(box[0], boxes[:, 0])
    y1_inter = np.maximum(box[1], boxes[:, 1])
    x2_inter = np.minimum(box[2], boxes[:, 2])
    y2_inter = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0, x2_inter - x1_inter)
    inter_h = np.maximum(0, y2_inter - y1_inter)
    inter_area = inter_w * inter_h

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union_area = box_area + boxes_area - inter_area
    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    # 返回每个 box 的交集参数
    inter_list = list(zip(x1_inter, y1_inter, x2_inter, y2_inter, inter_w, inter_h))
    return iou, inter_list


def draw_iou_diagram(box, boxes, ious, inter_list):
    """绘制 IOU 示意图"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
    labels = [f'boxes[{i}]' for i in range(len(boxes))]

    for idx, ax in enumerate(axes):
        box_color = '#333333'
        boxes_color = colors[idx]
        inter_color = '#9b59b6'
        iou_val = ious[idx]
        x1i, y1i, x2i, y2i, iw, ih = inter_list[idx]

        # --- box (虚线, 灰色) ---
        bx, by = box[0], box[1]
        bw, bh = box[2] - box[0], box[3] - box[1]
        rect_box = patches.Rectangle(
            (bx, by), bw, bh, linewidth=2, edgecolor=box_color,
            facecolor='#cccccc', linestyle='--', alpha=0.6, label='box'
        )
        ax.add_patch(rect_box)

        # --- boxes[idx] (实线, 彩色) ---
        b2 = boxes[idx]
        bx2, by2 = b2[0], b2[1]
        bw2, bh2 = b2[2] - b2[0], b2[3] - b2[1]
        rect_boxes = patches.Rectangle(
            (bx2, by2), bw2, bh2, linewidth=2, edgecolor=boxes_color,
            facecolor=boxes_color, alpha=0.3, label=labels[idx]
        )
        ax.add_patch(rect_boxes)

        # --- 交集区域 (紫色填充, 如果有交集) ---
        if iw > 0 and ih > 0:
            rect_inter = patches.Rectangle(
                (x1i, y1i), iw, ih, linewidth=2, edgecolor=inter_color,
                facecolor=inter_color, alpha=0.5, label=f'交集'
            )
            ax.add_patch(rect_inter)

        # --- 文字标注 ---
        # box 中心标注
        cx_box, cy_box = bx + bw / 2, by + bh / 2
        ax.text(cx_box, cy_box, 'box', fontsize=9, color=box_color,
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7))

        # boxes 中心标注
        cx_b2, cy_b2 = bx2 + bw2 / 2, by2 + bh2 / 2
        ax.text(cx_b2, cy_b2, labels[idx], fontsize=9, color=boxes_color,
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7))

        # IOU 值显示在左上角
        ax.text(0, 1.02, f'IOU = {iou_val:.4f}', transform=ax.transAxes,
                fontsize=13, fontweight='bold', color='#c0392b',
                va='bottom', ha='left')

        # --- 设置坐标轴 ---
        ax.set_xlim(10, 100)
        ax.set_ylim(10, 100)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # 让 y 轴从上到下，符合图像坐标习惯
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(loc='lower right', fontsize=8)

    fig.suptitle(' box 与 boxes 的 IOU 示意图', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/iou_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()


# ========== 主程序 ==========
box = np.array([20, 20, 60, 60])
boxes = np.array([[30, 30, 70, 70],
                  [30, 30, 50, 50],
                  [70, 70, 90, 90],
                  [60, 20, 80, 60]])

# 计算 IOU
ious, inter_rects = compute_iou(box, boxes)

# 打印结果
print("IOU 计算结果：")
for i, iou in enumerate(ious):
    print(f"  boxes[{i}] 的 IOU = {iou:.4f}")

# 画图
draw_iou_diagram(box, boxes, ious, inter_rects)
