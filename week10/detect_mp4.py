# ============================================================
# YOLOv3 视频推理 —— 对 mp4 逐帧检测并画框
# 改进：向量化解码 + torchvision.ops.nms + sigmoid分类
# ============================================================
import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np
import os
import cv2

# -------------------- 配置 --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 416
NUM_CLASSES = 3
CONF_THRESHOLD = 0.58  # obj_conf 下限
NMS_THRESHOLD = 0.45

CLASSES = ["red", "black", "white"]
CLASS_COLORS = {
    0: (0, 0, 255),    # red
    1: (0, 0, 0),      # black
    2: (255, 255, 255),# white
}

# Anchor（必须与训练时一致）
ANCHORS_GROUPS = {
    13: [[87, 74], [86, 104], [105, 88]],
    26: [[80, 50], [49, 84], [72, 84]],
    52: [[38, 58], [53, 43], [60, 64]],
}

VIDEO_PATH = r"1.mp4"
OUTPUT_PATH = r"output_detected.mp4"


# -------------------- 模型组件 --------------------
class CBL(nn.Module):
    def __init__(self, c_in, c_out, k, s):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, padding=k // 2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, x):
        return self.mod(x)

class Residual(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.mod = nn.Sequential(
            CBL(c_in, c_in // 2, 1, 1),
            CBL(c_in // 2, c_in, 3, 1),
        )
    def forward(self, x):
        return self.mod(x) + x

class Darknet53(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256, 512, 1024), block_nums=(1, 2, 8, 8, 4)):
        super().__init__()
        self.input_layer = CBL(3, 32, 3, 1)
        layers = []
        for idx, block_num in enumerate(block_nums):
            cin, cout = channels[idx], channels[idx + 1]
            layer = self._make_layer(cin, cout, block_num)
            layers.append(layer)
        self.stages = nn.Sequential(*layers)
    def _make_layer(self, c_in, c_out, block_nums):
        layers = [CBL(c_in, c_out, 3, 2)]
        for _ in range(block_nums):
            layers.append(Residual(c_out))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.input_layer(x)
        out_52 = self.stages[:3](x)
        out_26 = self.stages[3](out_52)
        out_13 = self.stages[4](out_26)
        return out_52, out_26, out_13

class CBLSET(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.sub_mod = nn.Sequential(
            CBL(c_in, c_out, 1, 1),
            CBL(c_out, c_in, 3, 1),
            CBL(c_in, c_out, 1, 1),
            CBL(c_out, c_in, 3, 1),
            CBL(c_in, c_out, 1, 1),
        )
    def forward(self, x):
        return self.sub_mod(x)

class YOLOv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.bone = Darknet53()
        self.cov_set_13 = CBLSET(1024, 512)
        self.out_neck_13 = nn.Sequential(
            CBL(512, 1024, 3, 1),
            nn.Conv2d(1024, 3 * (5 + NUM_CLASSES), 1, 1),
        )
        self.up_13 = nn.Sequential(
            CBL(512, 256, 1, 1),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.cov_set_26 = CBLSET(512 + 256, 256)
        self.out_neck_26 = nn.Sequential(
            CBL(256, 512, 3, 1),
            nn.Conv2d(512, 3 * (5 + NUM_CLASSES), 1, 1),
        )
        self.up_26_52 = nn.Sequential(
            CBL(256, 128, 1, 1),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.neck_52 = nn.Sequential(
            CBLSET(128 + 256, 128),
            CBL(128, 256, 3, 1),
            nn.Conv2d(256, 3 * (5 + NUM_CLASSES), 1, 1),
        )
    def forward(self, x):
        out_52, out_26, out_13 = self.bone(x)
        cov_set_13_out = self.cov_set_13(out_13)
        out_neck_13 = self.out_neck_13(cov_set_13_out)
        up_13_out_26 = self.up_13(cov_set_13_out)
        concat_26 = torch.cat((up_13_out_26, out_26), dim=1)
        cov_set_26_out = self.cov_set_26(concat_26)
        out_neck_26 = self.out_neck_26(cov_set_26_out)
        up_26_out_52 = self.up_26_52(cov_set_26_out)
        concat_52 = torch.cat((up_26_out_52, out_52), dim=1)
        out_neck_52 = self.neck_52(concat_52)
        return out_neck_13, out_neck_26, out_neck_52


# -------------------- 预处理 --------------------
def preprocess(frame, img_size=416):
    """OpenCV 直接预处理：resize + padding → [1,3,H,W] tensor"""
    h, w = frame.shape[:2]
    scale = min(img_size / w, img_size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))

    padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    dx, dy = (img_size - nw) // 2, (img_size - nh) // 2
    padded[dy:dy + nh, dx:dx + nw] = resized

    # BGR→RGB, HWC→CHW, /255
    tensor = torch.from_numpy(padded[..., ::-1].copy()).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(DEVICE), scale, dx, dy


# -------------------- 解码（向量化，参考GPT优化） --------------------
def decode_output(pred, feat_size):
    """
    pred: [1, 3*(5+NC), H, W]  模型原始输出
    返回: (boxes, scores)  — CPU tensor
    """
    B, C, H, W = pred.shape
    stride = IMG_SIZE / feat_size
    anchors = torch.tensor(ANCHORS_GROUPS[feat_size], device=DEVICE, dtype=torch.float32)

    # [B, 3, 5+NC, H, W] → [B, 3, H, W, 5+NC] → [B, 3*H*W, 5+NC]
    pred = pred.view(B, 3, 5 + NUM_CLASSES, H, W)
    pred = pred.permute(0, 1, 3, 4, 2).contiguous()
    pred = pred.view(B, -1, 5 + NUM_CLASSES)

    # 对象置信度
    obj_conf = torch.sigmoid(pred[0, :, 4])
    mask = obj_conf > CONF_THRESHOLD
    if not mask.any():
        return torch.empty((0, 4), device="cpu"), torch.empty((0,), device="cpu"), torch.empty((0,), dtype=torch.long, device="cpu")

    candidates = pred[0, mask]
    cand_conf = obj_conf[mask]

    # 分类：用 sigmoid（YOLOv3 多标签），obj_conf 直接做置信度不乘 cls_max
    cls_scores = torch.sigmoid(candidates[:, 5:])
    cls_max, cls_ids = cls_scores.max(dim=1)

    final_conf = cand_conf  # 用 obj_conf 直接作为最终置信度
    valid_indices = torch.where(mask)[0]

    grid_total = H * W
    anchor_idx = valid_indices // grid_total   # 哪个 anchor（0/1/2）
    grid_idx = valid_indices % grid_total      # 在 grid 中的位置
    grid_x = (grid_idx % W).float()
    grid_y = (grid_idx // W).float()

    tx, ty, tw, th = candidates[:, 0], candidates[:, 1], candidates[:, 2], candidates[:, 3]
    aw = anchors[anchor_idx, 0]
    ah = anchors[anchor_idx, 1]

    cx = (torch.sigmoid(tx) + grid_x) * stride
    cy = (torch.sigmoid(ty) + grid_y) * stride
    bw = torch.exp(tw) * aw
    bh = torch.exp(th) * ah

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    return boxes.cpu(), final_conf.cpu(), cls_ids.cpu()


# -------------------- 缩放到原图 --------------------
def rescale_boxes(boxes, scale, dx, dy, orig_w, orig_h):
    boxes = boxes.clone()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)
    return boxes


# -------------------- 画框 --------------------
def draw_boxes(frame, boxes, scores, cls_ids):
    for box, score, cls_id in zip(boxes, scores, cls_ids):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls_id = int(cls_id)
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))
        label = f"{CLASSES[cls_id]} {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw + 5, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# -------------------- 主流程 --------------------
def detect_video(video_path, output_path, weight_path):
    if not os.path.exists(weight_path):
        print(f"❌ 权重文件不存在: {weight_path}")
        print("请先运行 train_yolo_xml.py 训练模型")
        return

    print(f"加载模型: {weight_path}")
    model = YOLOv3().to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print("模型加载成功！")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息: {orig_w}x{orig_h}, {fps}fps, {total_frames} 帧")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  处理帧 {frame_idx}/{total_frames}")

        tensor, scale, dx, dy = preprocess(frame, IMG_SIZE)
        with torch.no_grad():
            o13, o26, o52 = model(tensor)

        all_boxes, all_scores, all_cls = [], [], []
        for pred, fs in [(o13, 13), (o26, 26), (o52, 52)]:
            boxes, scores, cls_ids = decode_output(pred, fs)
            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_cls.append(cls_ids)

        if all_boxes:
            boxes_cat = torch.cat(all_boxes, dim=0)
            scores_cat = torch.cat(all_scores, dim=0)
            cls_cat = torch.cat(all_cls, dim=0)

            # ★ torchvision.ops.nms（内置优化）
            keep = nms(boxes_cat, scores_cat, NMS_THRESHOLD)
            boxes_nms = boxes_cat[keep]
            scores_nms = scores_cat[keep]
            cls_nms = cls_cat[keep]

            boxes_nms = rescale_boxes(boxes_nms, scale, dx, dy, orig_w, orig_h)
            frame = draw_boxes(frame, boxes_nms, scores_nms, cls_nms)

        out.write(frame)

    cap.release()
    out.release()
    print(f"\n✓ 检测完成！输出视频: {output_path}")
    print(f"  共处理 {frame_idx} 帧")


if __name__ == "__main__":
    model_path = os.path.join("checkpoints", "yolov3_best.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join("yolov3_bug", "best_loss_60000.pt")

    detect_video(VIDEO_PATH, OUTPUT_PATH, model_path)
