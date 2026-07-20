import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
import os
from torch.utils.data import Dataset
import math

# ╔══════════════════════════════════════════════════════════════════╗
# ║   YOLOv3 从零实现 — 伪代码框架                                   ║
# ║   每个函数/类都给了详细说明和实现提示，你照着填空即可              ║
# ║   把 # TODO 替换为你的代码                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

# ====================================================================
#  1. 基础组件 —— 卷积块、残差单元
# ====================================================================

class ConvBlock(nn.Module):
    """
    Darknet风格的卷积块: Conv2D + BatchNorm + LeakyReLU
    - bias=False (因为后面有BN)
    - LeakyReLU 的 negative_slope = 0.1 (Darknet 默认)
    - padding 要保证卷积前后空间尺寸不变 (kernel=1→pad=0, kernel=3→pad=1)
    """
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding,bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.lrelu = nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))


class ConvBlockNoBN(nn.Module):
    """
    不带 BN 和激活的卷积层
    用于最后的预测层 (YOLO 的输出层不需要 BN)
    bias=True 因为没 BN
    """
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding,bias=True)

    def forward(self, x):
        return self.conv(x)
    


class ResidualUnit(nn.Module):
    """
    Darknet 残差单元: 1×1 降维 → 3×3 升维 → + 恒等映射
    通道变化: ch → ch//2 → ch
    forward: x + block(x)
    """
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(ConvBlock(ch, ch//2, 1),
                                    ConvBlock(ch//2, ch, 3))

    def forward(self, x):
        return x +self.block(x)


class ResidualBlock(nn.Module):
    """多个 ResidualUnit 堆叠"""
    def __init__(self, ch, num_units):
        super().__init__()
        self.units = nn.Sequential(*[ResidualUnit(ch) for _ in range(num_units)])

    def forward(self, x):
        return self.units(x)
        


# ====================================================================
#  2. Darknet-53 骨干网络
# ====================================================================
# 结构 (输入 416×416×3):
#   Stage 0: Conv 3×3/1 → 32                          416×416×32
#   Stage 1: Conv 3×3/2 → 64  + Residual ×1           208×208×64
#   Stage 2: Conv 3×3/2 → 128 + Residual ×2           104×104×128
#   Stage 3: Conv 3×3/2 → 256 + Residual ×8           52×52×256   ← route_52
#   Stage 4: Conv 3×3/2 → 512 + Residual ×8           26×26×512   ← route_26
#   Stage 5: Conv 3×3/2 → 1024 + Residual ×4          13×13×1024  ← route_13

class Darknet53(nn.Module):
    """
    Darknet-53 骨干网络
    返回 3 个尺度的特征图用于 FPN:
        route_52 (52×52×256)  — 大目标检测
        route_26 (26×26×512)  — 中目标检测
        route_13 (13×13×1024) — 小目标检测
    """
    def __init__(self):
        super().__init__()
        self.stage0 = ConvBlock(3,32,3,1)
        
        self.stage1 = nn.Sequential(ConvBlock(32,64,3,2),
                                    ResidualBlock(64,1),)
        
        self.stage2 = nn.Sequential(ConvBlock(64,128,3,2),
                                    ResidualBlock(128,2),)
        
        self.stage3 = nn.Sequential(ConvBlock(128,256,3,2),
                                    ResidualBlock(256,8),)
        
        self.stage4 = nn.Sequential(ConvBlock(256,512,3,2),
                                    ResidualBlock(512,8),)
        
        self.stage5 = nn.Sequential(ConvBlock(512,1024,3,2),
                                    ResidualBlock(1024,4),)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        route_52 = x
        x = self.stage4(x)
        route_26 = x
        x = self.stage5(x)
        route_13 = x
        return route_52, route_26, route_13
    


# ====================================================================
#  3. YOLOv3 检测头 (每个尺度一个)
# ====================================================================

class YOLOHead(nn.Module):
    """
    单个尺度的 YOLOv3 检测头
    输入特征图 → 5 个 ConvBlock (1×1 和 3×3 交替) → 1 个预测 Conv (无 BN)

    输出通道: num_anchors × (5 + num_classes)
    对 COCO 80 类: 3 × 85 = 255
    """
    def __init__(self, in_ch, num_classes=80, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        out_ch = num_anchors * (5 + num_classes)

        self.conv_block = nn.Sequential(
            ConvBlock(in_ch, in_ch // 2, 1, 1),
            ConvBlock(in_ch // 2, in_ch, 3, 1),
            ConvBlock(in_ch, in_ch // 2, 1, 1),
            ConvBlock(in_ch // 2, in_ch, 3, 1),
            ConvBlock(in_ch, in_ch // 2, 1, 1),
        )
        
        self.pred = ConvBlockNoBN(in_ch//2, out_ch, 1)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.pred(x)
        return x


# ====================================================================
#  4. YOLOv3 完整网络 (Backbone + FPN + 3个检测头)
# ====================================================================

class YOLOv3(nn.Module):
    def __init__(self, num_classes=80, anchors_cfg='coco'):
        super().__init__()
        self.num_classes = num_classes

        if anchors_cfg == 'coco':
            self.anchors = torch.tensor([
                [10,13], [16,30], [33,23],
                [30,61], [62,45], [59,119],
                [116,90], [156,198], [373,326],
            ], dtype=torch.float32)
        else:
            self.anchors = anchors_cfg

        self.backbone = Darknet53()

        self.head_13 = YOLOHead(1024, num_classes, 3)
        self.head_26 = YOLOHead(512 ,num_classes, 3)
        self.head_52 = YOLOHead(256, num_classes, 3)
        
        self.conv13_to_26 = nn.Sequential(
            ConvBlock(1024, 512, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.conv26_reduce = ConvBlock(1024,512,1,1)
        self.conv26_to_52 = nn.Sequential(
            ConvBlock(512, 256, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.conv52_reduce = ConvBlock(512,256,1,1)
        
        
    def forward(self, x):
    
        route_52, route_26, route_13 = self.backbone(x)
        out_13 = self.head_13(route_13)
        x_up_13 = self.conv13_to_26(route_13)
        
        x_cat_26 = torch.cat([x_up_13,route_26],dim=1)
        x_cat_26 = self.conv26_reduce(x_cat_26)
        out_26 = self.head_26(x_cat_26)
        x_up_26 = self.conv26_to_52(x_cat_26)
        
        x_cat_52 = torch.cat([x_up_26,route_52],dim=1)
        x_cat_52 = self.conv52_reduce(x_cat_52)
        out_52 = self.head_52(x_cat_52)
        
        return out_13, out_26, out_52
       

    @property
    def device(self):
        return next(self.parameters()).device


# ====================================================================
#  5. 解码: 网络原始输出 → 边界框坐标
# ====================================================================
# YOLOv3 解码公式:
#   bx = σ(tx) + cx        (cx, cy) = 网格单元格左上角坐标
#   by = σ(ty) + cy
#   bw = pw · exp(tw)      (pw, ph) = anchor 宽高
#   bh = ph · exp(th)
#   obj_conf = σ(obj)      目标置信度
#   class_prob = softmax(cls_logits)  类别概率
#   最终置信度 = obj_conf × max(class_prob)

def decode_scale(output, anchors, num_classes, stride, img_size=416):
    """
    解码单个尺度的 YOLO 输出

    参数:
        output: [B, 3×(5+C), H, W] — 网络原始输出
        anchors: [3, 2] — 该尺度的 3 个 anchor (w, h)
        num_classes: int
        stride: int — 下采样倍数 (8/16/32)
        img_size: int — 输入图像尺寸 (默认 416)

    返回:
        boxes: [B, H×W×3, 6] — (x1, y1, x2, y2, conf, class_id)
    """
    B, _, H, W = output.shape
    num_anchors = len(anchors)

    # 1) 重塑: [B, 3, 5+C, H, W] → permute → [B, 3, H, W, 5+C]
    # 2) 分离出 tx, ty, tw, th, obj_conf, cls_logits (用切片)
    # 3) 创建网格坐标 grid_x [1,1,1,W,1], grid_y [1,1,H,1,1]
    # 4) 解码:
    #    bx = sigmoid(tx) + grid_x
    #    by = sigmoid(ty) + grid_y
    #    bw = anchor_w * exp(tw)
    #    bh = anchor_h * exp(th)
    # 5) 解码置信度: conf = sigmoid(obj_conf)
    # 6) 解码类别: cls_probs = softmax(cls_logits), cls_conf, cls_id = max
    # 7) 最终置信度 = conf × cls_conf
    # 8) 坐标缩放: bx *= stride, 同理 by, bw, bh
    # 9) 转 (x1,y1,x2,y2): x1=bx-bw/2, 等
    # 10) 拼接: cat([x1,y1,x2,y2,final_conf,cls_id], dim=-1)
    # 11) view(B, -1, 6) 展平所有预测框
    # 12) clamp 到 [0, img_size]
    output = output.view(B, num_anchors, 5 + num_classes, H, W)
    output = output.permute(0, 1, 3, 4, 2).contiguous()
    
    tx = output[..., 0:1]
    ty = output[..., 1:2]
    tw = output[..., 2:3]
    th = output[..., 3:4]
    obj_conf = output[..., 4:5]
    cls_logits = output[..., 5:]
    
    grid_x = torch.arange(W, device=output.device).float().view(1, 1, 1, W, 1)
    grid_y = torch.arange(H, device=output.device).float().view(1, 1, H, 1, 1)


def decode_all_scales(out_13, out_26, out_52, anchors, num_classes, img_size=416):
    """解码所有 3 个尺度的输出并合并为一个大张量"""
    # 每个尺度对应的 stride 和 anchor 分组
    strides = [32, 16, 8]
    scale_anchors = [
        anchors[6:9],   # stride=32, 13×13
        anchors[3:6],   # stride=16, 26×26
        anchors[0:3],   # stride=8,  52×52
    ]
    outputs = [out_13, out_26, out_52]

    # TODO: 循环调用 decode_scale, 收集结果
    # TODO: 在 dim=1 上拼接所有框
    # 返回 [B, total_boxes, 6]
    pass


# ====================================================================
#  6. NMS (Non-Maximum Suppression)
# ====================================================================

def nms(boxes, conf_thresh=0.5, iou_thresh=0.45):
    """
    非极大值抑制

    参数:
        boxes: [N, 6] — (x1,y1,x2,y2,conf,class_id)
        conf_thresh: 置信度阈值
        iou_thresh:  IoU 阈值

    返回:
        keep: list[int] — 保留框的索引

    算法:
        1. 过滤 conf < conf_thresh 的框
        2. 按置信度降序排列
        3. 循环: 取最高分框 → 计算与其余框的 IoU → 删除 IoU > iou_thresh 且同类的框
    """
    # TODO: 实现 NMS
    pass


def post_process(all_boxes, conf_thresh=0.5, iou_thresh=0.45):
    """
    对批量预测进行后处理: 过滤 + NMS

    参数:
        all_boxes: [B, N, 6]

    返回:
        results: list of [M, 6] — 每张图的检测结果
    """
    # TODO: 对 batch 中每张图分别调用 nms()
    # TODO: 收集结果
    pass


# ====================================================================
#  7. 加载 Darknet 预训练权重
# ====================================================================
# yolov3.weights 二进制格式:
#   [int32×5: header] → [float32...: 权重数据]
# 每层顺序 (有 BN): bn_bias, bn_weight, bn_mean, bn_var, conv_weight
# 每层顺序 (无 BN): conv_bias, conv_weight

def load_darknet_weights(model, weights_path):
    """
    从 Darknet 的 .weights 文件加载预训练权重

    实现思路:
    1. 读取文件: header (5×int32) + 权重 (float32 数组)
    2. 遍历模型的所有 Conv2d 层, 区分"有 BN 的"和"无 BN 的"
       有 BN 的: 读 4 组 BN 参数 + conv_weight
       无 BN 的: 读 bias + conv_weight
    3. 按顺序将权重填入对应层
    """
    # TODO: 实现权重加载
    pass


# ====================================================================
#  8. 可视化
# ====================================================================

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_colors(num_colors):
    """生成 num_colors 个区分度高的颜色 (HSV 均匀采样)"""
    # TODO: HSV → RGB 转换, 返回 [(r,g,b), ...]
    pass


def draw_boxes(image, boxes, class_names=None, conf_thresh=0.5):
    """
    在图像上绘制检测框和标签

    参数:
        image: numpy array [H, W, 3] (BGR, 0-255)
        boxes: [N, 6] — (x1,y1,x2,y2,conf,class_id)
        class_names: list[str]
        conf_thresh: float

    返回:
        img_with_boxes: numpy array
    """
    # TODO: 遍历 boxes, 对 conf ≥ conf_thresh 的在图上画框和标签
    # 用 cv2.rectangle 画框, cv2.putText 写标签
    pass


def detect_image(model, image_path, conf_thresh=0.5, iou_thresh=0.45, img_size=416):
    """
    对单张图片执行 YOLOv3 检测

    流程: 读图 → 预处理 (resize + normalize) → 前向 → 解码 → NMS → 坐标缩放 → 绘图

    返回:
        (img_with_boxes, detections)
        img_with_boxes: numpy array 绘制了检测框的图像
        detections: [N, 6] 检测结果
    """
    # TODO: 实现检测流程
    pass


# ====================================================================
#  9. YOLO Dataset — 数据加载与标签编码
# ====================================================================
# 作业要求: 手搓完整的 Dataset 类
#
# 功能:
#   1. __init__: 读取 train.txt (每行: 图片路径  cls cx cy w h ...)
#   2. __getitem__:
#      a) 读图 → 转 Tensor [3, H, W] 归一化到 [0,1]
#      b) 解析标注框 (flat 列表 → 每5个一组)
#      c) 编码到三个尺度的 label 张量:
#          label[13]: [13,13,3,5+CLASS_NUM]
#          label[26]: [26,26,3,5+CLASS_NUM]
#          label[52]: [52,52,3,5+CLASS_NUM]
#      编码步骤 (每个框):
#        ① 计算中心点落在哪个网格 (cx_idx = floor(cx / stride))
#        ② 计算格内偏移 (offset_cx = cx/stride - cx_idx)
#        ③ 遍历 3 个 anchor, 算 offset_w = log(gt_w / anchor_w)
#        ④ 填入 label[feature_size][cy_idx, cx_idx, anchor_idx]
#           = [conf=1, offset_cx, offset_cy, offset_w, offset_h, *onehot]
#   3. 返回 (label_13, label_26, label_52, img_tensor)

class YOLODataset(Dataset):
    """完整的 YOLO 数据集类 (手搓版)"""
    def __init__(self, root_path, img_size=416, class_num=80):
        # TODO: 读取 root_path (train.txt), 保存所有行到 self.dataset
        # TODO: 保存 img_size, class_num
        pass

    def __len__(self):
        # TODO: 返回样本数量
        pass

    def __getitem__(self, index):
        # 1) 解析一行: line.split() → img_path + flat labels
        # 2) 读图: Image.open → ToTensor
        # 3) labels = np.array(line[1:]).reshape(-1, 5)
        #    每行: [cls, cx, cy, w, h]
        # 4) 初始化 label dict:
        #    label[13] = zeros(13,13,3, 5+CLASS_NUM)
        #    label[26] = zeros(26,26,3, 5+CLASS_NUM)
        #    label[52] = zeros(52,52,3, 5+CLASS_NUM)
        # 5) 遍历每个 bbox:
        #    a) cx_idx = int(cx / (IMG_SIZE/13)), offset_cx = cx/stride - cx_idx
        #    b) 遍历 3 个 anchor (用 cfg.ANCHORS_GROUPS[feat_size]):
        #       - offset_w = log(gt_w / anchor_w)
        #       - onehot = one_hot(cls, CLASS_NUM)
        #       - label[feat][cy_idx, cx_idx, anchor_idx] = [1, dx, dy, dw, dh, *onehot]
        # 6) 返回 (label[13], label[26], label[52], img_tensor)
        # TODO: 实现以上步骤
        pass


# ====================================================================
#  10. 测试函数 (写完后运行验证)
# ====================================================================

def test_forward():
    """
    测试 YOLOv3 前向传播
    预期输出:
        out_13: [1, 255, 13, 13]
        out_26: [1, 255, 26, 26]
        out_52: [1, 255, 52, 52]
        解码后总框数: 10647 (= 13²×3 + 26²×3 + 52²×3)
    """
    model = YOLOv3(num_classes=80)
    model.eval()

    # 统计参数量
    total = sum(p.numel() for p in model.parameters())
    print(f'总参数量: {total:,}')

    x = torch.randn(1, 3, 416, 416)
    with torch.no_grad():
        out_13, out_26, out_52 = model(x)

    print(f'out_13: {out_13.shape}')
    print(f'out_26: {out_26.shape}')
    print(f'out_52: {out_52.shape}')
    assert out_13.shape == (1, 255, 13, 13), f'out_13 shape 错误: {out_13.shape}'
    assert out_26.shape == (1, 255, 26, 26), f'out_26 shape 错误: {out_26.shape}'
    assert out_52.shape == (1, 255, 52, 52), f'out_52 shape 错误: {out_52.shape}'

    print('✅ 前向传播测试通过!')
    return model


def test_weight_loading():
    """测试加载 yolov3.weights (如果存在)"""
    if not os.path.exists('yolov3.weights'):
        print('⚠️ 未找到 yolov3.weights, 跳过')
        return
    model = YOLOv3(num_classes=80)
    load_darknet_weights(model, 'yolov3.weights')
    print('✅ 权重加载完成')


if __name__ == '__main__':
    test_forward()