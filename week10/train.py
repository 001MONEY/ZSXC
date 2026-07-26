# ============================================================
# YOLOv3 完整训练 —— 使用 datas/YOLODataset_origin_labels 数据集
# 4 类：bus(0), car(1), cat(2), person(3)
# ============================================================
import torch
import torch.nn as nn
import numpy as np
import math
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# -------------------- 配置参数 --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 200
IMG_SIZE = 416
NUM_CLASSES = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
DATASET_DIR = r"datas/YOLODataset_origin_labels"

# VOC 标准 anchor（416x416 输入）
ANCHORS_GROUPS = {
    13: [[116, 90], [156, 198], [373, 326]],
    26: [[30, 61], [62, 45], [59, 119]],
    52: [[10, 13], [16, 30], [33, 23]]
}

CLASSES = ['bus', 'car', 'cat', 'person']

print(f"设备: {DEVICE}")
print(f"类别: {CLASSES}")
print(f"训练设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
# -------------------- 模型组件 --------------------
class CBL(nn.Module):
    """Conv + BN + LeakyReLU"""
    def __init__(self, c_in, c_out, k, s):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, padding=k//2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.mod(x)

class Residual(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.mod = nn.Sequential(
            CBL(c_in, c_in//2, 1, 1),
            CBL(c_in//2, c_in, 3, 1),
        )
    def forward(self, x):
        return self.mod(x) + x

class Darknet53(nn.Module):
    """Darknet-53 骨干网络"""
    def __init__(self, channels=[32,64,128,256,512,1024], block_nums=[1,2,8,8,4]):
        super().__init__()
        self.input_layer = CBL(3, 32, 3, 1)
        layers = []
        for idx, block_num in enumerate(block_nums):
            cin, cout = channels[idx], channels[idx+1]
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
        out_52 = self.stages[:3](x)    # 52x52
        out_26 = self.stages[3](out_52)  # 26x26
        out_13 = self.stages[4](out_26)  # 13x13
        return out_52, out_26, out_13

class CBLSET(nn.Module):
    """ConvSet: 1x1, 3x3, 1x1, 3x3, 1x1"""
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
    """YOLOv3 完整网络（Darknet53 + FPN + 预测头）"""
    def __init__(self):
        super().__init__()
        self.bone = Darknet53()
        self.cov_set_13 = CBLSET(1024, 512)
        self.out_neck_13 = nn.Sequential(
            CBL(512, 1024, 3, 1),
            nn.Conv2d(1024, 3*(5+NUM_CLASSES), 1, 1)
        )
        self.up_13 = nn.Sequential(
            CBL(512, 256, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.cov_set_26 = CBLSET(512+256, 256)
        self.out_neck_26 = nn.Sequential(
            CBL(256, 512, 3, 1),
            nn.Conv2d(512, 3*(5+NUM_CLASSES), 1, 1)
        )
        self.up_26_52 = nn.Sequential(
            CBL(256, 128, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.neck_52 = nn.Sequential(
            CBLSET(128+256, 128),
            CBL(128, 256, 3, 1),
            nn.Conv2d(256, 3*(5+NUM_CLASSES), 1, 1)
        )

    def forward(self, x):
        out_52, out_26, out_13 = self.bone(x)
        # 13x13 尺度
        cov_set_13_out = self.cov_set_13(out_13)
        out_neck_13 = self.out_neck_13(cov_set_13_out)
        # 13→26 上采样
        up_13_out_26 = self.up_13(cov_set_13_out)
        concat_26 = torch.cat((up_13_out_26, out_26), dim=1)
        cov_set_26_out = self.cov_set_26(concat_26)
        out_neck_26 = self.out_neck_26(cov_set_26_out)
        # 26→52 上采样
        up_26_out_52 = self.up_26_52(cov_set_26_out)
        concat_52 = torch.cat((up_26_out_52, out_52), dim=1)
        out_neck_52 = self.neck_52(concat_52)
        return out_neck_13, out_neck_26, out_neck_52
    # -------------------- 数据集 --------------------
class YOLODataset(Dataset):
    """读取 datas/YOLODataset_origin_labels/train.txt 格式的数据集
    
    格式: images/train/xxx.jpg cls cx cy w h [cls cx cy w h ...]
    其中 cx, cy, w, h 均为像素坐标
    """
    def __init__(self, dataset_dir, split='train', transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform or transforms.ToTensor()
        txt_path = os.path.join(dataset_dir, f'{split}.txt')
        self.annotations = open(txt_path, 'r', encoding='utf-8').readlines()
        self.img_dir = os.path.join(dataset_dir, 'images', split)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        line = self.annotations[index].strip()
        parts = line.split()
        img_rel_path = parts[0]  # e.g. images/train/000001.jpg
        img_path = os.path.join(self.dataset_dir, img_rel_path)

        # 读取图像
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        # letterbox resize 到 416x416
        img_resized, ratio, (dw, dh) = self._letterbox(img)
        img_tensor = self.transform(img_resized)  # 3,416,416

        # 解析标签
        labels_data = [float(x) for x in parts[1:]]
        labels = np.array(labels_data).reshape(-1, 5)  # N,5

        # 初始化三个尺度的标签
        label_tensors = {}
        for feat_size, anchors in ANCHORS_GROUPS.items():
            label_tensors[feat_size] = torch.zeros((feat_size, feat_size, 3, 5+NUM_CLASSES))

        for label in labels:
            cls_id = int(label[0])
            cx, cy, bw, bh = label[1:5]

            # 转换坐标到 416x416 空间（中心点坐标也要对应缩放）
            cx_new = (cx * ratio + dw)
            cy_new = (cy * ratio + dh)
            bw_new = bw * ratio
            bh_new = bh * ratio

            for feat_size, anchors in ANCHORS_GROUPS.items():
                stride = IMG_SIZE / feat_size
                # 计算中心点所在的 grid cell
                cx_idx = int(cx_new // stride)
                cy_idx = int(cy_new // stride)
                # grid cell 内的偏移
                offset_cx = (cx_new / stride) - cx_idx
                offset_cy = (cy_new / stride) - cy_idx

                # 为每个 anchor 计算 IoU，选最佳 anchor
                best_iou = 0
                best_anchor_idx = 0
                for aidx, (aw, ah) in enumerate(anchors):
                    # 计算该 anchor 与 gt 的 IoU
                    inter_w = min(bw_new, aw)
                    inter_h = min(bh_new, ah)
                    inter_area = inter_w * inter_h
                    union_area = bw_new * bh_new + aw * ah - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_anchor_idx = aidx

                offset_w = math.log(bw_new / anchors[best_anchor_idx][0] + 1e-16)
                offset_h = math.log(bh_new / anchors[best_anchor_idx][1] + 1e-16)

                # one-hot 类别
                cls_onehot = torch.zeros(NUM_CLASSES)
                cls_onehot[cls_id] = 1.0

                # 编码: [conf=1, offset_cx, offset_cy, offset_w, offset_h, onehot_cls]
                if 0 <= cx_idx < feat_size and 0 <= cy_idx < feat_size:
                    label_tensors[feat_size][cy_idx, cx_idx, best_anchor_idx] = torch.tensor(
                        [1.0, offset_cx, offset_cy, offset_w, offset_h, *cls_onehot],
                        dtype=torch.float32
                    )

        return (label_tensors[13], label_tensors[26], label_tensors[52], img_tensor)

    def _letterbox(self, img, color=(114, 114, 114)):
        """将图像 resize 为 416x416 正方形（保持比例，填充灰边）"""
        w, h = img.size
        ratio = min(IMG_SIZE / w, IMG_SIZE / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        dw = (IMG_SIZE - new_w) / 2
        dh = (IMG_SIZE - new_h) / 2
        result = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color)
        result.paste(img, (int(dw), int(dh)))
        return result, ratio, (dw, dh)
    # -------------------- 损失函数 --------------------
class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, predictions, targets, feat_size):
        """
        predictions: [B, 3*(5+C), H, W]  模型原始输出
        targets:    [B, H, W, 3, 5+C]    标签
        """
        B = predictions.size(0)
        # reshape: [B, 3*(5+C), H, W] -> [B, H, W, 3, 5+C]
        pred = predictions.permute(0, 2, 3, 1)
        pred = pred.reshape(B, feat_size, feat_size, 3, 5+NUM_CLASSES)

        # 正样本掩码 (conf > 0)
        pos_mask = targets[..., 0] > 0  # [B, H, W, 3]
        neg_mask = targets[..., 0] == 0

        # ---- 正样本损失 ----
        if pos_mask.any():
            pred_pos = pred[pos_mask]
            target_pos = targets[pos_mask]

            # 置信度损失 (BCE)
            loss_conf = self.bce(pred_pos[:, 0], target_pos[:, 0])

            # 坐标损失 (MSE)
            loss_xy = self.mse(pred_pos[:, 1:3], target_pos[:, 1:3])
            loss_wh = self.mse(pred_pos[:, 3:5], target_pos[:, 3:5])
            loss_box = loss_xy + loss_wh

            # 分类损失 (CE)
            loss_cls = self.ce(pred_pos[:, 5:], torch.argmax(target_pos[:, 5:], dim=1))

            pos_loss = loss_conf + loss_box + loss_cls
        else:
            pos_loss = torch.tensor(0.0, device=predictions.device)

        # ---- 负样本损失（仅置信度） ----
        if neg_mask.any():
            pred_neg = pred[neg_mask]
            target_neg = targets[neg_mask]
            neg_loss = self.bce(pred_neg[:, 0], target_neg[:, 0])
        else:
            neg_loss = torch.tensor(0.0, device=predictions.device)

        # 平衡因子：正样本加权，负样本降权
        total_loss = 5.0 * pos_loss + 0.25 * neg_loss
        return total_loss / B
    # -------------------- 训练循环 --------------------
def train():
    # 创建数据集和数据加载器
    train_dataset = YOLODataset(DATASET_DIR, split='train')
    val_dataset = YOLODataset(DATASET_DIR, split='val')

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )

    print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")
    print(f"批次大小: {BATCH_SIZE}, 训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")

    # 初始化模型
    model = YOLOv3().to(DEVICE)
    criterion = YOLOLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 150], gamma=0.1)

    # 创建保存目录
    os.makedirs('checkpoints', exist_ok=True)

    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS} [Train]')
        for batch_idx, (label_13, label_26, label_52, img_tensor) in enumerate(pbar):
            label_13 = label_13.to(DEVICE)
            label_26 = label_26.to(DEVICE)
            label_52 = label_52.to(DEVICE)
            img_tensor = img_tensor.to(DEVICE)

            out_13, out_26, out_52 = model(img_tensor)

            loss_13 = criterion(out_13, label_13, 13)
            loss_26 = criterion(out_26, label_26, 26)
            loss_52 = criterion(out_52, label_52, 52)
            loss = loss_13 + loss_26 + loss_52

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f'Epoch {epoch}/{EPOCHS} [Val]')
            for label_13, label_26, label_52, img_tensor in pbar_val:
                label_13 = label_13.to(DEVICE)
                label_26 = label_26.to(DEVICE)
                label_52 = label_52.to(DEVICE)
                img_tensor = img_tensor.to(DEVICE)

                out_13, out_26, out_52 = model(img_tensor)
                loss_13 = criterion(out_13, label_13, 13)
                loss_26 = criterion(out_26, label_26, 26)
                loss_52 = criterion(out_52, label_52, 52)
                batch_loss = (loss_13 + loss_26 + loss_52).item()
                val_loss += batch_loss
                pbar_val.set_postfix(loss=f'{batch_loss:.4f}')

        avg_val_loss = val_loss / len(val_loader)
        print(f'--- Epoch [{epoch}/{EPOCHS}] 完成 --- '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'checkpoints/yolov3_best.pth')
            print(f'  ✓ 保存最佳模型 (loss={best_loss:.4f})')

        # 每 50 轮保存检查点
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, f'checkpoints/yolov3_epoch_{epoch}.pth')
            print(f'  ✓ 保存检查点 epoch_{epoch}')

    print(f'训练完成！最佳验证损失: {best_loss:.4f}')
    return model

# 开始训练
if __name__ == '__main__':
    model = train()