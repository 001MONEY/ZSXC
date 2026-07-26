# ============================================================
# YOLOv3 完整训练 (P100-16GB 优化版)
# 4 类：bus(0), car(1), cat(2), person(3)
# 修复：Warmup / Mosaic增强 / 标签平滑 / SGD / Anchor去重 / 显存优化
# ============================================================
import torch
import torch.nn as nn
import numpy as np
import math
import os
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# -------------------- 随机种子 --------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        # P100 输入尺寸固定为416，开启benchmark可加速卷积算子选择
        torch.backends.cudnn.benchmark = True
set_seed(42)

# -------------------- 硬编码配置 (P100-16GB 专属) --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16           # ↑ P100-16GB 跑 YOLOv3-416 推荐BS=16~32
EPOCHS = 200
IMG_SIZE = 416
NUM_CLASSES = 4
LEARNING_RATE = 0.001     # Adam 标准初始LR
WEIGHT_DECAY = 0.0005
NUM_WORKERS = 4           # ↓ 避免共享内存(/dev/shm)溢出导致卡死
WARMUP_EPOCHS = 3         # ↑ 线性预热防止初期发散
LABEL_SMOOTHING = 0.01    # ↑ 标签平滑防止过拟合
DATASET_DIR = r"YOLODataset_origin_labels"
RESUME = None             # 设为 checkpoint 路径可恢复训练

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
print(f"数据集: {DATASET_DIR}")
print(f"批次大小: {BATCH_SIZE}, 线程数: {NUM_WORKERS}")
print(f"Warmup: {WARMUP_EPOCHS} epochs, 标签平滑: {LABEL_SMOOTHING}")

# -------------------- 模型组件 --------------------
class CBL(nn.Module):
    """Conv + BN + LeakyReLU"""
    def __init__(self, c_in, c_out, k, s):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, padding=k//2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.1, inplace=True)
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
        out_52 = self.stages[:3](x)
        out_26 = self.stages[3](out_52)
        out_13 = self.stages[4](out_26)
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


# -------------------- 数据集 (含 Mosaic 增强) --------------------
class YOLODataset(Dataset):
    def __init__(self, dataset_dir, split='train', transform=None, use_mosaic=True):
        self.dataset_dir = dataset_dir
        self.split = split
        self.use_mosaic = use_mosaic and split == 'train'
        self.transform = transform or transforms.ToTensor()
        txt_path = os.path.join(dataset_dir, f'{split}.txt')

        raw_lines = open(txt_path, 'r', encoding='utf-8').readlines()
        self.annotations = []
        self._skipped = 0
        for line in raw_lines:
            parts = line.strip().split()
            if not parts: continue
            img_path = os.path.join(dataset_dir, parts[0])
            if os.path.exists(img_path):
                self.annotations.append(line.strip())
            else:
                self._skipped += 1
        if self._skipped > 0:
            print(f'  ⚠ {split} 集: 跳过 {self._skipped} 条缺失图片的标注')
        if len(self.annotations) == 0:
            raise RuntimeError(f'{split} 集没有可用图片: {dataset_dir}')

    def __len__(self):
        return len(self.annotations)

    def _load_image_and_labels(self, index):
        line = self.annotations[index]
        parts = line.split()
        img_path = os.path.join(self.dataset_dir, parts[0])
        img = Image.open(img_path).convert('RGB')
        labels = np.array([float(x) for x in parts[1:]]).reshape(-1, 5)
        return img, labels

    def _letterbox(self, img, color=(114, 114, 114)):
        w, h = img.size
        ratio = min(IMG_SIZE / w, IMG_SIZE / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        dw = (IMG_SIZE - new_w) / 2
        dh = (IMG_SIZE - new_h) / 2
        result = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color)
        result.paste(img, (int(dw), int(dh)))
        return result, ratio, (dw, dh)

    def _mosaic(self, index):
        """简易 Mosaic 增强：拼接4张图"""
        indices = [index] + [random.randint(0, len(self)-1) for _ in range(3)]
        imgs, all_labels = [], []
        for idx in indices:
            img, labels = self._load_image_and_labels(idx)
            imgs.append(img)
            all_labels.append(labels)

        # 简化版：取中心裁剪拼接（保留核心多尺度语义）
        half = IMG_SIZE // 2
        canvas = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (114, 114, 114))
        positions = [(0, 0), (half, 0), (0, half), (half, half)]
        merged_labels = []

        for i, (img, labels) in enumerate(zip(imgs, all_labels)):
            resized = img.resize((half, half), Image.BICUBIC)
            canvas.paste(resized, positions[i])
            if len(labels) > 0:
                # 缩放标签到 mosaic 子区域
                scale = half / max(img.size)
                px, py = positions[i]
                new_labels = labels.copy()
                new_labels[:, 1] = labels[:, 1] * scale + px
                new_labels[:, 2] = labels[:, 2] * scale + py
                new_labels[:, 3] = labels[:, 3] * scale
                new_labels[:, 4] = labels[:, 4] * scale
                merged_labels.append(new_labels)

        final_labels = np.concatenate(merged_labels, axis=0) if merged_labels else np.zeros((0, 5))
        return canvas, final_labels

    def __getitem__(self, index):
        if self.use_mosaic and random.random() < 0.5:
            img, labels = self._mosaic(index)
            ratio, (dw, dh) = 1.0, (0.0, 0.0)
        else:
            img, labels = self._load_image_and_labels(index)
            img, ratio, (dw, dh) = self._letterbox(img)

        img_tensor = self.transform(img)

        # 构建多尺度标签
        label_tensors = {}
        for feat_size in ANCHORS_GROUPS:
            label_tensors[feat_size] = torch.zeros((feat_size, feat_size, 3, 5+NUM_CLASSES))

        for label in labels:
            cls_id = int(label[0])
            cx, cy, bw, bh = label[1:5]
            cx_new = cx * ratio + dw
            cy_new = cy * ratio + dh
            bw_new = bw * ratio
            bh_new = bh * ratio

            # ★ 修复：全局选最佳anchor，避免跨尺度重复分配
            best_iou, best_feat, best_aidx = 0, None, 0
            for feat_size, anchors in ANCHORS_GROUPS.items():
                for aidx, (aw, ah) in enumerate(anchors):
                    inter = min(bw_new, aw) * min(bh_new, ah)
                    union = bw_new * bh_new + aw * ah - inter
                    iou = inter / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou, best_feat, best_aidx = iou, feat_size, aidx

            if best_feat is None: continue
            stride = IMG_SIZE / best_feat
            cx_idx = int(cx_new // stride)
            cy_idx = int(cy_new // stride)
            if not (0 <= cx_idx < best_feat and 0 <= cy_idx < best_feat): continue

            offset_cx = (cx_new / stride) - cx_idx
            offset_cy = (cy_new / stride) - cy_idx
            aw, ah = ANCHORS_GROUPS[best_feat][best_aidx]
            offset_w = math.log(bw_new / aw + 1e-16)
            offset_h = math.log(bh_new / ah + 1e-16)

            # ★ 修复：标签平滑
            cls_vec = torch.full((NUM_CLASSES,), LABEL_SMOOTHING / NUM_CLASSES)
            cls_vec[cls_id] = 1.0 - LABEL_SMOOTHING + LABEL_SMOOTHING / NUM_CLASSES

            label_tensors[best_feat][cy_idx, cx_idx, best_aidx] = torch.tensor(
                [1.0, offset_cx, offset_cy, offset_w, offset_h, *cls_vec], dtype=torch.float32
            )

        return (label_tensors[13], label_tensors[26], label_tensors[52], img_tensor)


# -------------------- 损失函数 (修复权重) --------------------
class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # [FIX] reduction='mean' — 每个样本平均，避免负样本数量压倒正样本
        self.mse = nn.MSELoss(reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, predictions, targets, feat_size):
        B = predictions.size(0)
        pred = predictions.permute(0, 2, 3, 1).reshape(B, feat_size, feat_size, 3, 5+NUM_CLASSES)

        pos_mask = targets[..., 0] > 0
        neg_mask = targets[..., 0] == 0

        if pos_mask.any():
            pred_pos = pred[pos_mask]
            target_pos = targets[pos_mask]
            loss_conf = self.bce(pred_pos[:, 0], target_pos[:, 0])
            loss_xy = self.mse(pred_pos[:, 1:3], target_pos[:, 1:3])
            loss_wh = self.mse(pred_pos[:, 3:5], target_pos[:, 3:5])
            loss_cls = self.ce(pred_pos[:, 5:], torch.argmax(target_pos[:, 5:], dim=1))
            pos_loss = loss_conf + loss_xy + loss_wh + loss_cls
        else:
            pos_loss = torch.tensor(0.0, device=predictions.device)

        if neg_mask.any():
            pred_neg = pred[neg_mask]
            target_neg = targets[neg_mask]
            # [FIX] 使用 mean 后，正负样本损失量级相当，用合理权重平衡即可
            neg_loss = self.bce(pred_neg[:, 0], target_neg[:, 0])
        else:
            neg_loss = torch.tensor(0.0, device=predictions.device)

        # [FIX] 正样本权重5.0（含conf+xy+wh+cls），负样本权重0.5（仅conf）
        # 这样正负置信度梯度比例约为 5:0.5=10:1，模型能学会高置信度
        total_loss = 5.0 * pos_loss + 0.5 * neg_loss
        return total_loss


# -------------------- 训练循环 --------------------
def train():
    train_dataset = YOLODataset(DATASET_DIR, split='train', use_mosaic=True)  # [FIX] 启用 Mosaic 增强
    val_dataset = YOLODataset(DATASET_DIR, split='val', use_mosaic=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")

    model = YOLOv3().to(DEVICE)
    criterion = YOLOLoss()
    # ★ Adam：自适应学习率，数值稳定性好，不易NaN
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 150], gamma=0.1)

    os.makedirs('checkpoints', exist_ok=True)
    start_epoch = 1
    best_loss = float('inf')

    if RESUME and os.path.isfile(RESUME):
        print(f'  ✓ 加载 checkpoint: {RESUME}')
        ckpt = torch.load(RESUME, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f'  ✓ 从第 {start_epoch} 轮恢复 (历史最佳: {best_loss:.4f})')

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        # ★ 修复：Warmup 线性预热
        if epoch <= WARMUP_EPOCHS:
            warmup_ratio = epoch / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg['lr'] = LEARNING_RATE * warmup_ratio
        else:
            scheduler.step()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS} [Train]', ncols=100, leave=True)
        for batch_idx, (l13, l26, l52, imgs) in enumerate(pbar):
            l13 = l13.to(DEVICE, non_blocking=True)
            l26 = l26.to(DEVICE, non_blocking=True)
            l52 = l52.to(DEVICE, non_blocking=True)
            imgs = imgs.to(DEVICE, non_blocking=True)

            o13, o26, o52 = model(imgs)
            loss = criterion(o13, l13, 13) + criterion(o26, l26, 26) + criterion(o52, l52, 52)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.2f}', lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        avg_train = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_v = tqdm(val_loader, desc=f'Epoch {epoch}/{EPOCHS} [Val]', ncols=100, leave=True)
            for l13, l26, l52, imgs in pbar_v:
                l13 = l13.to(DEVICE, non_blocking=True)
                l26 = l26.to(DEVICE, non_blocking=True)
                l52 = l52.to(DEVICE, non_blocking=True)
                imgs = imgs.to(DEVICE, non_blocking=True)
                o13, o26, o52 = model(imgs)
                vl = criterion(o13, l13, 13) + criterion(o26, l26, 26) + criterion(o52, l52, 52)
                val_loss += vl.item()
                pbar_v.set_postfix(loss=f'{vl.item():.2f}')

        # 清理显存碎片
        torch.cuda.empty_cache()

        avg_val = val_loss / len(val_loader)
        print(f'Epoch [{epoch}/{EPOCHS}] Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # ★ 修复：统一checkpoint格式，确保best_loss不丢失
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }, 'checkpoints/yolov3_best.pth')
            print(f'  ✓ 保存最佳模型 (loss={best_loss:.4f})')

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }, 'checkpoints/yolov3_last.pth')
            print(f'  ✓ 保存检查点 epoch_{epoch}')

    print(f'\n训练完成！最佳验证损失: {best_loss:.4f}')
    return model

if __name__ == '__main__':
    model = train()