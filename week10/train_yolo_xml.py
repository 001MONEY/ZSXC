# ============================================================
# YOLOv3 训练 —— 从 图片文件/ + 标注文件/ 读取数据
# 3 类：red(0), black(1), white(2)
# ============================================================
import torch
import torch.nn as nn
import math
import os
import random
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# -------------------- 配置 --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 300
IMG_SIZE = 416
NUM_CLASSES = 3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005

# 图片和标注文件夹路径（相对于项目根目录）
IMG_DIR = r"图片文件"
XML_DIR = r"标注文件"
OUTPUT_DIR = r"checkpoints"

# 类别映射（xml 中的 name → id）
CLASS_NAME_TO_ID = {
    "red": 0,
    "black": 1,
    "white": 2,
}

CLASSES = ["red", "black", "white"]

# Anchor（基于数据集 K-Means 聚类得到）
ANCHORS_GROUPS = {
    13: [[87, 74], [86, 104], [105, 88]],
    26: [[80, 50], [49, 84], [72, 84]],
    52: [[38, 58], [53, 43], [60, 64]],
}

print(f"设备: {DEVICE}")
print(f"类别: {CLASSES}")
print(f"图片目录: {os.path.abspath(IMG_DIR)}")
print(f"标注目录: {os.path.abspath(XML_DIR)}")


# -------------------- 解析 XML 标注 --------------------
def parse_xml(xml_path):
    """解析 Pascal VOC XML 标注文件，返回 (cls_id, cx, cy, w, h) 列表"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 读取图片尺寸
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    for obj in root.iter("object"):
        name = obj.find("name").text
        if name not in CLASS_NAME_TO_ID:
            continue  # 跳过未定义的类别
        cls_id = CLASS_NAME_TO_ID[name]

        bndbox = obj.find("bndbox")
        x1 = float(bndbox.findtext("xmin"))
        y1 = float(bndbox.findtext("ymin"))
        x2 = float(bndbox.findtext("xmax"))
        y2 = float(bndbox.findtext("ymax"))

        # 计算中心点坐标和宽高（像素坐标）
        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw / 2
        cy = y1 + bh / 2

        objects.append((cls_id, cx, cy, bw, bh))

    return objects, width, height


# -------------------- 数据集 --------------------
class YOLOXMLDataset(Dataset):
    """从图片文件目录和 XML 标注目录读取训练数据"""
    def __init__(self, img_dir, xml_dir, split="train", val_ratio=0.15, transform=None):
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        self.transform = transform or transforms.ToTensor()

        # 收集所有匹配的图片-标注对
        all_pairs = []
        xml_names = {os.path.splitext(f)[0] for f in os.listdir(xml_dir) if f.endswith(".xml")}
        img_names = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith((".jpg", ".jpeg", ".png"))}
        common = sorted(xml_names & img_names, key=lambda x: int(x) if x.isdigit() else x)

        for name in common:
            img_path = os.path.join(img_dir, f"{name}.jpg")
            xml_path = os.path.join(xml_dir, f"{name}.xml")
            if os.path.exists(img_path) and os.path.exists(xml_path):
                all_pairs.append((img_path, xml_path))

        if len(all_pairs) == 0:
            raise RuntimeError(f"没有找到匹配的图片-标注对！\n  图片数: {len(img_names)}, 标注数: {len(xml_names)}, 交集: {len(common)}")

        # 划分 train / val
        random.seed(42)
        indices = list(range(len(all_pairs)))
        random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        if split == "val":
            self.pairs = [all_pairs[i] for i in indices[:n_val]]
        else:
            self.pairs = [all_pairs[i] for i in indices[n_val:]]

        print(f"  [{split}] {len(self.pairs)} 个样本 (共 {len(all_pairs)} 个匹配)")

    def __len__(self):
        return len(self.pairs)

    def _letterbox(self, img, color=(114, 114, 114)):
        """保持宽高比的 resize + 填充到 IMG_SIZE"""
        w, h = img.size
        ratio = min(IMG_SIZE / w, IMG_SIZE / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        dw = (IMG_SIZE - new_w) / 2
        dh = (IMG_SIZE - new_h) / 2
        result = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color)
        result.paste(img, (int(dw), int(dh)))
        return result, ratio, (dw, dh)

    def __getitem__(self, index):
        img_path, xml_path = self.pairs[index]

        # 读取原图
        img = Image.open(img_path).convert("RGB")

        # 解析标注（像素坐标）
        objects, _, _ = parse_xml(xml_path)

        # letterbox resize
        img_resized, ratio, (dw, dh) = self._letterbox(img)
        img_tensor = self.transform(img_resized)

        # ★ 用新的标签分配函数（含 ignore 区域）
        label_dict = assign_labels(objects, ratio, dw, dh)
        l13 = label_dict[13]["target"]
        l26 = label_dict[26]["target"]
        l52 = label_dict[52]["target"]
        m13 = label_dict[13]["noobj_mask"]
        m26 = label_dict[26]["noobj_mask"]
        m52 = label_dict[52]["noobj_mask"]

        return l13, l26, l52, img_tensor, m13, m26, m52


# -------------------- 模型组件 --------------------
class CBL(nn.Module):
    """Conv + BN + LeakyReLU"""
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
        out_52 = self.stages[:3](x)   # 52×52
        out_26 = self.stages[3](out_52)  # 26×26
        out_13 = self.stages[4](out_26)  # 13×13
        return out_52, out_26, out_13


class CBLSET(nn.Module):
    """ConvSet: 1×1, 3×3, 1×1, 3×3, 1×1"""
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
    """YOLOv3 完整网络"""
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
        # 13×13
        cov_set_13_out = self.cov_set_13(out_13)
        out_neck_13 = self.out_neck_13(cov_set_13_out)
        # 13 → 26
        up_13_out_26 = self.up_13(cov_set_13_out)
        concat_26 = torch.cat((up_13_out_26, out_26), dim=1)
        cov_set_26_out = self.cov_set_26(concat_26)
        out_neck_26 = self.out_neck_26(cov_set_26_out)
        # 26 → 52
        up_26_out_52 = self.up_26_52(cov_set_26_out)
        concat_52 = torch.cat((up_26_out_52, out_52), dim=1)
        out_neck_52 = self.neck_52(concat_52)
        return out_neck_13, out_neck_26, out_neck_52


# -------------------- 标签分配（增加 ignore 区域 + IoU阈值） --------------------
def assign_labels(objects, ratio, dw, dh):
    """
    为一张图的所有GT分配标签
    返回: {feat_size: {'target': tensor, 'obj_mask': bool, 'noobj_mask': bool}}
    """
    targets = {}
    for feat_size in ANCHORS_GROUPS:
        n_anchors = len(ANCHORS_GROUPS[feat_size])
        targets[feat_size] = {
            "target": torch.zeros((feat_size, feat_size, n_anchors, 5 + NUM_CLASSES)),
            "obj_mask": torch.zeros((feat_size, feat_size, n_anchors), dtype=torch.bool),
            "noobj_mask": torch.ones((feat_size, feat_size, n_anchors), dtype=torch.bool),
        }

    IGNORE_IOU_THRESH = 0.5  # IoU > 此值的anchor不参与负样本loss

    for cls_id, cx, cy, bw, bh in objects:
        cx_new = cx * ratio + dw
        cy_new = cy * ratio + dh
        bw_new = bw * ratio
        bh_new = bh * ratio

        # 找全局最佳 anchor（跨尺度 + 跨anchor）
        best_iou, best_feat, best_aidx, best_gx, best_gy = 0, None, 0, 0, 0
        for feat_size, anchors in ANCHORS_GROUPS.items():
            stride = IMG_SIZE / feat_size
            gx = int(cx_new // stride)
            gy = int(cy_new // stride)
            if not (0 <= gx < feat_size and 0 <= gy < feat_size):
                continue
            for aidx, (aw, ah) in enumerate(anchors):
                inter = min(bw_new, aw) * min(bh_new, ah)
                union = bw_new * bh_new + aw * ah - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_feat, best_aidx = feat_size, aidx
                    best_gx, best_gy = gx, gy

        if best_feat is None:
            continue

        # 正样本赋值
        stride = IMG_SIZE / best_feat
        offset_cx = (cx_new / stride) - best_gx
        offset_cy = (cy_new / stride) - best_gy
        aw, ah = ANCHORS_GROUPS[best_feat][best_aidx]
        offset_w = math.log(bw_new / aw + 1e-16)
        offset_h = math.log(bh_new / ah + 1e-16)

        cls_vec = torch.zeros(NUM_CLASSES)
        cls_vec[cls_id] = 1.0

        t = targets[best_feat]
        t["target"][best_gy, best_gx, best_aidx] = torch.tensor(
            [offset_cx, offset_cy, offset_w, offset_h, 1.0, *cls_vec], dtype=torch.float32
        )
        t["obj_mask"][best_gy, best_gx, best_aidx] = True
        t["noobj_mask"][best_gy, best_gx, best_aidx] = False

        # ★ Ignore: 与该GT IoU较高的其他anchor，不惩罚也不奖励
        for feat_size, anchors in ANCHORS_GROUPS.items():
            stride = IMG_SIZE / feat_size
            gx = int(cx_new // stride)
            gy = int(cy_new // stride)
            if not (0 <= gx < feat_size and 0 <= gy < feat_size):
                continue
            for aidx, (aw, ah) in enumerate(anchors):
                if targets[feat_size]["obj_mask"][gy, gx, aidx]:
                    continue  # 已经是正样本
                inter = min(bw_new, aw) * min(bh_new, ah)
                union = bw_new * bh_new + aw * ah - inter
                iou = inter / union if union > 0 else 0
                if iou > IGNORE_IOU_THRESH:
                    targets[feat_size]["noobj_mask"][gy, gx, aidx] = False

    return targets

# -------------------- 损失函数（GPT改进版） --------------------
#  ① BCE for cls（与sigmoid推理一致）
#  ② sigmoid on xy（YOLOv3要求xy偏移经sigmoid）
#  ③ sum reduction + 高正权重

class YOLOLoss(nn.Module):
    def __init__(self, pos_weight=15.0, neg_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions, targets, noobj_mask, feat_size):
        """targets格式: [..., (tx,ty,tw,th,conf,cls...)]"""
        B = predictions.size(0)
        pred = predictions.permute(0, 2, 3, 1).reshape(B, feat_size, feat_size, 3, 5 + NUM_CLASSES)

        pos_mask = targets[..., 4] > 0   # conf 在位置4
        noobj_mask = noobj_mask.to(dtype=torch.bool) & (targets[..., 4] == 0)

        pos_loss = torch.tensor(0.0, device=predictions.device)
        if pos_mask.any():
            pred_pos = pred[pos_mask]
            tgt_pos = targets[pos_mask]
            xy_loss = self.mse(torch.sigmoid(pred_pos[:, :2]), tgt_pos[:, :2])
            wh_loss = self.mse(pred_pos[:, 2:4], tgt_pos[:, 2:4])
            conf_loss = self.bce(pred_pos[:, 4], tgt_pos[:, 4])
            cls_loss = self.bce(pred_pos[:, 5:], tgt_pos[:, 5:])
            pos_loss = xy_loss + wh_loss + conf_loss + cls_loss

        neg_loss = torch.tensor(0.0, device=predictions.device)
        if noobj_mask.any():
            pred_neg = pred[noobj_mask]
            neg_loss = self.bce(pred_neg[:, 4], torch.zeros_like(pred_neg[:, 4]))

        return self.pos_weight * pos_loss + self.neg_weight * neg_loss


# -------------------- 训练函数 --------------------
def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 数据集
    train_dataset = YOLOXMLDataset(IMG_DIR, XML_DIR, split="train", val_ratio=0.15)
    val_dataset = YOLOXMLDataset(IMG_DIR, XML_DIR, split="val", val_ratio=0.15)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")

    # 模型
    model = YOLOv3().to(DEVICE)
    criterion = YOLOLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ---- 训练 ----
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", ncols=100)
        for l13, l26, l52, imgs, nm13, nm26, nm52 in pbar:
            l13 = l13.to(DEVICE, non_blocking=True)
            l26 = l26.to(DEVICE, non_blocking=True)
            l52 = l52.to(DEVICE, non_blocking=True)
            nm13 = nm13.to(DEVICE, non_blocking=True)
            nm26 = nm26.to(DEVICE, non_blocking=True)
            nm52 = nm52.to(DEVICE, non_blocking=True)
            imgs = imgs.to(DEVICE, non_blocking=True)

            o13, o26, o52 = model(imgs)
            loss = (
                criterion(o13, l13, nm13, 13)
                + criterion(o26, l26, nm26, 26)
                + criterion(o52, l52, nm52, 52)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # ---- 验证 ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for l13, l26, l52, imgs, nm13, nm26, nm52 in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", ncols=100):
                l13 = l13.to(DEVICE, non_blocking=True)
                l26 = l26.to(DEVICE, non_blocking=True)
                l52 = l52.to(DEVICE, non_blocking=True)
                nm13 = nm13.to(DEVICE, non_blocking=True)
                nm26 = nm26.to(DEVICE, non_blocking=True)
                nm52 = nm52.to(DEVICE, non_blocking=True)
                imgs = imgs.to(DEVICE, non_blocking=True)

                o13, o26, o52 = model(imgs)
                loss = (
                    criterion(o13, l13, nm13, 13)
                    + criterion(o26, l26, nm26, 26)
                    + criterion(o52, l52, nm52, 52)
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "yolov3_best.pth"))
            print(f"  ✓ 保存最佳模型 (loss={best_loss:.4f})")

        # 定期保存 checkpoint
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"yolov3_epoch{epoch}.pth"))
            print(f"  ✓ 保存 checkpoint epoch {epoch}")

    print(f"\n训练完成！最佳模型已保存至: {os.path.join(OUTPUT_DIR, 'yolov3_best.pth')}")


if __name__ == "__main__":
    train()
