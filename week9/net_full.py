# encoding: utf-8
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1):
        super().__init__()
        p = (k-1)//2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.lr = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        return self.lr(self.bn(self.conv(x)))


class ConvNoBN(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, (k-1)//2, bias=True)
    def forward(self, x):
        return self.conv(x)


class ResUnit(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.b = nn.Sequential(ConvBlock(ch,ch//2,1,1), ConvBlock(ch//2,ch,3,1))
    def forward(self, x):
        return x + self.b(x)


class ResBlock(nn.Module):
    def __init__(self, ch, n):
        super().__init__()
        self.u = nn.Sequential(*[ResUnit(ch) for _ in range(n)])
    def forward(self, x):
        return self.u(x)


class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.s0=ConvBlock(3,32,3,1)
        self.s1=nn.Sequential(ConvBlock(32,64,3,2),ResBlock(64,1))
        self.s2=nn.Sequential(ConvBlock(64,128,3,2),ResBlock(128,2))
        self.s3=nn.Sequential(ConvBlock(128,256,3,2),ResBlock(256,8))
        self.s4=nn.Sequential(ConvBlock(256,512,3,2),ResBlock(512,8))
        self.s5=nn.Sequential(ConvBlock(512,1024,3,2),ResBlock(1024,4))
    def forward(self, x):
        x=self.s0(x); x=self.s1(x); x=self.s2(x); x=self.s3(x)
        r52=x; x=self.s4(x); r26=x; x=self.s5(x); r13=x
        return r52,r26,r13


class YOLOHead(nn.Module):
    def __init__(self, in_ch, nc=80, na=3):
        super().__init__()
        oc=na*(5+nc)
        self.c=nn.Sequential(
            ConvBlock(in_ch,in_ch//2,1,1), ConvBlock(in_ch//2,in_ch,3,1),
            ConvBlock(in_ch,in_ch//2,1,1), ConvBlock(in_ch//2,in_ch,3,1),
            ConvBlock(in_ch,in_ch//2,1,1))
        self.p=ConvNoBN(in_ch//2,oc,1,1)
    def forward(self, x):
        return self.p(self.c(x))


class YOLOv3(nn.Module):
    def __init__(self, nc=80):
        super().__init__()
        self.nc=nc
        self.anchors=torch.tensor([[10,13],[16,30],[33,23],[30,61],[62,45],
            [59,119],[116,90],[156,198],[373,326]],dtype=torch.float32)
        self.b=Darknet53()
        self.h13=YOLOHead(1024,nc,3)
        self.h26=YOLOHead(512,nc,3)
        self.h52=YOLOHead(256,nc,3)
        self.u13=nn.Sequential(ConvBlock(1024,512,1,1),nn.Upsample(scale_factor=2,mode='nearest'))
        self.u26=nn.Sequential(ConvBlock(512,256,1,1),nn.Upsample(scale_factor=2,mode='nearest'))
        self.r26=ConvBlock(1024,512,1,1)
        self.r52=ConvBlock(512,256,1,1)
    def forward(self, x):
        r52,r26,r13=self.b(x)
        o13=self.h13(r13); u13=self.u13(r13)
        c26=self.r26(torch.cat([u13,r26],1)); o26=self.h26(c26); u26=self.u26(c26)
        o52=self.h52(self.r52(torch.cat([u26,r52],1)))
        return o13,o26,o52
    @property
    def device(self):
        return next(self.parameters()).device


# ====================================================================
#  YOLO Dataset
# ====================================================================

from torch.utils.data import Dataset
import math

class YOLODataset(Dataset):
    """
    YOLOv3 数据集

    返回: (label_13, label_26, label_52, img_tensor)
      - label_13: [13, 13, 3, 5+num_classes]
      - label_26: [26, 26, 3, 5+num_classes]
      - label_52: [52, 52, 3, 5+num_classes]
      - img_tensor: [3, 416, 416]
    """
    def __init__(self, root_path, img_size=416, num_classes=80,
                 anchors=None, anchor_groups=None):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.dataset = open(root_path, 'r', encoding='utf-8').readlines()

        if anchors is None:
            self.anchors = torch.tensor([
                [10,13],[16,30],[33,23],
                [30,61],[62,45],[59,119],
                [116,90],[156,198],[373,326]], dtype=torch.float32)
        else:
            self.anchors = anchors

        if anchor_groups is None:
            self.anchor_groups = {
                13: self.anchors[6:9],
                26: self.anchors[3:6],
                52: self.anchors[0:3],
            }
        else:
            self.anchor_groups = anchor_groups

        print(f'[YOLODataset] 加载 {len(self.dataset)} 张图片, '
              f'{num_classes} 类')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        line = self.dataset[idx].strip()
        parts = line.split()
        img_path = parts[0]

        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img_tensor = torch.from_numpy(np.array(img)).float().permute(2,0,1) / 255.0

        nums = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        boxes = nums.reshape(-1, 5)

        labels = {}
        for feat_size in [13, 26, 52]:
            labels[feat_size] = torch.zeros(
                feat_size, feat_size, 3, 5 + self.num_classes)

        for box in boxes:
            cls_id, cx, cy, bw, bh = box

            for feat_size, anchors_fs in self.anchor_groups.items():
                stride = self.img_size // feat_size

                cx_idx = cx * feat_size
                cy_idx = cy * feat_size
                grid_x = int(math.floor(cx_idx))
                grid_y = int(math.floor(cy_idx))
                offset_x = cx_idx - grid_x
                offset_y = cy_idx - grid_y

                if grid_x < 0 or grid_x >= feat_size: continue
                if grid_y < 0 or grid_y >= feat_size: continue

                for a_idx in range(3):
                    anchor_w = anchors_fs[a_idx, 0]
                    anchor_h = anchors_fs[a_idx, 1]

                    bw_px = bw * self.img_size
                    bh_px = bh * self.img_size
                    offset_w = float(np.log(bw_px / anchor_w.item() + 1e-8))
                    offset_h = float(np.log(bh_px / anchor_h.item() + 1e-8))

                    cls_onehot = torch.zeros(self.num_classes)
                    cls_onehot[int(cls_id)] = 1.0

                    val = [1.0, offset_x, offset_y, offset_w, offset_h]
                    labels[feat_size][grid_y, grid_x, a_idx] = \
                        torch.tensor(val + cls_onehot.tolist(), dtype=torch.float32)

        return labels[13], labels[26], labels[52], img_tensor


# ====================================================================
#  解码
# ====================================================================

def decode_scale(o, anchors, nc, stride, sz=416):
    B,_,H,W=o.shape; na=len(anchors)
    out=o.view(B,na,5+nc,H,W).permute(0,1,3,4,2)
    tx,ty,tw,th,obj=out[...,0],out[...,1],out[...,2],out[...,3],out[...,4]
    cls=out[...,5:]
    gx=torch.arange(W,device=o.device).float().view(1,1,1,W)
    gy=torch.arange(H,device=o.device).float().view(1,1,H,1)
    bx=(torch.sigmoid(tx)+gx)*stride; by=(torch.sigmoid(ty)+gy)*stride
    anc=anchors.to(o.device).view(1,na,1,1,2)
    bw=anc[...,0]*torch.exp(tw); bh=anc[...,1]*torch.exp(th)
    conf=torch.sigmoid(obj)
    cc,ci=torch.softmax(cls,-1).max(dim=-1,keepdim=True)
    fc=conf.unsqueeze(-1)*cc
    x1=(bx-bw/2).clamp(0,sz); y1=(by-bh/2).clamp(0,sz)
    x2=(bx+bw/2).clamp(0,sz); y2=(by+bh/2).clamp(0,sz)
    return torch.stack([x1,y1,x2,y2,fc.squeeze(-1),ci.squeeze(-1).float()],-1).view(B,-1,6)


def decode_all(o13,o26,o52,anchors,nc,sz=416):
    return torch.cat([
        decode_scale(o13,anchors[6:9],nc,32,sz),
        decode_scale(o26,anchors[3:6],nc,16,sz),
        decode_scale(o52,anchors[0:3],nc,8,sz)],dim=1)


def nms(boxes, ct=0.5, it=0.45):
    if boxes.shape[0]==0: return []
    b=boxes[boxes[:,4]>ct]
    if b.shape[0]==0: return []
    o=b[:,4].argsort(descending=True); b=b[o]; k=[]
    while b.shape[0]>0:
        k.append(o[0].item())
        if b.shape[0]==1: break
        r=b[1:]
        i=torch.clamp(torch.minimum(b[0,2],r[:,2])-torch.maximum(b[0,0],r[:,0]),min=0)*\
           torch.clamp(torch.minimum(b[0,3],r[:,3])-torch.maximum(b[0,1],r[:,1]),min=0)
        u=(b[0,2]-b[0,0])*(b[0,3]-b[0,1])+(r[:,2]-r[:,0])*(r[:,3]-r[:,1])-i
        m=(i/(u+1e-6)<=it)|(b[0,5]!=r[:,5])
        b=r[m]; o=o[1:][m]
    return k


def post_process(ab, ct=0.5, it=0.45):
    res=[]
    for i in range(ab.shape[0]):
        b=ab[i]; b=b[b[:,4]>ct]
        if b.shape[0]==0: res.append(torch.zeros((0,6),device=ab.device)); continue
        o=b[:,4].argsort(descending=True); b=b[o]; r=[]
        while b.shape[0]>0:
            r.append(b[0:1])
            if b.shape[0]==1: break
            rest=b[1:]
            i=torch.clamp(torch.minimum(b[0,2],rest[:,2])-torch.maximum(b[0,0],rest[:,0]),min=0)*\
               torch.clamp(torch.minimum(b[0,3],rest[:,3])-torch.maximum(b[0,1],rest[:,1]),min=0)
            u=(b[0,2]-b[0,0])*(b[0,3]-b[0,1])+(rest[:,2]-rest[:,0])*(rest[:,3]-rest[:,1])-i
            m=(i/(u+1e-6)<=it)|(b[0,5]!=rest[:,5])
            b=rest[m]
        res.append(torch.cat(r,dim=0) if r else torch.zeros((0,6),device=ab.device))
    return res


COCO_CLASSES=['person','bicycle','car','motorcycle','airplane','bus','train',
    'truck','boat','traffic light','fire hydrant','stop sign','parking meter',
    'bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra',
    'giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis',
    'snowboard','sports ball','kite','baseball bat','baseball glove','skateboard',
    'surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon',
    'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
    'donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv',
    'laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster',
    'sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']


COLORS=[tuple(int(c*255) for c in plt.cm.hsv(i/80)[:3]) for i in range(80)]


def draw_boxes(img, boxes, ct=0.5):
    img=img.copy()
    for b in boxes:
        if b[4]<ct: continue
        x1,y1,x2,y2=map(int,[b[0],b[1],b[2],b[3]]); cid=int(b[5])
        lbl=COCO_CLASSES[cid] if cid<80 else f'c{cid}'
        cv2.rectangle(img,(x1,y1),(x2,y2),COLORS[cid%80],2)
        cv2.putText(img,f'{lbl} {b[4]:.2f}',(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[cid%80],2)
    return img


def detect_image(model, path, ct=0.5, it=0.45, sz=416):
    model.eval()
    pil=Image.open(path).convert('RGB'); rgb=np.array(pil); bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    h,w=rgb.shape[:2]
    x=torch.from_numpy(cv2.resize(rgb,(sz,sz))).float().permute(2,0,1).unsqueeze(0)/255
    with torch.no_grad():
        o13,o26,o52=model(x)
        ab=decode_all(o13,o26,o52,model.anchors,model.nc,sz)
        dets=post_process(ab,ct,it)[0]
    if dets.shape[0]>0:
        dets[:,[0,2]]*=w/sz; dets[:,[1,3]]*=h/sz
    return draw_boxes(bgr,dets.cpu().numpy(),ct),dets


def test_forward():
    print('='*50)
    print('  YOLOv3 前向传播测试 (随机权重)')
    print('='*50)
    m=YOLOv3(80); m.eval()
    print(f'总参数量: {sum(p.numel() for p in m.parameters()):,}')
    x=torch.randn(1,3,416,416)
    with torch.no_grad():
        o13,o26,o52=m(x)
    print(f'out_13: {o13.shape}  (stride=32)')
    print(f'out_26: {o26.shape}  (stride=16)')
    print(f'out_52: {o52.shape}  (stride=8)')
    ab=decode_all(o13,o26,o52,m.anchors,80)
    print(f'\n解码框数: {ab.shape[1]} (理论 10647)')
    res=post_process(ab)
    print(f'NMS 后: {res[0].shape[0]} 个框')
    print('测试通过!')


def test_dataset():
    """生成模拟数据, 测试 Dataset 是否能正常运行"""
    import tempfile, shutil

    tmpdir = tempfile.mkdtemp()
    img_dir = os.path.join(tmpdir, 'images')
    os.makedirs(img_dir)

    train_lines = []
    for i in range(2):
        img = np.random.randint(0, 256, (300, 500, 3), dtype=np.uint8)
        img_path = os.path.join(img_dir, f'{i}.jpg').replace('\\', '/')
        cv2.imwrite(img_path, img)
        num_obj = np.random.randint(1, 4)
        line = img_path
        for _ in range(num_obj):
            xc = np.random.uniform(0.1, 0.9)
            yc = np.random.uniform(0.1, 0.9)
            w = np.random.uniform(0.05, 0.3)
            h = np.random.uniform(0.05, 0.3)
            cls = np.random.randint(0, 3)
            line += f' {cls} {xc:.4f} {yc:.4f} {w:.4f} {h:.4f}'
        train_lines.append(line)

    train_txt = os.path.join(tmpdir, 'train.txt')
    with open(train_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))

    ds = YOLODataset(train_txt, num_classes=3)
    print(f'\n📊 Dataset 测试:')
    print(f'   样本数: {len(ds)}')

    l13, l26, l52, img = ds[0]
    print(f'   图片张量: {img.shape}')
    print(f'   label_13: {l13.shape}  (有目标格子: {(l13[...,0]==1).sum()})')
    print(f'   label_26: {l26.shape}  (有目标格子: {(l26[...,0]==1).sum()})')
    print(f'   label_52: {l52.shape}  (有目标格子: {(l52[...,0]==1).sum()})')

    for feat_size, lbl in [(13, l13), (26, l26), (52, l52)]:
        mask = lbl[..., 0] == 1
        if mask.any():
            offsets = lbl[..., 1:5][mask]
            print(f'   {feat_size}×{feat_size}: {mask.sum()} 个正样本, '
                  f'offset 范围: [{offsets.min():.2f}, {offsets.max():.2f}]')

    print('✅ Dataset 测试通过!')
    shutil.rmtree(tmpdir)


if __name__=='__main__':
    test_forward()
    test_dataset()
