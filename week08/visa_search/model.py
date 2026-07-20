"""
VisA 以图搜图 — 特征提取模型
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from config import DEVICE


class FeatureExtractor:
    """加载预训练模型，提取特征向量"""

    def __init__(self, model_name='resnet18'):
        self.model_name = model_name
        self.model, self.pool, self.feat_dim = self._build_model(model_name)
        self.model.eval().to(DEVICE)

        # 图片预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        params = sum(p.numel() for p in self.model.parameters())
        print(f"  模型: {model_name} | 参数量: {params/1e6:.2f}M | 特征维度: {self.feat_dim}")

    def _build_model(self, name):
        if name == 'resnet18':
            model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
            feat_layer = nn.Sequential(*list(model.children())[:-2])
            pool = nn.AdaptiveAvgPool2d((1, 1))
            feat_dim = 512

        elif name == 'mobilenet_v2':
            model = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
            feat_layer = model.features
            pool = nn.AdaptiveAvgPool2d((1, 1))
            feat_dim = 1280

        else:
            raise ValueError(f"未知模型: {name}（可选: resnet18 / mobilenet_v2）")

        return feat_layer, pool, feat_dim

    def extract(self, img_path):
        """
        提取单张图片的特征向量（与讲义一致）。

        参数:
            img_path: 图片文件路径
        返回:
            1280 维 numpy 数组 (未归一化，与讲义一致)
        """
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat = self.model(tensor)       # [1, C, H, W]
            feat = self.pool(feat)          # [1, C, 1, 1]
            feat = feat.view(1, -1)         # [1, C]

        return feat.cpu().numpy().flatten()

    def extract_from_tensor(self, img_tensor):
        """直接从预处理好的张量提取特征"""
        with torch.no_grad():
            feat = self.model(img_tensor)
            feat = self.pool(feat)
            feat = feat.view(1, -1)
        return feat.cpu().numpy().flatten()


if __name__ == '__main__':
    # 测试
    extractor = FeatureExtractor('resnet18')
    print(f"  设备: {DEVICE}")
    dummy = Image.new('RGB', (224, 224))
    feat = extractor.extract('__dummy__')
    print(f"  提取特征维度: {len(feat)}")
