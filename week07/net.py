import torch
import torch.nn as nn


class ThreeLayerFCNet(nn.Module):
    """三层全连接网络（输出 logits，配合 CrossEntropyLoss）"""
    def __init__(self, input_dim=784, hidden1=512, hidden2=256, num_classes=10):
        super().__init__()
        self.fc1   = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3   = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


class ThreeLayerFCNet_MSE(nn.Module):
    """三层全连接网络 + Softmax（配合 MSELoss）"""
    def __init__(self, input_dim=784, hidden1=512, hidden2=256, num_classes=10):
        super().__init__()
        self.fc1     = nn.Linear(input_dim, hidden1)
        self.relu1   = nn.ReLU()
        self.fc2     = nn.Linear(hidden1, hidden2)
        self.relu2   = nn.ReLU()
        self.fc3     = nn.Linear(hidden2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.softmax(self.fc3(x))


class CIFARNet_ReLU_BN(nn.Module):
    """CIFAR-10 分类网络：Conv + ReLU + BatchNorm"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
