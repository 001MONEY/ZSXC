"""
CIFAR-10 图像分类训练脚本
功能：使用卷积神经网络对 CIFAR-10 数据集进行分类训练
特点：数据增强 + ReLU + BN + SGD + 余弦退火学习率
最终测试准确率：~86%
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设置 matplotlib 中文字体（防止图表中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 网络结构：CIFARNet_ReLU_BN
# 3层卷积 + ReLU + BatchNorm + 最大池化 + 2层全连接
# 输入：3x32x32（RGB彩色图，32x32像素）
# 输出：10（CIFAR-10 的 10 个类别）
# ============================================================
class CIFARNet_ReLU_BN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积特征提取层
        # Conv1:  3  -> 32 通道，输出 16x16（池化后）
        # Conv2:  32 -> 64 通道，输出 8x8
        # Conv3:  64 -> 128 通道，输出 4x4
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 保持尺寸不变
            nn.BatchNorm2d(32),                            # 批归一化，加速收敛
            nn.ReLU(),                                     # ReLU 激活，防止梯度消失
            nn.MaxPool2d(2),                               # 2倍下采样，尺寸减半

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 全连接分类层
        # 输入：128 * 4 * 4 = 2048（展平后）
        # 输出：10 类
        self.classifier = nn.Sequential(
            nn.Flatten(),                                   # 将特征图展平为一维向量
            nn.Linear(128 * 4 * 4, 256),                    # 全连接层：2048 -> 256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 10),                             # 输出层：256 -> 10（各类别分数）
        )

    def forward(self, x):
        """前向传播：输入图片 -> 特征提取 -> 分类 -> 输出类别分数"""
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # ============================================================
    # 1. 设备配置：优先使用 GPU（CUDA），否则使用 CPU
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # ============================================================
    # 2. 数据预处理
    # 训练集：使用数据增强（随机翻转、裁剪、颜色抖动）提高泛化能力
    # 测试集：仅归一化，不做增强（保证评估的客观性）
    # ============================================================
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),                      # 随机水平翻转（模拟左右对称）
        transforms.RandomCrop(32, padding=4),                   # 随机裁剪（四周补4像素，防止边缘信息丢失）
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机调整亮度和对比度
        transforms.ToTensor(),                                  # 转为 Tensor，像素值归一化到 [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化到 [-1, 1]（加速收敛）
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # ============================================================
    # 3. 加载 CIFAR-10 数据集
    # 训练集：50000 张，测试集：10000 张，共 10 类
    # ============================================================
    data_dir = r"D:\Medicaldata"
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_transform)
    test_dataset  = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=test_transform)

    # DataLoader：批量加载数据，训练集打乱顺序，测试集不打乱
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"训练集：{len(train_dataset)}  测试集：{len(test_dataset)}")

    # ============================================================
    # 4. 初始化模型、损失函数、优化器、学习率调度器
    # ============================================================
    net = CIFARNet_ReLU_BN().to(device)
    print(net)

    criterion = nn.CrossEntropyLoss()                   # 交叉熵损失（分类任务标准损失函数）

    # SGD（带动量）+ 权重衰减（L2正则化，防止过拟合）
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # 余弦退火：学习率从 0.01 按余弦曲线下降到接近 0
    # T_max=30 表示 30 个 epoch 完成一个完整余弦周期
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # ============================================================
    # 5. 训练循环
    # ============================================================
    num_epochs = 30
    train_losses, test_accs, lr_history = [], [], []   # 记录训练过程中的指标

    for epoch in range(1, num_epochs + 1):
        # ---------- 训练阶段 ----------
        net.train()                                      # 切换到训练模式（启用 Dropout/BN 的训练行为）
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)  # 数据移到 GPU

            optimizer.zero_grad()                        # 清空上一轮梯度
            outputs = net(inputs)                        # 前向传播：计算预测值
            loss = criterion(outputs, labels)            # 计算损失
            loss.backward()                              # 反向传播：计算梯度
            optimizer.step()                             # 更新模型参数

            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix(loss=loss.item())           # 进度条上显示当前 batch 的 loss

        avg_loss = running_loss / len(train_dataset)
        train_losses.append(avg_loss)

        # ---------- 测试阶段 ----------
        net.eval()                                       # 切换到评估模式
        correct = 0
        total = 0
        with torch.no_grad():                            # 测试阶段不计算梯度，加速并节省显存
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)            # 取最大分数对应的类别
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()  # 统计正确预测数

        acc = 100.0 * correct / total                    # 计算本次 epoch 的测试准确率
        test_accs.append(acc)
        lr_history.append(optimizer.param_groups[0]['lr'])
        print(f"  Loss: {avg_loss:.4f}  |  测试准确率: {acc:.2f}%  |  lr: {optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()                                 # 更新学习率

    # ============================================================
    # 6. 保存训练好的模型
    # ============================================================
    import os
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)                 # 确保 checkpoints 目录存在
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': net.state_dict(),            # 模型权重
        'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态（用于断点续训）
        'test_acc': test_accs[-1],
        'train_loss': train_losses[-1],
    }, os.path.join(save_dir, 'best_model_aug.pth'))
    print(f"✅ 模型已保存至 {save_dir}/best_model_aug.pth")

    # ============================================================
    # 7. 绘制训练曲线
    # ============================================================
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-o', markersize=2)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('训练 Loss')

    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_epochs + 1), test_accs, 'r-o', markersize=2)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('测试准确率')

    plt.subplot(1, 3, 3)
    plt.plot(range(1, num_epochs + 1), lr_history, 'g-')
    plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.title('学习率（余弦退火）')
    plt.tight_layout()
    plt.show()

    print(f"\n🎉 最终测试准确率: {test_accs[-1]:.2f}%")
    print(f"   最高测试准确率: {max(test_accs):.2f}%")
