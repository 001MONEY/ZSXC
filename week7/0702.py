import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


# ========== 网络：ReLU + BN ==========
class CIFARNet_ReLU_BN(nn.Module):
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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # ========== 训练集：数据增强 ==========
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_dir = r"D:\Medicaldata"
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_transform)
    test_dataset  = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"训练集：{len(train_dataset)}  测试集：{len(test_dataset)}")

    net = CIFARNet_ReLU_BN().to(device)
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    num_epochs = 30
    train_losses, test_accs, lr_history = [], [], []

    for epoch in range(1, num_epochs + 1):
        net.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_dataset)
        train_losses.append(avg_loss)

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = 100.0 * correct / total
        test_accs.append(acc)
        lr_history.append(optimizer.param_groups[0]['lr'])
        print(f"  Loss: {avg_loss:.4f}  |  测试准确率: {acc:.2f}%  |  lr: {optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step()

    # ========== 保存模型 ==========
    import os
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_accs[-1],
        'train_loss': train_losses[-1],
    }, os.path.join(save_dir, 'best_model_aug.pth'))
    print(f"✅ 模型已保存至 {save_dir}/best_model_aug.pth")

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
