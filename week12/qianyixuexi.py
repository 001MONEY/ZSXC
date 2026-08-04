"""
迁移学习：
对比-有预训练模型-没有预训练模型 
对比-优化器：ADam-SGD

1. 没有预训练模型+ADam

"""



import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==================== 1. 数据准备 ====================
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),     # 随机裁剪 → 统一 224×224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),                 # 先缩放到 256
        transforms.CenterCrop(224),             # 中心裁出 224×224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(root=r"../datasets/hymenoptera_data/train", transform=train_tf)
    val_set = datasets.ImageFolder(root=r"../datasets/hymenoptera_data/val", transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=2)

    # ==================== 2. 构建模型 ====================
    # model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    model = models.resnet50()
    print(model)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(train_set.classes))
    model = model.to(DEVICE)

    # ==================== 3. 训练 ====================
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    EPOCHS = 10
    print(f"\n开始训练 {EPOCHS} 轮...\n{'='*60}")
    for epoch in range(EPOCHS):
        # ---- 训练 ----
        model.train()
        train_loss, train_correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()


        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"训练 Loss: {train_loss/len(train_loader):.4f}  ")
             
    # ==================== 4. 保存模型 ====================
    torch.save(model.state_dict(), "resnet50_imagenette.pth")
    print(f"\n模型已保存 → resnet50_imagenette.pth")