import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# 超参数
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
NUM_CLASSES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 训练函数
def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(train_loader.dataset)

# 验证函数
def validate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    # 数据加载
    train_dataset = datasets.ImageFolder(
        'D:\\CODE\\myGitHub\\dish_classifier\\UECFOOD100_resnet\\train',
        transform=transform_train)
    val_dataset = datasets.ImageFolder(
        'D:\\CODE\\myGitHub\\dish_classifier\\UECFOOD100_resnet\\val',
        transform=transform_val)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 模型加载
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 训练循环
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc = validate(model, val_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 保存模型
    os.makedirs('resnet_classifier', exist_ok=True)
    torch.save(model.state_dict(), 'resnet_classifier/model2.pth')
    print("✅ 训练完成，模型已保存！")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
