import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 设备选择
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 100

# 预测时的数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# 加载类别映射
def load_id_to_name(category_path='category.txt'):
    id_to_name = {}
    with open(category_path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过标题行
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                cls_id, name = parts
                id_to_name[int(cls_id) - 1] = name  # 减1与模型index对齐
    return id_to_name


# 加载模型
def load_model(weights_path='resnet_classifier/model.pth'):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


# 图片预测
def predict_image(image_path, model, id_to_name):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        pred_class = pred.item()

    name = id_to_name.get(pred_class, "未知")
    print("✅ 预测结果：")
    print(f"预测类索引：{pred_class}")
    print(f"菜品名称：{name}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    args = parser.parse_args()

    # 加载必要内容
    id_to_name = load_id_to_name('category.txt')
    model = load_model()

    # 执行预测
    predict_image(args.image, model, id_to_name)
