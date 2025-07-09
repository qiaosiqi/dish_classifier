# resnet_predict.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 模型路径 & 类别映射
MODEL_PATH = 'resnet_classifier/model.pth'
ID_NAME_PATH = 'resnet_classifier/id_name_mapping.txt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载类别映射
def load_id2name():
    id2name = {}
    with open(ID_NAME_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('id'):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    id2name[int(parts[0]) - 1] = parts[1]  # ImageFolder 从 0 开始编号
    return id2name

id2name = load_id2name()

# 加载模型
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

model = load_model()

# 预测函数
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        pred_id = pred.item()
        pred_name = id2name.get(pred_id, "unknown")
        return {'id': pred_id + 1, 'name': pred_name}
