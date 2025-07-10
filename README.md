
# 🍱 Dish Classifier

基于 [UECFOOD100](https://www.kaggle.com/datasets/lsind18/uecfood100) 数据集的菜品识别系统，支持目标检测（YOLOv5）与图像分类（ResNet50）两种方式。

---

## 📁 项目结构（简略）

```
dish_classifier/
├── yolov5/                  # YOLOv5源码目录
├── UECFOOD100_YOLO/         # YOLO格式的数据集
├── UECFOOD100_resnet/       # ResNet格式的数据集
├── resnet_train.py          # ResNet模型训练脚本
├── resnet_predict.py        # ResNet预测脚本
├── predict.py               # YOLOv5的预测脚本
├── local_test.py            # 本地图片预测测试（YOLO）
├── app.py                   # FastAPI接口（YOLO）
├── category.txt             # 类别索引与菜品名称映射
└── ...
```

---

## 🚀 本地部署与运行

### 1. 克隆仓库

```bash
git clone https://github.com/qiaosiqi/dish_classifier.git
cd dish_classifier
```

### 2. 安装依赖

建议使用 Python 3.10+，可创建虚拟环境后：

```bash
pip install -r yolov5/requirements.txt
pip install torch torchvision fastapi uvicorn
```

---

## 🏋️‍♀️ 模型训练

### 【YOLOv5】

确保你已准备好 `UECFOOD100_YOLO` 数据集结构如下：

```
UECFOOD100_YOLO/
├── images/
│   ├── train/xxx.jpg
│   └── val/xxx.jpg
├── labels/
│   ├── train/xxx.txt
│   └── val/xxx.txt
└── uec_food.yaml
```

运行训练：

```bash
cd yolov5
python train.py --img 640 --batch 16 --epochs 50 --data ../UECFOOD100_YOLO/uec_food.yaml --weights yolov5s.pt --name lowmem_run
```

### 【ResNet50】

确保你的图片结构如下：

```
UECFOOD100_resnet/
├── train/
│   ├── 0/xxx.jpg
│   ├── 1/xxx.jpg
├── val/
│   ├── 0/xxx.jpg
│   ├── 1/xxx.jpg
```

运行训练：

```bash
python resnet_train.py
```

---

## 🔍 模型预测

### 【YOLOv5】

**本地预测（图片路径）**：

```bash
python local_test.py --source path/to/image.jpg
```

**API接口**：

```bash
uvicorn app:app --reload
# POST /predict 接口，上传图片即可
```

### 【ResNet50】

```bash
python resnet_predict.py --img path/to/image.jpg
```

---

## 📄 类别标签

`category.txt` 中提供了类别 ID 与菜品名称的映射，用于解析预测结果。

---

## ✨ TODO（未来计划）

- ✅ 支持 YOLOv5 检测训练与预测
- ✅ 支持 ResNet 图像分类
- ⏳ 网页端部署（Streamlit/FastAPI）
- ⏳ 精度优化（数据增强、多模型对比）

