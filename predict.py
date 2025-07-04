import sys
from pathlib import Path
import torch
import cv2
import numpy as np

# 修复路径
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[0]  # dish_classifier 根目录
YOLOV5_DIR = PROJECT_ROOT / 'yolov5'

# 把 yolov5 模块加入 Python 搜索路径
if str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.augmentations import letterbox

# 配置
MODEL_PATH = Path(__file__).resolve().parent / 'yolov5' / 'runs' / 'train' / 'lowmem_run2' / 'weights' / 'best.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(MODEL_PATH, device=DEVICE)
model.eval()
names = model.names

def predict_image(image_path: str):
    img0 = cv2.imread(image_path)
    assert img0 is not None, f"Image not found: {image_path}"

    img = letterbox(img0, new_shape=640)[0]
    img = img.transpose((2, 0, 1))[::-1]  # BGR -> RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(DEVICE).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    if pred is not None and len(pred):
        top_pred = pred[0]
        class_id = int(top_pred[5])
        class_name = names[class_id]
        return class_id
    else:
        return -1
