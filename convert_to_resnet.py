import os
import shutil
import random

# 配置路径
DATA_ROOT = 'D:\\CODE\\myGithub\\dish_classifier\\UECFOOD100'
OUTPUT_ROOT = 'D:\\CODE\\myGithub\\dish_classifier\\UECFOOD100_resnet'
CATEGORY_FILE = os.path.join(DATA_ROOT, 'D:\\CODE\\myGithub\\dish_classifier\\UECFOOD100\\category.txt')

# 创建目录
os.makedirs(os.path.join(OUTPUT_ROOT, 'train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, 'val'), exist_ok=True)

# 读取类别映射 id -> name
id2name = {}
with open(CATEGORY_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()[1:]  # 跳过第一行表头
    for line in lines:
        id_, name = line.strip().split('\t')
        # 将名称中的空格换成下划线方便文件夹命名
        safe_name = name.replace(' ', '_').replace("'", "").replace('-', '_')
        id2name[id_] = safe_name

# 拆分比例
train_ratio = 0.8

for id_, cname in id2name.items():
    src_dir = os.path.join(DATA_ROOT, id_)
    train_dir = os.path.join(OUTPUT_ROOT, 'train', cname)
    val_dir = os.path.join(OUTPUT_ROOT, 'val', cname)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    imgs = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    random.shuffle(imgs)
    split_idx = int(len(imgs) * train_ratio)
    train_imgs = imgs[:split_idx]
    val_imgs = imgs[split_idx:]

    for img in train_imgs:
        shutil.copy(os.path.join(src_dir, img), os.path.join(train_dir, img))
    for img in val_imgs:
        shutil.copy(os.path.join(src_dir, img), os.path.join(val_dir, img))

print("数据集拆分完成！")
