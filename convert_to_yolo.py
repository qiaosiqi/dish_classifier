#This file converts origin UECFOOD100 data struction to UECFOOD100_YOLO data struction
import os
import shutil
import random
from pathlib import Path

# 原始数据路径
ORIG_DATASET = "D:\\CODE\\myGitHub\\dish_classifier\\UECFOOD100"
OUTPUT_DIR = "D:\\CODE\\myGitHub\\dish_classifier\\UECFOOD100_YOLO"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
LBL_DIR = os.path.join(OUTPUT_DIR, "labels")


def load_category_mapping(txt_path):
    id_to_name = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]  # skip header
        for line in lines:
            class_id, class_name = line.strip().split("\t")
            id_to_name[int(class_id)] = class_name
    return id_to_name


def prepare_dirs():
    for split in ['train', 'val']:
        os.makedirs(os.path.join(IMG_DIR, split), exist_ok=True)
        os.makedirs(os.path.join(LBL_DIR, split), exist_ok=True)


def create_yaml_file(class_dict):
    with open(os.path.join(OUTPUT_DIR, "uec_food.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {OUTPUT_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_dict)}\n")
        names = [f'"{name}"' for _, name in sorted(class_dict.items())]
        f.write(f"names: [{', '.join(names)}]\n")


def convert_dataset(id_to_name):
    for class_id in range(1, 101):
        folder = os.path.join(ORIG_DATASET, str(class_id))
        if not os.path.isdir(folder):
            continue
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(image_files)
        split_idx = int(0.8 * len(image_files))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

        for split, file_list in [('train', train_files), ('val', val_files)]:
            for file_name in file_list:
                src_path = os.path.join(folder, file_name)
                dst_img_path = os.path.join(IMG_DIR, split, f"{class_id}_{file_name}")
                dst_lbl_path = os.path.join(LBL_DIR, split, f"{class_id}_{Path(file_name).stem}.txt")

                shutil.copy(src_path, dst_img_path)

                # NOTE：此处仅生成“类别ID+无框”，因为原始数据没有标注框
                with open(dst_lbl_path, "w", encoding="utf-8") as f:
                    f.write(f"{class_id - 1} 0.5 0.5 1.0 1.0\n")  # 伪造全图为目标


if __name__ == "__main__":
    category_txt = os.path.join(ORIG_DATASET, "category.txt")
    if not os.path.exists(category_txt):
        raise FileNotFoundError("category.txt 文件不存在，请确认路径正确")

    class_map = load_category_mapping(category_txt)
    prepare_dirs()
    convert_dataset(class_map)
    create_yaml_file(class_map)
    print("✅ 数据集转换完成，输出路径：UECFOOD100_YOLO/")
