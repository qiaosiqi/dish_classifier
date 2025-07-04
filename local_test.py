from pathlib import Path
from predict import predict_image

# 📄 修改成你的 category.txt 文件路径
CATEGORY_FILE = Path("UECFOOD100/category.txt")

def load_id_to_name(category_file: Path) -> dict:
    """
    从 category.txt 构建 id_to_name 映射
    YOLO 从 0 开始，原始 ID 从 1 开始
    """
    id_to_name = {}
    with category_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()[1:]  # 跳过第一行 header
        for i, line in enumerate(lines):
            parts = line.strip().split('\t')
            if len(parts) == 2:
                original_id, name = parts
                id_to_name[i] = {"id": int(original_id), "name": name}
    return id_to_name

def test_local_image(image_path: str):
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"[错误] 找不到图片: {image_path}")
        return

    id_to_name = load_id_to_name(CATEGORY_FILE)

    try:
        result = predict_image(str(image_path))  # 这里返回的是 class_idx 或 list of idx
        print("\n✅ 预测结果：")
        if isinstance(result, int):
            mapped = id_to_name.get(result, {"id": -1, "name": "未知"})
            print(f"预测类索引：{result}")
            print(f"菜品 ID：{mapped['id']}，名称：{mapped['name']}")
        elif isinstance(result, list):
            for idx in result:
                mapped = id_to_name.get(idx, {"id": -1, "name": "未知"})
                print(f"- 类索引：{idx} → 菜品 ID：{mapped['id']}，名称：{mapped['name']}")
        else:
            print("[警告] 返回格式未知：", result)
    except Exception as e:
        print(f"[异常] 预测失败: {e}")

if __name__ == "__main__":
    # ❗局长可以修改这里的图片路径测试
    test_image_path = "D:\\CODE\\myGitHub\\dish_classifier\\UECFOOD100\\1\\1.jpg"  # 举例：test_images 文件夹下的图片
    test_local_image(test_image_path)


