import os
import shutil
import random


def split_images(root_dir, train_ratio=0.5):
    train_dir = os.path.join(root_dir, "train")
    test_dir = os.path.join(root_dir, "test")

    # 确保目标文件夹存在
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 遍历每个类别文件夹
    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        if not os.path.isdir(category_path):  # 跳过非文件夹
            continue

        # 目标类别文件夹
        train_category_dir = os.path.join(train_dir, category)
        test_category_dir = os.path.join(test_dir, category)
        os.makedirs(train_category_dir, exist_ok=True)
        os.makedirs(test_category_dir, exist_ok=True)

        # 获取所有图片文件
        images = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg','.tiff'))]
        random.shuffle(images)  # 随机打乱

        # 切分为 train 和 test
        split_idx = int(len(images) * train_ratio)
        train_images, test_images = images[:split_idx], images[split_idx:]

        # 复制文件到 train 和 test 目录
        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(train_category_dir, img))
        for img in test_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(test_category_dir, img))

    print("数据集拆分完成！")


# 使用示例
split_images("./FUSAR1")
