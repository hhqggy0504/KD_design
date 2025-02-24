import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
# def fetch_dataloader(types):

def load_data(type):
    train_path="./MSTAR/train"
    test_path="./MSTAR/test"

    train_transforms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.Grayscale(num_output_channels=3), ##灰度图转为RGB三通道
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    test_transforms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.Grayscale(num_output_channels=3), ##灰度图转为RGB三通道
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    val_dataset = datasets.ImageFolder(root=test_path, transform=test_transforms)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)


    # print(f"Number of training images: {len(train_dataset)}")
    # print(f"Number of validation images: {len(val_dataset)}")

    if type == "train":
        return train_loader
    elif type == "val":
        return val_loader
    # return train_loader, val_loader
# # 获取一个批次的数据
# data_iter = iter(train_loader)
# images, labels = next(data_iter)
#
# # 将图像从Tensor转换为NumPy数组
# images = images.numpy()
#
# # 显示前几个图像
# fig, axes = plt.subplots(1, 5, figsize=(15, 15))
# for i in range(5):
#     ax = axes[i]
#     ax.imshow(np.transpose(images[i], (1, 2, 0)))  # 转换为HWC格式
#     ax.axis('off')
#     ax.set_title(f"Label: {labels[i].item()}")
# plt.show()