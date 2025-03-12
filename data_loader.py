from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
# from torchtoolbox.transform import Cutout

class DualTransformDataset(Dataset):
    """ 让同一张图片分别经过两种 transform 处理 """
    def __init__(self, root, teacher_transform, student_transform, train=True):
        self.dataset = datasets.ImageFolder(root=root, transform=None)  # 不使用默认 transform
        self.teacher_transform = teacher_transform
        self.student_transform = student_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]  # 读取原始图像（PIL 格式）

        img_teacher = self.teacher_transform(img)  # 教师网络的输入
        img_student = self.student_transform(img)  # 学生网络的输入

        return img_teacher, img_student, label  # 返回两种变换后的图像和标签

def load_data(type,val_ratio=0.2):
    train_path="./MSTAR/train"
    test_path="./MSTAR/test"
    # train_path="./FUSAR1/train"
    # test_path="./FUSAR1/test"

    train_transforms = transforms.Compose([
        # transforms.Resize((224, 224)),  # 调整大小
        # FrequencyTransform(),  # 先计算高低频
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),  # 插值放缩
        transforms.RandomRotation(degrees=10),  # 轻微旋转
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 亮度/对比度/饱和度/色调微调
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 轻微高斯模糊
        transforms.Grayscale(num_output_channels=3),  ##灰度图转为RGB三通道
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # Cutout()
    ])

    test_transforms = transforms.Compose([
        # transforms.Resize((224, 224)),  # 调整大小
        # FrequencyTransform(),  # 先计算高低频
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),  # 插值放缩
        transforms.RandomRotation(degrees=10),  # 轻微旋转
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 亮度/对比度/饱和度/色调微调
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 轻微高斯模糊
        transforms.Grayscale(num_output_channels=3),  ##灰度图转为RGB三通道
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # Cutout()
    ])

    train_transforms_teacher = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),  # 插值放缩
        transforms.RandomRotation(degrees=10),  # 轻微旋转
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 亮度/对比度/饱和度/色调微调
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 轻微高斯模糊
        transforms.Grayscale(num_output_channels=3),  ##灰度图转为RGB三通道
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    train_dataset = DualTransformDataset(root=train_path, teacher_transform=train_transforms_teacher,
                                     student_transform=train_transforms_teacher)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True,num_workers=0,pin_memory = True)


    val_dataset = datasets.ImageFolder(root=test_path, transform=test_transforms)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False,num_workers=0,pin_memory = True)



    if type == "train":
        return train_loader
    elif type == "val":
        return val_loader
