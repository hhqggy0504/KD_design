import torch
import numpy as np
import torch.nn as nn
from torch.optim import optimizer

from teacher_net import model_ResNet50


def count_parameters(model):
    """
    计算模型的参数量
    :param model: 神经网络模型
    :return: 模型的参数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=4)
        self.conv2 = nn.Conv2d(48, 128, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(128, 192, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(192, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 512)  # 输入维度需要根据前面的层计算
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.pool1(x)
        x = nn.ReLU()(self.conv2(x))
        x = self.pool2(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))
        x = nn.ReLU()(self.conv5(x))
        x = self.pool3(x)
        # print(x.shape)
        x = torch.flatten(x, 1)  # 展平池化层的输出
        # print(x.shape)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = nn.LogSoftmax(dim=1)(self.fc3(x))
        return x

model2=ConvNet2()

class MyLenet5_stu(nn.Module):
    def __init__(self):
        super(MyLenet5_stu, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(3, 8, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(8, 60, kernel_size=5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(21660, 42)
        self.fc2 = nn.Linear(42, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
model_Lenet5=MyLenet5_stu()
# teacher_params_count = count_parameters(model_Lenet5)
# print(f"教师网络的参数量为: {teacher_params_count}")

class AlexNet_stu(nn.Module):
    def __init__(self):
        super(AlexNet_stu, self).__init__()
        # 定义卷积层和池化层
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 自适应层，将上一层的数据转换成6x6大小
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(4608, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
model_AlexNet=AlexNet_stu()
# teacher_params_count = count_parameters(model_AlexNet)
# print(f"教师网络的参数量为: {teacher_params_count}")

# 定义 VGG16 模型
class Vgg16_net_stu(nn.Module):
    def __init__(self):
        super(Vgg16_net_stu, self).__init__()


        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1), #(32-3+2)/1+1=32   32*32*64
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1), #(32-3+2)/1+1=32    32*32*64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)   #(32-2)/2+1=16         16*16*64
        )


        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),  #(16-3+2)/1+1=16  16*16*128
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1), #(16-3+2)/1+1=16   16*16*128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)    #(16-2)/2+1=8     8*8*128
        )

        self.layer3=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)     #(8-2)/2+1=4      4*4*256
        )

        self.layer4=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),  #(4-3+2)/1+1=4    4*4*512
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)    #(4-2)/2+1=2     2*2*512
        )

        self.layer5=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),   #(2-3+2)/1+1=2    2*2*512
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2     2*2*512
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2      2*2*512
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)   #(2-2)/2+1=1      1*1*512
        )


        self.conv=nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )


        # 自适应平均池化层，将特征图大小调整为 7x7
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 10)
        )


    def forward(self,x):
        x=self.conv(x)
        #这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成512列
        # 那不确定的地方就可以写成-1
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model_VGG16=Vgg16_net_stu()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 将卷积层的通道数调整为原来的一半
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, (out_channels // 2) * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d((out_channels // 2) * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        # 将初始输入通道数调整为原来的一半
        self.in_channels = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 调整每个层的输出通道数为原来的一半
        self.layer1 = self._make_layer(32, 3)
        self.layer2 = self._make_layer(64, 4, stride=2)
        self.layer3 = self._make_layer(128, 6, stride=2)
        self.layer4 = self._make_layer(256, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 调整全连接层的输入通道数为原来的一半
        self.fc = nn.Linear(128 * Bottleneck.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != (out_channels // 2) * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, (out_channels // 2) * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d((out_channels // 2) * Bottleneck.expansion)
            )

        layers = []
        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample))
        self.in_channels = (out_channels // 2) * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

model_ResNet50=ResNet50()
student_params_count = count_parameters(model_ResNet50)
print(f"学生网络的参数量为: {student_params_count}")
# torch.save(model.state_dict(),'./model_params/student_model_params.pt')
optimizer_stu=torch.optim.Adam(model_ResNet50.parameters(), lr=1e-3)
