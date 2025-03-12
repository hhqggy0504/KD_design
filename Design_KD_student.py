import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models.quantization import resnet18
import FEM
from wtconv import WTConv2d
from SE import SE
from CBAM import CBAM

class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):        # 普通Block简单完成两次卷积操作
        super(CommonBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x                                            # 普通Block的shortcut为直连，不需要升维下采样

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)       # 完成一次卷积
        x = self.bn2(self.conv2(x))                             # 第二次卷积不加relu激活函数

        x += identity                                           # 两路相加
        return F.relu(x, inplace=True)                          # 添加激活函数输出


class SpecialBlock(nn.Module):                                  # 特殊Block完成两次卷积操作，以及一次升维下采样
    def __init__(self, in_channel, out_channel, stride):        # 注意这里的stride传入一个数组，shortcut和残差部分stride不同
        super(SpecialBlock, self).__init__()
        self.change_channel = nn.Sequential(                    # 负责升维下采样的卷积网络change_channel
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = self.change_channel(x)                       # 调用change_channel对输入修改，为后面相加做变换准备

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))                             # 完成残差部分的卷积

        x += identity
        return F.relu(x, inplace=True)                          # 输出卷积单元


class ResNet18(nn.Module):
    def __init__(self, classes_num=10):
        super(ResNet18, self).__init__()
        self.prepare = nn.Sequential(  # 所有的ResNet共有的预处理==》[batch, 64, 56, 56]
            nn.Conv2d(3, 64, kernel_size=3, stride=1),  # Depthwise
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.conv_local_11 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=1, bias=False)
        self.conv_local_33 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_local_55 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv_local_77 = nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv_local_99 = nn.Conv2d(32, 32, kernel_size=9, stride=1, padding=4, bias=False)

        self.conv_local_batchnorm=nn.BatchNorm2d(32)
        self.ReLU = nn.ReLU(inplace=True)
        self.local_pool = nn.AdaptiveAvgPool2d((224, 224))

        self.domain_11=nn.Conv2d(32,32,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv_domain = WTConv2d(32, 32, kernel_size=5, stride=1, wt_levels=3, bias=False)
        self.batchnorm_domain=nn.BatchNorm2d(32)

        self.conv_global = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=1, bias=False)

        self.layer1 = nn.Sequential(            # layer1有点特别，由于输入输出的channel均是64，故两个CommonBlock
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(            # layer234类似，由于输入输出的channel不同，故一个SpecialBlock，一个CommonBlock
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            SpecialBlock(128, 256, [2, 1]),
            CommonBlock(256, 256, 1)
        )
        self.layer4 = nn.Sequential(
            SpecialBlock(256, 512, [2, 1]),
            CommonBlock(512, 512, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))    # 卷积结束，通过一个自适应均值池化==》 [batch, 512, 1, 1]
        self.fc = nn.Sequential(                # 最后用于分类的全连接层，根据需要灵活变化
            nn.AdaptiveAvgPool1d(128),  # 额外降维，减少参数量
            nn.Linear(128, classes_num)
        )

        self.conv_local_fusion = nn.Sequential(
            nn.Conv2d(32 * 4, 32 * 4, kernel_size=3, stride=1, padding=1, groups=32 * 4, bias=False),  # 深度可分离卷积
            nn.Conv2d(32 * 4, 32, kernel_size=1, stride=1, padding=0, bias=False),  # Pointwise 1x1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 预处理部分
        x = self.prepare(x)  # 预处理

        feature1 = self.layer1(x)
        # print(f"After prepare: {x.shape}")  # 打印prepare后的维度

        domain, local = torch.split(feature1, 32, dim=1)  # 拆分通道
        # print(f"After split: domain: {domain.shape}, local: {local.shape}")  # 打印拆分后的维度

        # 对local的卷积操作
        local1 = self.conv_local_33(local)
        # print(f"After conv_local_33: {local1.shape}")

        local2 = self.conv_local_55(local)
        # print(f"After conv_local_55: {local2.shape}")

        local3 = self.conv_local_77(local)
        # print(f"After conv_local_77: {local3.shape}")

        local4 = self.conv_local_99(local)
        # print(f"After conv_local_99: {local4.shape}")

        local2_resized = F.adaptive_avg_pool2d(local2, (111, 111))
        local3_resized = F.adaptive_avg_pool2d(local3, (111, 111))
        local4_resized = F.adaptive_avg_pool2d(local4, (111, 111))

        # 先拼接
        local = torch.cat((local1, local2_resized, local3_resized, local4_resized), dim=1)

        # local = torch.cat((local1, local2, local3, local4), dim=1)

        # 使用 1×1 卷积融合通道信息
        local = self.conv_local_fusion(local)

        # 继续后续的 BatchNorm 和 ReLU
        local = self.conv_local_batchnorm(local)
        local = self.ReLU(local)

        local = CBAM(local)

        # 对domain的卷积操作
        domain = self.conv_domain(domain)
        # print(f"After conv_domain: {domain.shape}")

        domain = self.batchnorm_domain(domain)
        # print(f"After batchnorm_domain: {domain.shape}")

        domain = self.ReLU(domain)
        # print(f"After ReLU domain: {domain.shape}")
        domain = CBAM(domain)

        domain = F.adaptive_avg_pool2d(domain, (1, 1))
        domain = domain.view(domain.shape[0], -1)
        domain = domain.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, local.shape[2], local.shape[3])
        # 拼接domain和local
        x = torch.cat((domain, local), dim=1)
        # print(f"After concat: {x.shape}")
        # 添加 GAP 提取全局特征

        # 调用SE模块
        # x = SE(x)
        # print(f"After SE: {x.shape}")

        # print(x.shape)
        # x = FEM.block(x)
                 # 四个卷积单元
        feature2 = self.layer2(x)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)
        # feature4 = feature3
        x = self.pool(feature4)            # 池化
        x = x.reshape(x.shape[0], -1)   # 将x展平，输入全连接层
        x = self.fc(x)

        return x,feature1,feature2,feature3,feature4

model_resnet18=ResNet18()



