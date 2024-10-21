# 导入库：
import torch
import torch.nn as nn
from model import MobleNetV1,Fusion_2
# torch.nn 是 PyTorch 中定义神经网络层和其他相关功能的模块。
# torch 是 PyTorch 的核心库。


# 定义MobleNetV1类：
class MobleNetV_five(nn.Module):
    # 定义类的构造函数：
    def __init__(self, num_classes):
        # ch_in 表示输入通道数。
        # n_classes 表示分类的类别数。
        super(MobleNetV_five, self).__init__()
        # super(MobileNetV1, self).__init__()
        # 调用 nn 的__init__()方法
        # 调用父类的构造函数以正确初始化基类 nn.Module。
        # 定义主网络结构 self.model：

        self.features1 = MobleNetV1()
        self.features2 = MobleNetV1()
        self.features3 = MobleNetV1()
        self.features4 = MobleNetV1()
        self.features5 = MobleNetV1()
        self.Fusion_2 = Fusion_2(in_channels=1024, out_channels=1024)
        # self.conv1 = self._conv_st(3, 32, 2)
        # # 第一层是一个标准卷积层，输入通道为 3，输出通道为 32，步幅为 2。
        # # 随后的层是深度可分离卷积块（使用 conv_dw），依次定义了不同输入和输出通道数以及步幅的卷积层。
        # self.conv_dw1 = self._conv_dw(32, 64, 1)
        # self.conv_dw2 = self._conv_dw(64, 128, 2)
        # self.conv_dw3 = self._conv_dw(128, 128, 1)
        # self.conv_dw4 = self._conv_dw(128, 256, 2)
        # self.conv_dw5 = self._conv_dw(256, 256, 1)
        # self.conv_dw6 = self._conv_dw(256, 512, 2)
        # self.conv_dw_x5 = self._conv_x5(512, 512, 5)
        # self.conv_dw7 = self._conv_dw(512, 1024, 2)
        # self.conv_dw8 = self._conv_dw(1024, 1024, 1)
        # self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        # # 自适应平均池化到 1x1 大小。
        # 定义全连接层 self.fc：
        self.fc = nn.Linear(1024, num_classes)
        # nn.Linear(1024, n_classes) 定义了一个全连接层，输入维度为 1024，输出维度为 n_classes，对应分类的类别数。
        # classifier = []
        # classifier.extend([
        #     nn.Linear(5120, 2560),
        #     nn.ReLU(),
        #     nn.Linear(2560, 1280),
        #     nn.ReLU(),
        #     nn.Linear(1280, num_classes)
        # ])
        # self.classifier = nn.Sequential(*classifier)
        self.dropout = nn.Dropout(0.3)

# 定义前向传播函数 forward：
    def forward(self, x, y, z, a, b):
        # x = self.conv1(x)
        # x = self.conv_dw1(x)
        # x = self.conv_dw2(x)
        # x = self.conv_dw3(x)
        # x = self.conv_dw4(x)
        # x = self.conv_dw5(x)
        # x = self.conv_dw6(x)
        # x = self.conv_dw_x5(x)
        # x = self.conv_dw7(x)
        # x = self.conv_dw8(x)
        # x = self.avgpool(x)
        # x = self.dropout1(x)
        #
        # y = self.conv1(y)
        # y = self.conv_dw1(y)
        # y = self.conv_dw2(y)
        # y = self.conv_dw3(y)
        # y = self.conv_dw4(y)
        # y = self.conv_dw5(y)
        # y = self.conv_dw6(y)
        # y = self.conv_dw_x5(y)
        # y = self.conv_dw7(y)
        # y = self.conv_dw8(y)
        # y = self.avgpool(y)
        # y = self.dropout1(y)
        #
        # z = self.conv1(z)
        # z = self.conv_dw1(z)
        # z = self.conv_dw2(z)
        # z = self.conv_dw3(z)
        # z = self.conv_dw4(z)
        # z = self.conv_dw5(z)
        # z = self.conv_dw6(z)
        # z = self.conv_dw_x5(z)
        # z = self.conv_dw7(z)
        # z = self.conv_dw8(z)
        # z = self.avgpool(z)
        # z = self.dropout1(z)
        #
        # a = self.conv1(a)
        # a = self.conv_dw1(a)
        # a = self.conv_dw2(a)
        # a = self.conv_dw3(a)
        # a = self.conv_dw4(a)
        # a = self.conv_dw5(a)
        # a = self.conv_dw6(a)
        # a = self.conv_dw_x5(a)
        # a = self.conv_dw7(a)
        # a = self.conv_dw8(a)
        # a = self.avgpool(a)
        # a = self.dropout1(a)
        #
        # b = self.conv1(b)
        # b = self.conv_dw1(b)
        # b = self.conv_dw2(b)
        # b = self.conv_dw3(b)
        # b = self.conv_dw4(b)
        # b = self.conv_dw5(b)
        # b = self.conv_dw6(b)
        # b = self.conv_dw_x5(b)
        # b = self.conv_dw7(b)
        # b = self.conv_dw8(b)
        # b = self.avgpool(b)
        # b = self.dropout1(b)

        # x = self.fc(x) 将展平的输出通过全连接层进行分类。
        # y = torch.softmax(x)
        # M = torch.cat([x, y, z], dim=1)  # (n_img, n_stc, total_paths)
        # M = x + y
        x = self.features1(x)
        # x = self.dropout1(x)
        y = self.features2(y)
        # y = self.dropout1(y)
        z = self.features3(z)
        # z = self.dropout1(z)
        a = self.features4(a)
        # a = self.dropout1(a)
        b = self.features5(b)
        # b = self.dropout1(b)
        X2 = self.Fusion_2(x, y, z, a, b)
        # M = torch.cat([x, y, z, a, b], dim=1)
        M = torch.flatten(X2, start_dim=1)
        M = self.dropout(M)
        M = self.fc(M)
        # M = self.dropout(M)

        return M

# 生成一系列卷积层：https://blog.csdn.net/guoqingru0311/article/details/134112455中2.MobileNet V1网络结构下图1的五层dw卷积
    def _conv_x5(self, in_channel, out_channel, blocks):
        # in_channel：输入通道数。
        # out_channel：输出通道数。
        # blocks：要生成的卷积块的数量。
        layers = []
        # 初始化一个空列表 layers，用于存储卷积层。
        for i in range(blocks):
            # 使用 for 循环迭代 blocks 次，生成多个卷积层。
            layers.append(self._conv_dw(in_channel, out_channel, 1))
            # self._conv_dw(in_channel, out_channel, 1)：
            # 调用类中的 _conv_dw 方法生成一个深度可分离卷积层（Depthwise Separable Convolution Layer），
            # 输入通道数为 in_channel，输出通道数为 out_channel，步幅为 1。
            # 每次生成的卷积层都会被添加到 layers 列表中。
        return nn.Sequential(*layers)
    # 使用 nn.Sequential 将列表 layers 中的所有层串联成一个顺序容器（Sequential Container），并返回该容器。
    # *layers 表示将列表中的所有元素作为参数传递给 nn.Sequential。

# 定义辅助函数 conv_st：
    def _conv_st(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    # conv_st 定义了一个包含卷积、批归一化（Batch Normalization）和 ReLU 激活的组合层。
    # nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False) 定义了一个 2D 卷积层，
    # 输入通道数为 in_channels，输出通道数为 out_channels，卷积核大小为 3x3，步幅为 stride，填充为 1，不使用偏置。
    # nn.BatchNorm2d(oup) 添加批归一化层。
    # nn.ReLU() 添加 ReLU 激活函数，
    # inplace=False（默认）：这是默认设置，表示不进行就地操作，激活函数会创建一个新的张量来保存结果，而不是直接修改输入张量。

# 定义辅助函数 conv_dw：
    def _conv_dw(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            # 定义了深度卷积层，输入和输出通道数相同，卷积核大小为 3x3，步幅为 stride，填充为 1，分组数等于输入通道数（实现深度卷积）。
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            # 定义了逐点卷积层，卷积核大小为 1x1，步幅为 1，填充为 0，不使用偏置。
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

