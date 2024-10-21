# 导入库：
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import RandomAffine, ToTensor


# torch.nn 是 PyTorch 中定义神经网络层和其他相关功能的模块。
# torch 是 PyTorch 的核心库。


# 定义MobleNetV1类：
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class MobleNetV1(nn.Module):
    # 定义类的构造函数：
    def __init__(self):
        # ch_in 表示输入通道数。
        # n_classes 表示分类的类别数。
        super(MobleNetV1, self).__init__()
        # super(MobileNetV1, self).__init__()
        # 调用 nn 的__init__()方法
        # 调用父类的构造函数以正确初始化基类 nn.Module。
        # 定义主网络结构 self.model：
        features = []
        features.extend([
            self._conv_st(3, 32, 2),
            self._conv_dw(32, 64, 1),
            self._conv_dw(64, 128, 2),
            self._conv_dw(128, 128, 1),
            self._conv_dw(128, 256, 2),
            self._conv_dw(256, 256, 1),
            self._conv_dw(256, 512, 2),
            self._conv_x5(512, 512, 5),
            self._conv_dw(512, 1024, 2),
            self._conv_dw(1024, 1024, 1),
            nn.AvgPool2d(kernel_size=7, stride=1)
        ])
        self.features1 = nn.Sequential(*features)
    # 定义前向传播函数 forward：
    def forward(self, x):

        x = self.features1(x)

        return x

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
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
                      bias=False),
            # 定义了深度卷积层，输入和输出通道数相同，卷积核大小为 3x3，步幅为 stride，填充为 1，分组数等于输入通道数（实现深度卷积）。
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            # 定义了逐点卷积层，卷积核大小为 1x1，步幅为 1，填充为 0，不使用偏置。
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

class Fusion_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        # block即为BasicBlock模型，blocks_num可控制传入的Bottleneck

        super(Fusion_2, self).__init__()  # 可见ResNet也是nn.Module的子�?
        self.W = nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=1,
                           stride=1)
        self.conv1_1 = nn.Conv2d(in_channels=1024*5, out_channels=1024, kernel_size=1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x, y, z, a, b):

        x = F.normalize(x, p=2, dim=1, eps=1e-12)
        y = F.normalize(y, p=2, dim=1, eps=1e-12)
        z = F.normalize(z, p=2, dim=1, eps=1e-12)
        a = F.normalize(a, p=2, dim=1, eps=1e-12)
        b = F.normalize(b, p=2, dim=1, eps=1e-12)


        fx = x
        fy = y
        fz = z
        fa = a
        fb = b

        # 定义仿射变换范围
        degrees = 360  # 旋转角度范围  180
        translate = (0.1, 0.1)  # 平移范围 (fraction of image size) #表示图像在水平方向上可以被随机平移最多图像宽度的50%，在垂直方向上可以被随机平移最多图像高度的80%。
        scale = (0.8, 1.2)  # 缩放范围  0.8-1.2   # 表示图像的缩放比例可以在0.2到1.2之间变化。这意味着图像可以缩小到原来的20%或放大到原来的120%。

        # 创建随机仿射变换对象
        affine_transform = RandomAffine(degrees=degrees, translate=translate, scale=scale)

        # 应用仿射变换
        transformed_x = affine_transform(x)
        transformed_y = affine_transform(y)
        transformed_z = affine_transform(z)
        transformed_a = affine_transform(a)
        transformed_b = affine_transform(b)

        transformed_x = F.normalize(transformed_x, p=2, dim=1, eps=1e-12)
        transformed_y = F.normalize(transformed_y, p=2, dim=1, eps=1e-12)
        transformed_z = F.normalize(transformed_z, p=2, dim=1, eps=1e-12)
        transformed_a = F.normalize(transformed_a, p=2, dim=1, eps=1e-12)
        transformed_b = F.normalize(transformed_b, p=2, dim=1, eps=1e-12)

        # print('fx.shape,transformed_x.shape = ',fx.shape,transformed_x.shape)  # torch.Size([4, 128, 28, 28])
        b, c, h, w = fx.size()

        mult_x = torch.matmul(fx, transformed_x) / h
        mult_y = torch.matmul(fy, transformed_y) / h
        mult_z = torch.matmul(fz, transformed_z) / h
        mult_a = torch.matmul(fa, transformed_a) / h
        mult_b = torch.matmul(fb, transformed_b) / h

        mult_x = F.softmax(mult_x, dim=1)  # 这将在通道维度上应用 softmax，使得每个像素位置上的通道值和为1。
        mult_y = F.softmax(mult_y, dim=1)
        mult_z = F.softmax(mult_z, dim=1)
        mult_a = F.softmax(mult_a, dim=1)
        mult_b = F.softmax(mult_b, dim=1)

        # print('torch.matmul(fx, transformed_x).shape = ',torch.matmul(fx, transformed_x))  # torch.Size([4, 128, 28, 28])
        # print('mult_x.shape = ',mult_x)  # torch.Size([4, 128, 28, 28])

        # 加一个动态1*1卷积
        mult_x = self.W(mult_x)
        mult_y = self.W(mult_y)
        mult_z = self.W(mult_z)
        mult_a = self.W(mult_a)
        mult_b = self.W(mult_b)


        # mult_x = F.normalize(mult_x, p=2, dim=1, eps=1e-12)
        # mult_y = F.normalize(mult_y, p=2, dim=1, eps=1e-12)

        mult_x = mult_x + fx
        mult_y = mult_y + fy
        mult_z = mult_z + fz
        mult_a = mult_a + fa
        mult_b = mult_b + fb

        fusion_2 = torch.cat((mult_x, mult_y, mult_z, mult_a, mult_b), 1)
        fusion_2 = self.conv1_1(fusion_2)
        fusion_2 = self.dropout(fusion_2)

        prelu = nn.PReLU()
        prelu.to(device)
        fusion_2 = prelu(fusion_2)

        return fusion_2
