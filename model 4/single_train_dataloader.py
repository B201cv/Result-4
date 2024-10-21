# GPT + CSDN
from torch.utils.data import Dataset, DataLoader
# 从模块torch.utils.data导入工具Dataset
# Dataset: 定义了抽象的数据集类，用户可以通过继承该类来构建自己的数据集。
# Dataset 类提供了两个必须实现的方法：__getitem__ 用于访问单个样本，__len__ 用于返回数据集的大小。
# DataLoader: 数据加载器类，用于批量加载数据集。
# 它接受一个数据集对象作为输入，并提供多种数据加载和预处理的功能，如设置批量大小、多线程数据加载和数据打乱等。
from PIL import Image
# 导入 PIL 库中的 Image 模块
# PIL 是 Python Imaging Library 的缩写，它是 Python 中用于图像处理的一个强大的库。
# 而 Image 模块则是 PIL 库中的一个子模块，提供了处理图像的各种功能。
import torch
import numpy as np
import cv2
# cv2：OpenCV 库，用于图像处理
import random
import os
# os：用于处理文件和目录
import torchvision.transforms as transforms
# torchvision.transforms：PyTorch 的图像变换模块。
# import my_transforms

# 定义图像转换函数：
def trans_form(img):

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    img = transform(img)
    # transforms.Compose：组合多个变换。
    # transforms.ToPILImage()：将图像转换为 PIL 图像。
    # transforms.Resize((224, 224))：将图像大小调整为 224x224。
    # transforms.ToTensor()：将图像转换为张量，并且将像素值归一化到 [0, 1]。能够把灰度范围从0-255变换到0-1之间。
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])：使用给定的均值和标准差进行归一化。
    # img = img.unsqueeze(0)
    return img

# # 初始化空列表a，用于记录已经采样过的图像索引，以避免重复采样：
# a = []

# 自定义数据集类：
class MyData(Dataset):
    def __init__(self, print_root_dir, vein_root_dir, third_root_dir, forth_root_dir, fifth_root_dir, training=True):
        # print_root_dir：数据集的根目录。
        self.print_root_dir = print_root_dir
        self.vein_root_dir = vein_root_dir
        self.third_root_dir = third_root_dir
        self.forth_root_dir = forth_root_dir
        self.fifth_root_dir = fifth_root_dir
        self.person_path = os.listdir(self.print_root_dir)
        # self.person_path：根目录下所有子目录（每个子目录代表一个人）的路径列表。
        # os.listdir()：列举的当前文件下的所有文件。

    # 获取数据集的一个样本：
    def __getitem__(self, idx):
        person_name = self.person_path[idx // 10]
        # person_name：根据索引获取对应的人的子目录名。

        print_imgs_path = os.listdir(os.path.join(self.print_root_dir, person_name))
        vein_imgs_path = os.listdir(os.path.join(self.vein_root_dir, person_name))
        third_imgs_path = os.listdir(os.path.join(self.third_root_dir, person_name))
        forth_imgs_path = os.listdir(os.path.join(self.forth_root_dir, person_name))
        fifth_imgs_path = os.listdir(os.path.join(self.fifth_root_dir, person_name))
        # print_imgs_path：获取该人的所有图像路径。
        # os.path.join()：作用：根据你的操作系统使用正确的路径分隔符
        # 语法：os.path.join(path, *paths)
        # 参数：
        #     path： 表示文件系统路径的类路径对象
        #     *path：表示文件系统路径的类路径对象。
        #            它表示要连接的路径组件，类路径对象是表示路径的字符串或字节对象。


        length1_imgs = len(print_imgs_path)
        length2_imgs = len(vein_imgs_path)
        length3_imgs = len(third_imgs_path)
        length4_imgs = len(forth_imgs_path)
        length5_imgs = len(fifth_imgs_path)
        # length1_imgs：该人的图像数量。

        sample1_index = random.sample(range(length1_imgs), 1)
        sample2_index = random.sample(range(length2_imgs), 1)
        sample3_index = random.sample(range(length3_imgs), 1)
        sample4_index = random.sample(range(length4_imgs), 1)
        sample5_index = random.sample(range(length5_imgs), 1)
        # sample1_index：随机选择一个图像索引，并确保不会与之前选择的索引重复。
        # random.sample 是 random 模块中的一个函数，用于从一个序列中随机抽取指定数量的不重复元素。
        # 其签名为 random.sample(population, k)，其中 population 是要抽取元素的序列，k 是要抽取的元素数量。


        # for item in a:
        #     if sample1_index == item:
        #         sample1_index = random.sample(range(length1_imgs), 1)
        # # a是记录已经采样过的图像索引，该段是避免重复采样。

        # a.append(sample1_index)
        # # 将选出的非重复图像索引加入a，以防下次被选中。



        print_img_path = print_imgs_path[sample1_index[0]]
        vein_img_path = vein_imgs_path[sample2_index[0]]
        third_img_path = third_imgs_path[sample3_index[0]]
        forth_img_path = forth_imgs_path[sample4_index[0]]
        fifth_img_path = fifth_imgs_path[sample5_index[0]]
        # sample1_index[0] 是在访问 sample1_index 列表的第一个元素（索引为 0）。
        # 由于 random.sample(range(length1_imgs), 1) 返回一个包含一个元素的列表，
        # 所以 sample1_index[0] 是该列表中的唯一元素，即一个整数索引。
        # print_imgs_path[sample1_index[0]] 使用从 sample1_index 中获取的索引来访问 print_imgs_path 列表中的对应元素。
        # print_imgs_path在上面得到的是一个表


        p_img_item_path = os.path.join(self.print_root_dir, person_name, print_img_path)
        v_img_item_path = os.path.join(self.vein_root_dir, person_name, vein_img_path)
        t_img_item_path = os.path.join(self.third_root_dir, person_name, third_img_path)
        f_img_item_path = os.path.join(self.forth_root_dir, person_name, forth_img_path)
        h_img_item_path = os.path.join(self.fifth_root_dir, person_name, fifth_img_path)
        # p_img_item_path：构建图像的完整路径。得到的是根目录-人名-具体的照片名的路径

        # p_img：读取图像并转换为张量，进行归一化和通道维度变换。
        p_img = cv2.imread(p_img_item_path)
        # 函数 cv2.imread() 用于从指定的文件读取图像。
        p_img = torch.tensor(p_img / 255.0).to(torch.float).permute(2, 0, 1)
        # p_img / 255.0：
        # 这一部分将图像像素值从 [0, 255] 范围归一化到 [0, 1] 范围。
        # 假设 p_img 是一个 numpy 数组或其他图像格式的数据，该操作将每个像素值除以 255.0，从而完成归一化。
        #
        # torch.tensor(p_img / 255.0)：
        # 这个函数将归一化后的图像数据转换为 PyTorch 的张量 (torch.Tensor)。
        # torch.tensor 是 PyTorch 中用于从数据（如列表、numpy 数组等）创建张量的方法。
        #
        # .to(torch.float)：
        # .to(torch.float) 将张量的数据类型转换为 float32。
        # 在 PyTorch 中，许多深度学习模型都需要输入的数据类型为浮点数（float32），以便进行后续的计算。
        #
        # .permute(2, 0, 1)
        # .permute 函数用于重新排列张量的维度顺序。
        # 图像数据通常以 (高度, 宽度, 通道) 或 (H, W, C) 的顺序存储，但 PyTorch 通常期望图像数据以 (通道, 高度, 宽度) 或 (C, H, W) 的顺序。
        # .permute(2, 0, 1) 将维度顺序从 (H, W, C) 改变为 (C, H, W)。
        # 具体来说，这里的 2, 0, 1 表示将原来的第2个维度（通道）移到第1个位置，将第0个维度（高度）移到第2个位置，将第1个维度（宽度）移到第3个位置。
        p_img = trans_form(p_img)
        # trans_form(p_img)：对图像进行预处理（调整大小、转换为张量和归一化）。
        v_img = cv2.imread(v_img_item_path)
        v_img = torch.tensor(v_img / 255.0).to(torch.float).permute(2, 0, 1)
        v_img = trans_form(v_img)

        t_img = cv2.imread(t_img_item_path)
        t_img = torch.tensor(t_img / 255.0).to(torch.float).permute(2, 0, 1)
        t_img = trans_form(t_img)

        f_img = cv2.imread(f_img_item_path)
        f_img = torch.tensor(f_img / 255.0).to(torch.float).permute(2, 0, 1)
        f_img = trans_form(f_img)

        h_img = cv2.imread(h_img_item_path)
        h_img = torch.tensor(h_img / 255.0).to(torch.float).permute(2, 0, 1)
        h_img = trans_form(h_img)

        return p_img, v_img, t_img, f_img, h_img, person_name
        # 返回预处理后的图像和对应的人的名称。


    # 获取数据集的长度：
    def __len__(self):
        return len(self.person_path) * 10
        # 返回数据集的总长度，人数*10
