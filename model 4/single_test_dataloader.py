from torch.utils.data import Dataset, DataLoader
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
from collections import Counter
# Counter可以对字符串、列表、元祖、字典进行计数，返回一个字典类型的数据，键是元素，值是元素出现的次数。

def trans_form(img):

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    # transforms.Compose：组合多个变换。
    # transforms.ToPILImage()：将图像转换为 PIL 图像。
    # transforms.Resize((224, 224))：将图像大小调整为 224x224。
    # transforms.ToTensor()：将图像转换为张量，并且将像素值归一化到 [0, 1]。能够把灰度范围从0-255变换到0-1之间。
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])：使用给定的均值和标准差进行归一化。
    img = transform(img)
    return img


# a = []


class Test_Data(Dataset):

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

    def __getitem__(self, idx):
        person_name = self.person_path[idx // 10]
        # print('idx = ', idx)
        # print('person_name = ', person_name)
        # a.append(person_name)
        # # print('a = ',a)
        # bb = Counter(a)
        # # 使用 Counter 统计 a 中每个人出现的次数。
        # b = bb[person_name] - 1
        # # 计算当前人已经被访问的次数减去 1，得到当前样本的索引。
        # # print('bb = ',bb)
        # # print('b = ',b)

        # 获取该人的所有图像路径。
        print_imgs_path = os.listdir(os.path.join(self.print_root_dir, person_name))
        vein_imgs_path = os.listdir(os.path.join(self.vein_root_dir, person_name))
        third_imgs_path = os.listdir(os.path.join(self.third_root_dir, person_name))
        forth_imgs_path = os.listdir(os.path.join(self.forth_root_dir, person_name))
        fifth_imgs_path = os.listdir(os.path.join(self.fifth_root_dir, person_name))

        length1_imgs = len(print_imgs_path)
        length2_imgs = len(vein_imgs_path)
        length3_imgs = len(third_imgs_path)
        length4_imgs = len(forth_imgs_path)
        length5_imgs = len(fifth_imgs_path)
        # if len(a) == len(print_imgs_path):
        #     a.clear()
        #     # 如果 a 中元素的数量等于当前人的图像数量，清空 a
        # print_img_path = print_imgs_path[b]
        # # 根据计算得到的索引 b 获取当前图像文件的路径。
        sample1_index = random.sample(range(length1_imgs), 1)
        sample2_index = random.sample(range(length2_imgs), 1)
        sample3_index = random.sample(range(length3_imgs), 1)
        sample4_index = random.sample(range(length4_imgs), 1)
        sample5_index = random.sample(range(length5_imgs), 1)

        print_img_path = print_imgs_path[sample1_index[0]]
        vein_img_path = vein_imgs_path[sample2_index[0]]
        third_img_path = third_imgs_path[sample3_index[0]]
        forth_img_path = forth_imgs_path[sample4_index[0]]
        fifth_img_path = fifth_imgs_path[sample5_index[0]]

        p_img_item_path = os.path.join(self.print_root_dir, person_name, print_img_path)
        v_img_item_path = os.path.join(self.vein_root_dir, person_name, vein_img_path)
        t_img_item_path = os.path.join(self.third_root_dir, person_name, third_img_path)
        f_img_item_path = os.path.join(self.forth_root_dir, person_name, forth_img_path)
        h_img_item_path = os.path.join(self.fifth_root_dir, person_name, fifth_img_path)
        # p_img_item_path：构建图像的完整路径。得到的是根目录-人名-具体的照片名的路径。

        p_img = cv2.imread(p_img_item_path)
        p_img = torch.tensor(p_img / 255.0).to(torch.float).permute(2, 0, 1)
        # 前两行p_img：读取图像并转换为张量，进行归一化和通道维度变换。
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


    def __len__(self):
        return len(self.person_path) * 10
        # 返回数据集的总长度，人数*10。

