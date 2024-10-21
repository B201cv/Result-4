# attention:该文件每次运行前line103:data_test,line127:weight_path都需要更改。

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from numpy import *
# 引入numpy库中的所有函数、函数、对象、变量等等。
from torch.utils.data import Dataset
import torch
import os
import sys
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
# torchvision.transforms：PyTorch 的图像变换模块。
from tqdm import tqdm
import cv2
# cv2：OpenCV 库，用于图像处理
from torchvision.models.resnet import resnet18
from model_MobileNet_n import MobleNetV_five
from torchvision.models.alexnet import AlexNet
from torchvision.models.vgg import vgg16
from collections import Counter
from single_train_dataloader import MyData

lenth_Casia = 3  # Casia测试集每类3张图片
lenth_CUMT = 5  # CUMT测试集每类5张图片


# 整个测试过程的主函数:
def testmodel():
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
        # 对图像进行预处理（调整大小、转换为张量和归一化）。
        # img = img.unsqueeze(0)
        return img

    a = []

    class TestData(Dataset):
        def __init__(self, print_root_dir, vein_root_dir, third_root_dir, forth_root_dir, fifth_root_dir, training=True):
            self.print_root_dir = print_root_dir
            self.vein_root_dir = vein_root_dir
            self.third_root_dir = third_root_dir
            self.forth_root_dir = forth_root_dir
            self.fifth_root_dir = fifth_root_dir
            self.person_path = os.listdir(self.print_root_dir)

        def __getitem__(self, idx):
            person_name = self.person_path[idx // 10]  # 原本5
            # print()
            # print('self.person_path = ',self.person_path)
            # print('idx = ', idx)

            # print('person_name = ', person_name)
            a.append(person_name)
            # print('a = ', a)

            bb = Counter(a)
            b = bb[person_name] - 1
            print_imgs_path = os.listdir(os.path.join(self.print_root_dir, person_name))
            vein_imgs_path = os.listdir(os.path.join(self.vein_root_dir, person_name))
            third_imgs_path = os.listdir(os.path.join(self.third_root_dir, person_name))
            forth_imgs_path = os.listdir(os.path.join(self.forth_root_dir, person_name))
            fifth_imgs_path = os.listdir(os.path.join(self.fifth_root_dir, person_name))
            # print('bb = ', bb)
            # print('print_imgs_path',print_imgs_path)
            # # print('b = ', b)
            # print()
            # print()

            length1_imgs = len(print_imgs_path)
            length2_imgs = len(vein_imgs_path)
            length3_imgs = len(third_imgs_path)
            length4_imgs = len(forth_imgs_path)
            length5_imgs = len(fifth_imgs_path)

            if len(a) == len(print_imgs_path):
                a.clear()

            print_img_path = print_imgs_path[b]
            vein_img_path = vein_imgs_path[b]
            third_img_path = third_imgs_path[b]
            forth_img_path = forth_imgs_path[b]
            fifth_img_path = fifth_imgs_path[b]

            p_img_item_path = os.path.join(self.print_root_dir, person_name, print_img_path)
            v_img_item_path = os.path.join(self.vein_root_dir, person_name, vein_img_path)
            t_img_item_path = os.path.join(self.third_root_dir, person_name, third_img_path)
            f_img_item_path = os.path.join(self.forth_root_dir, person_name, forth_img_path)
            h_img_item_path = os.path.join(self.fifth_root_dir, person_name, fifth_img_path)

            # 图像处理，读取图像并转换为张量，进行归一化和通道维度变换。
            p_img = cv2.imread(p_img_item_path)
            p_img = torch.tensor(p_img / 255.0).to(torch.float).permute(2, 0, 1)
            p_img = trans_form(p_img)

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
            # 对图像进行预处理（调整大小、转换为张量和归一化）。
            return p_img, v_img, t_img, f_img, h_img, person_name

        def __len__(self):
            return len(self.person_path) * 10  # 原本5

    # 设置设备为GPU或者CPU：
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # data_test = TestData('../our/c-Palm-print-test/', '../our/palm-vein-test/', '../our/Dorsal-Vein-test/')
    # data_test = TestData('D:/D_study/A_data/Qinghua-Multimodal/vein-test/')
    # data_test = TestData('D:/D_study/A_data/Casia_2modality/print-test/')

    # data_test = TestData('D:/D_study/A_data/TJ-Multimodal/vein-test/')
    # data_test = TestData('F:/photo_data/hand-multi-dataset_gather_and split_31_20240613/split/fingervein_test_processed',
    #                      'F:/photo_data/hand-multi-dataset_gather_and split_31_20240613/split/knuckle_test')
    # data_test = TestData(
    #     'F:/photo_data/hand-multi-dataset_gather_and split_31_20240613/split/fingervein_test_processed',
    #     'F:/photo_data/hand-multi-dataset_gather_and split_31_20240613/split/palmprint_test',
    #     'F:/photo_data/hand-multi-dataset_gather_and split_31_20240613/split/palmvein_test',
    #     'F:/photo_data/hand-multi-dataset_gather_and split_31_20240613/split/knuckle_test',
    #     'F:/photo_data/hand-multi-dataset_gather_and split_31_20240613/split/print_test')

    data_test = MyData(
        'F:/photo_data/hand-multi-dataset_gather_and split_31_20240613/split/fingervein_train_processed',
        'F:/photo_data/hand-multi-dataset_gather_and split_31_20240613/split/palmprint_train',
        'F:/photo_data/hand-multi-dataset_gather_and split_31_20240613/split/palmvein_train',
        'F:/photo_data/hand-multi-dataset_gather_and split_31_20240613/split/knuckle_train',
        'F:/photo_data/hand-multi-dataset_gather_and split_31_20240613/split/print_train')



    # data_test = TestData('../casia/print-test-roi/', '../casia/vein-test2/')
    # data_test = TestData('../print-test/', '../vein-test/')
    # loader = DataLoader(data_test)
    batch_size = 10
    loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=batch_size,
        shuffle=False,
        # 不打乱数据
        drop_last=False,
        # 保留最后一个不完整的批次
        num_workers=0)
    # num_workers 指定了用来加载数据的子进程数量。
    # num_workers=0 表示数据加载将在主进程中进行，而不是使用子进程。
    print("data_loader = ", loader)
    print("start test......")
    model_name = "MobileNet"
    # model = Vgg16()
    # model = resnet18()
    model = MobleNetV_five(532)

    weights_path = "CASIA_Naive_MobileNet_five_model_best.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    # torch.load(weights_path, map_location=device)：从 weights_path 加载模型权重。
    # map_location=device 确保权重加载到指定的设备上（如 CPU 或 GPU）。
    model.eval()
    # 将模型设置为评估模式。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("using {} device.".format(device))

    # 测试模型准确性：
    accurate = 0
    arr = []
    # 空列表，储存每个epoch的准确率
    for epoch in range(1):
        acc = 0.0  # accumulate accurate number / epoch
        num = 0
        running_loss = 0.0
        # 禁用梯度计算：
        with torch.no_grad():
            bar = tqdm(loader, file=sys.stdout)
            # 对可以迭代的对象使用tqdm进行封装实现可视化进度。
            # sys.stdout是python中的标准输出流，默认是映射到控制台的，即将信息打印到控制台。
            for data_test in bar:
                p_imgs, v_imgs, t_imgs, f_imgs, h_imgs, person_name = data_test
                p_imgs = p_imgs.to(device)
                v_imgs = v_imgs.to(device)
                t_imgs = t_imgs.to(device)
                f_imgs = f_imgs.to(device)
                h_imgs = h_imgs.to(device)

                outputs = model(p_imgs, v_imgs, t_imgs, f_imgs, h_imgs)
                predict_y = torch.max(outputs, dim=1)[1]
                person_labels = [int(_) - 1 for _ in person_name]
                # for _ in person_name：
                # 这部分遍历 person_name 列表中的每个元素。

                # int(_) - 1：
                # int(_) 将当前元素 _（一个字符串）转换为整数。

                # _：这是列表推导式中的循环变量，用于遍历 person_name 列表中的每个元素。
                # 在每次迭代中，_ 表示 person_name 列表的当前元素。
                person_labels = torch.tensor(person_labels)
                acc += torch.eq(predict_y, person_labels.to(device)).sum().item()
                num = len(loader) * batch_size
            accurate = acc / num
            arr.append(accurate)
            print('[epoch %d] ' % (epoch + 1))
            print('  num:{},test_accuracy:{:.3f},acc:{}'.format(num, accurate, acc))
        # 打印关于epoch编号、总预测数（num）、准确率（accurate）和正确预测总数（acc）的信息。
    # accurate += accurate
    # ave = accurate / 40
    # 计算准确率的平均值和标准差：
    ave = mean(arr)
    std = np.std(arr)
    print('ave = ', ave)
    print('std = ', std)


#     # 在testmodel函数中初始化用于收集数据的列表
#     y_true = []
#     y_scores = []
#
#     # 修改测试循环，收集模型的预测分数和真实标签
#     # 注意：这里需要根据您的实际输出调整下面的代码
#     bar = tqdm(loader, file=sys.stdout)
#     for data_test in bar:
#         p_imgs, person_name = data_test
#         p_imgs = p_imgs.to(device)
#
#         outputs = model(p_imgs)
#         scores = torch.softmax(outputs, dim=1)  # 假设使用softmax得到每个类别的预测概率
#         scores = scores.detach().cpu().numpy()   # 将tensor转换为numpy数组
#         # 假设person_labels是转换为数值的标签列表
#         person_labels = [int(_) - 1 for _ in person_name]
#         y_true.extend(person_labels)
#         # 假设您想要收集属于某个特定类别的概率作为分数
#         y_scores.extend(scores)
#     y_true_one_hot = np.eye(100)[y_true]
#     eers = []
#     for i in range(100):  # 针对每个类别计算EER
#         fpr, tpr, thresholds = roc_curve(y_true_one_hot[:, i], np.array(y_scores)[:, i])
#         eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
#         eers.append(eer)
#
#     mean_eer = np.mean(eers)
#     print(f"Mean EER: {mean_eer}")
#
# # 输出每个类别的EER和对应的阈值
#     for i, (eer, threshold) in enumerate(zip(eers, thresholds)):
#         print(f"Class {i + 1}: EER = {eer:.4f}, Threshold = {threshold:.4f}")


if __name__ == "__main__":
    testmodel()
