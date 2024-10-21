# attention:该文件每次运行前line97:data_train,line124:data_test,line281:torch.save,lin287:model_path都需要更改，
# 且训练结束后所有的权重文件需要保存至"F:\Mobilenet\file"下.注意epoch数
from tqdm import tqdm
# tqdm是一个快速、可扩展的Python进度条，可以在Python长循环中添加一个进度提示信息，用户只需要封装任意的迭代器tqdm(iterator)。
# 它可以帮助我们监测程序运行的进度，估计运行的时长，甚至可以协助debug。
from single_train_dataloader import MyData
from torch.utils.data import Dataset, DataLoader
import sys
# Python的sys库是一种内建模块，可对Python的运行环境进行访问和操作。

from single_test_dataloader import Test_Data
import torch
from torch import nn, optim
# torch.nn 是pytorch 的一种API工具，该工具是为了快速搭建神经网络任务而创建的。
# optim：优化器
# 导入各种模型：
from torchvision.models.vgg import vgg16

from torchvision.models.resnet import resnet18
from model_MobileNet_n import MobleNetV_five

from torchvision.models.alexnet import AlexNet
from torchvision.models.densenet import DenseNet
from torchvision.models.efficientnet import efficientnet_b0 as create_model

# 设置分类类别数：
Len_class = 100
add_epoch = True


# 主函数：
def main():
    # 设置设备为GPU或者CPU：
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_name = "vgg16"
    # 初始化 MobileNet V1 模型并将其移动到设备上，532是所需训练的人数（文件夹数）
    net = MobleNetV_five(532)
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))
    # if add_epoch:
    #     weights_path = "CASIA_Naive_MobileNet_fingervein&knuckle_best.pth"
    #     net.load_state_dict(torch.load(weights_path, map_location=device))
    # net = AlexNet(600)

    # dict_trained = torch.load("resnet18.pth")
    # dict_trained = torch.load("vgg16.pth")

    # net.load_state_dict(dict_trained, strict=False)




    # inchannel = net.fc.in_features
    # net.fc = nn.Linear(inchannel, 100)



    # net = MobileNetV2(num_classes=Len_class)  # Casia=100
    # net = AlexNet(num_classes=Len_class, init_weights=True)
    # net = Densenet(num_classes=Len_class)
    # net = create_model(num_classes=Len_class).to(device)

# 将模型移动到指定设备：
    net.to(device)

# 打印当前使用的设备信息，帮助用户确认模型正在使用哪个设备进行计算：
    print("using {} device.".format(device))

    # lr = 0.000001  # lr = 0.0001  lr = 0.0005  lr = 0.0008
    # 定义损失函数为交叉熵损失：
    celoss = nn.CrossEntropyLoss()
    # 交叉熵主要用于度量同一个随机变量X的预测分布Q与真实分布P之间的差距。

    # 设置学习率：
    learning_rate = 1e-3
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.005)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.00001)

# 使用Adam 优化器：
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001)  # jingdu tiaocan 0.0001->0.001
    # Adam 优化算法是随机梯度下降算法的扩展式
    # Adam的参数配置：
    # alpha：同样也称为学习率或步长因子，它控制了权重的更新比率（如 0.001）。
    # 较大的值（如 0.3）在学习率更新前会有更快的初始学习，而较小的值（如 1.0E-5）会令训练收敛到更好的性能。
    # beta1：一阶矩估计的指数衰减率（如 0.9）。
    # beta2：二阶矩估计的指数衰减率（如 0.999）。该超参数在稀疏梯度（如在 NLP 或计算机视觉任务中）中应该设置为接近 1 的数。
    # epsilon：该参数是非常小的数，其为了防止在实现中除以零（如 10E-8）。
    # weight_decay 是权重衰减，通常也叫做L2正则化。
    # 它的作用是防止过拟合，通过在损失函数中加入权重的平方和的一部分来限制模型的复杂度。具体来说，0.001 是权重衰减的系数，即正则化强度。

    # optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.001)  # jingdu tiaocan 0.0001->0.001


    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95, )
    # 设置训练的epoch 数和 batch size：
    epochs = 50
    batch_size = 4
    # batch_size 指定了每个批次的数据量。
    # data_train = MyData('../our/c-Palm-print-train/')
    # data_train = MyData('../Casia/print1-train/', '../Casia/print2-train/', '../Casia/vein1-train/')
    # data_train = MyData('D:/D_study/A_data/Casia/print1-train/', 'D:/D_study/A_data/Casia/print2-train/', 'D:/D_study/A_data/Casia/vein1-train/')
    # data_train = MyData('D:/D_study/A_data/Casia_2modality/print-train/')
    # data_train = MyData('D:/D_study/A_data/Qinghua-Multimodal/print-train')
    data_train = MyData("../../hand-multi-dataset/palmvein_train/",
                           "../../hand-multi-dataset/palmprint_train/",
                           "../../hand-multi-dataset/print_train/",
                           "../../hand-multi-dataset/knuckle_train/",
                           "../../hand-multi-dataset/fingervein_train/")



    data_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True,
        # 设定 shuffle=True 表示在每个 epoch 开始之前将数据集打乱。
        # 这种打乱操作有助于提升模型的泛化能力，防止模型在训练过程中记住数据的顺序。
        drop_last=False,
        # 当数据集中的数据量不能被 batch_size 整除时，drop_last 参数控制是否丢弃最后一个不完整的批次。
        # 如果 drop_last=False，则保留最后一个不完整的批次；
        # 如果 drop_last=True，则丢弃最后一个不完整的批次。
        num_workers=0)
    # num_workers 指定了用来加载数据的子进程数量。
    # num_workers=0 表示数据加载将在主进程中进行，而不是使用子进程。


    # torch.utils.data.DataLoader 主要是对数据进行 batch 的划分。

    # data_test = Test_Data('../our/c-Palm-print-test/')

    # data_test = Test_Data('../our/c-Palm-print-test/', '../our/palm-vein-test/', '../our/Dorsal-Vein-test/')
    # data_test = Test_Data('../Casia/print1-test/', '../Casia/print2-test/', '../Casia/vein1-test/')
    # data_test = Test_Data('D:/D_study/A_data/Casia/print1-test/', 'D:/D_study/A_data/Casia/print2-test/', 'D:/D_study/A_data/Casia/vein1-test/')
    # data_test = Test_Data('D:/D_study/A_data/Qinghua-Multimodal/print-test/')
    data_test = Test_Data("../../hand-multi-dataset/palmvein_test/",
                           "../../hand-multi-dataset/palmprint_test/",
                           "../../hand-multi-dataset/print_test/",
                           "../../hand-multi-dataset/knuckle_test/",
                           "../../hand-multi-dataset/fingervein_test/")

    test_loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)

# 打印训练和测试数据集的大小：
    best_acc = 0
    best_batch = 0
    print('len_data_train = ',len(data_train))
    print('len_data_test = ',len(data_test))
    print('len_data_loader = ',len(data_loader))
    print('len_test_loader = ',len(test_loader))


# 训练和验证循环：
    for epoch in range(0, epochs):
        # 20代表起始epoch
        # train
        net.train()
        # 进入训练模式
        acc = 0.0  # accumulate accurate number / epoch
        num = 0
        running_loss = 0.0
        # 初始化准确率和损失
        train_bar = tqdm(data_loader, file=sys.stdout)
        # 使用 tqdm 创建训练进度条
        for step, data in enumerate(train_bar):
            # 对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
            # enumerate多用于在for循环中得到计数
            # step 是当前批次的索引，data 包含当前批次的数据。
            p_imgs, v_imgs, t_imgs, f_imgs, h_imgs, person_name = data
            # 从 data 中解包得到图像数据 p_imgs 和对应的标签 person_name
            p_imgs = p_imgs.to(device)
            v_imgs = v_imgs.to(device)
            t_imgs = t_imgs.to(device)
            f_imgs = f_imgs.to(device)
            h_imgs = h_imgs.to(device)
            # 将图像数据 p_imgs 转移到指定的计算设备（CPU 或 GPU）

            person_labels = [int(_) - 1 for _ in person_name]
            # 将标签列表 person_name 转换为整数列表并减去 1，因为标签通常是从 1 开始计数的，这里将其转换为从 0 开始计数。
            person_labels = torch.tensor(person_labels).to(device)
            # 将标签列表转换为 PyTorch 张量并转移到指定的计算设备

            optimizer.zero_grad()
            # 将梯度归零
            # 为了避免梯度累积，每次计算梯度之前都需要将其清零。
            outputs = net(p_imgs, v_imgs, t_imgs, f_imgs, h_imgs)
            # 将图像数据输入到模型 net 中，得到模型的输出 outputs
            # print('outputs.shape = ',outputs.shape)
            #######################################################
            predict_y0 = torch.max(outputs, dim=1)[1]
            # 从模型的输出中得到预测的标签
            # dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
            # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
            # [1]：
            # 这个索引操作提取元组中的第二个元素，即 indices
            # indices 张量包含每行最大值的索引，即每个样本的预测类别索引
            # print('predict_y0.shape = ',predict_y0.shape)

            ################################### TCP ####################################
            person_labels = torch.tensor(person_labels).to(device)
            # 重新将标签转化为张量并转移到计算设备上。这一步实际上是多余的，因为在157已经做过这个操作。

            # print(predict_y0,person_labels)
            acc += torch.eq(predict_y0, person_labels.to(device)).sum().item()
            # torch.eq 是 PyTorch 的一个函数，用于逐元素比较两个张量。如果元素相等，则返回 1，否则返回 0
            # predict_y0 是模型预测的类别索引，形状为 [batch_size]
            # person_labels 是实际的标签，也是一个形状为 [batch_size] 的张量
            # person_labels.to(device) 将实际标签移动到与 predict_y0 相同的设备（如 GPU），确保它们在同一设备上进行比较
            # 这个操作返回一个布尔张量（即 torch.bool 类型），每个元素表示预测是否与实际标签匹配
            # sum() 方法将布尔张量中的 True 值加起来（在 PyTorch 中，True 相当于 1，False 相当于 0），得到预测正确的样本数
            # item() 方法将单个值的张量转换为 Python 标量。它通常用于从张量中提取单个值
            # 将当前批次中正确预测的样本数累加到 acc 变量中。acc 通常在训练或验证循环的开始初始化为 0

            loss = celoss(outputs, person_labels.to(device))
            # 使用交叉熵损失函数计算模型输出与真实标签之间的损失


            #############################
            optimizer.zero_grad()
            # 再次清除梯度
            loss.backward()
            # 反向传播，计算得到每个参数的梯度值
            optimizer.step()
            # 使用优化器更新模型参数，通过梯度下降执行一步参数更新
            running_loss += loss.item()
            # 将当前批次的损失累加到 running_loss 中
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} ".format(epoch + 1,
                                                                      epochs,
                                                                      running_loss / (step + 1))
        # 更新进度条的描述信息，显示当前 epoch 的编号和平均损失
        num = len(data_loader) * batch_size
        # 计算一个 epoch 中所有样本的数量
        accurate = acc / num
        # 计算当前 epoch 的准确率。acc 是正确预测的数量，num 是总样本数
        print('  num:{},train_accuracy:{:.3f},acc:{}'.format(num, accurate, acc))
        # 打印当前 epoch 的结果
        # if (epoch + 1) % 5 == 0:
        #     torch.save(net.state_dict(), 'ckpt_best_%s.pth' % (str(epoch + 1)))
        print()

        # validate
        # 验证循环：
        net.eval()
        # 将模型设置为评估模式。这会影响某些层的行为，例如 Dropout（正则化） 和 BatchNorm（标准化），使其在评估过程中使用评估模式

        acc = 0.0  # accumulate accurate number / epoch
        # 初始化准确预测的数量
        test_bar = tqdm(test_loader, file=sys.stdout)
        # # 对可以迭代的对象使用tqdm进行封装实现可视化进度。
        # sys.stdout是python中的标准输出流，默认是映射到控制台的，即将信息打印到控制台。
        # 禁用梯度计算。在验证过程中不需要计算梯度，可以节省内存和计算资源：
        with torch.no_grad():
            # 逐批次遍历验证数据加载器中的数据：
            for step, data in enumerate(test_bar):
                p_imgs, v_imgs, t_imgs, f_imgs, h_imgs, person_name = data
                p_imgs = p_imgs.to(device)
                v_imgs = v_imgs.to(device)
                t_imgs = t_imgs.to(device)
                f_imgs = f_imgs.to(device)
                h_imgs = h_imgs.to(device)

                person_labels = [int(_) - 1 for _ in person_name]
                person_labels = torch.tensor(person_labels).to(device)
                # 将标签列表转换为 PyTorch 张量并转移到指定的计算设备

                outputs = net(p_imgs, v_imgs, t_imgs, f_imgs, h_imgs)
                #######################################################
                predict_y0 = torch.max(outputs, dim=1)[1]
                # [1]：
                # 这个索引操作提取元组中的第二个元素，即 indices；第0维是batch
                # indices 张量包含每行最大值的索引，即每个样本的预测类别索引
                # 该行目的在于挑出预测概率最大的类别

                ################################### TCP ####################################
                person_labels = torch.tensor(person_labels).to(device)

                acc += torch.eq(predict_y0, person_labels.to(device)).sum().item()

                #############################
            num = len(test_loader) * batch_size
            accurate = acc / num

            print('  num:{},test_accuracy:{:.3f},acc:{}'.format(num, accurate, acc))


# # 如果当前准确率高于历史最好准确率：
#         if best_acc < accurate:
#             best_acc = accurate
#             best_batch = epoch+1
#         print('best_acc = ', best_acc)
#         print('best_batch = ', best_batch)
#
#         torch.save(net.state_dict(), 'CUMT_Naive_MobileNet_knuckle.pth')
#         # 原先是MobileNet变AlexNet
#         print('Finished Training')

        if best_acc < accurate:
            best_acc = accurate
            best_batch = epoch+1
            torch.save(net.state_dict(), 'CASIA_Naive_MobileNet_five_model_best.pth')
            print('Finished Training')
        print('best_acc = ', best_acc)
        print('best_batch = ', best_batch)

        if epoch > 5 and (epoch + 1) % 3 == 0:
            model_path = 'CASIA_Naive_MobileNet_five_model_%s.pth' % str(epoch + 1)
            torch.save(net.state_dict(), model_path)
            # f.write('Saved model to {}\n'.format(model_path))

        torch.save(net.state_dict(), 'CASIA_Naive_MobileNet_five_model_last.pth')

        # torch.save(net.state_dict(), 'CUMT_Single_vgg16_Dorsal_98.76.pth')

# 主函数调用：
if __name__ == '__main__':
    main()
# 调用 main 函数开始训练和验证过程
