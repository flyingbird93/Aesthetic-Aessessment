# coding: utf-8

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append("..")
from utils.utils import MyDataset, validate, show_confMat
from tensorboardX import SummaryWriter
from datetime import datetime
from data.dataset import DogCat
from torchvision import models
#from torchsummary import summary



# train_txt_path = '../../Data/train.txt'
# valid_txt_path = '../../Data/valid.txt'
#
# classes_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
classes_name = ['high', 'low']

# 超参数
batch_size = 128
#valid_bs = 128
lr_init = 0.001
max_epoch = 20


# 1.数据准备
root = '../../Smith/Data/AVA/images/'
train_data = DogCat(root, train=True)
train_dataloader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4)

val_data = DogCat(root, train=False)
val_dataloader = DataLoader(val_data, batch_size, shuffle=False, num_workers=4)

test_data = DogCat(root, test=True)
test_dataloader = DataLoader(test_data, batch_size, shuffle=False, num_workers=4)

# log
result_dir = 'results/logs/'

now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir=log_dir)

# ------------------------------------ step 1/5 : 加载数据------------------------------------

# # 数据预处理设置
# normMean = [0.4948052, 0.48568845, 0.44682974]
# normStd = [0.24580306, 0.24236229, 0.2603115]
# normTransform = transforms.Normalize(normMean, normStd)
# trainTransform = transforms.Compose([
#     transforms.Resize(32),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     normTransform
# ])
#
# validTransform = transforms.Compose([
#     transforms.ToTensor(),
#     normTransform
# ])
#
# # 构建MyDataset实例
# train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)
# valid_data = MyDataset(txt_path=valid_txt_path, transform=validTransform)
#
# # 构建DataLoader
# train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
# valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs)

# ------------------------------------ step 2/5 : 定义网络------------------------------------

#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#     '''
# class AlexNet(nn.Module):
#     """
#     code from torchvision/models/alexnet.py
#     结构参考 <https://arxiv.org/abs/1404.5997>
#     """
#
#     def __init__(self, num_classes=10):
#         super(AlexNet, self).__init__()
#
#         self.model_name = 'alexnet'
#
#         self.features = nn.Sequential(
#             nn.Conv2d(16, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), 256 * 6 * 6)
#         x = self.classifier(x)
#         return x
#     '''
#
#
#     # 定义权值初始化
#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.xavier_normal_(m.weight.data)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 torch.nn.init.normal_(m.weight.data, 0, 0.01)
#                 m.bias.data.zero_()

# 2.网络模型
net = models.resnet34(pretrained=True)
net.fc = nn.Linear(512, 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
#summary(net, (3, 224, 224))


#net = Net().cuda()     # 创建一个网络
#net.initialize_weights()    # 初始化权值

# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------

criterion = nn.CrossEntropyLoss()
criterion.cuda()# 选择损失函数
optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)    # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)     # 设置学习率下降策略

# ------------------------------------ step 4/5 : 训练 --------------------------------------------------

for epoch in range(max_epoch):

    loss_sigma = 0.0    # 记录一个epoch的loss之和
    correct = 0.0
    total = 0.0
    scheduler.step()  # 更新学习率

    for i, data in enumerate(train_dataloader):
        # if i == 30 : break
        # 获取图片和标签
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        # forward, backward, update weights
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 统计预测信息
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().cpu().numpy()
        loss_sigma += loss.item()

        # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
        if i % 10 == 9:
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch + 1, max_epoch, i + 1, len(train_dataloader), loss_avg, correct / total))

            # 记录训练loss
            writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
            # 记录learning rate
            writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)
            # 记录Accuracy
            writer.add_scalars('Accuracy_group', {'train_acc': correct / total}, epoch)

    # 每个epoch，记录梯度，权值
    for name, layer in net.named_parameters():
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

    # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
    if epoch % 2 == 0:
        loss_sigma = 0.0
        cls_num = len(classes_name)
        #cls_num = 2
        conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
        net.eval()
        for i, data in enumerate(val_dataloader):

            # 获取图片和标签
            images, labels = data
            images, labels = Variable(images).cuda(), Variable(labels).cuda()

            # forward
            outputs = net(images)
            outputs.detach_()

            # 计算loss
            loss = criterion(outputs, labels)
            loss_sigma += loss.item()

            # 统计
            _, predicted = torch.max(outputs.data, 1)
            # labels = labels.data    # Variable --> tensor

            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.0

        print('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))
        # 记录Loss, accuracy
        writer.add_scalars('Loss_group', {'valid_loss': loss_sigma / len(val_dataloader)}, epoch)
        writer.add_scalars('Accuracy_group', {'valid_acc': conf_mat.trace() / conf_mat.sum()}, epoch)
print('Finished Training')


# 测试训练结果
for i, data in enumerate(test_dataloader):
    # if i == 30 : break
    loss_sigma = 0.0    # 记录一个epoch的loss之和
    correct = 0.0
    total = 0.0

    # 获取图片和标签
    inputs, labels = data
    inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

    net,eval()
    # forward, backward, update weights
    # optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    #loss.backward()
    #optimizer.step()

    # 统计预测信息
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).squeeze().sum().cpu().numpy()
    loss_sigma += loss.item()

    # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
    if i % 10 == 9:
        loss_avg = loss_sigma / 10
        loss_sigma = 0.0
        print("Testing: Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
            i + 1, len(test_dataloader), loss_avg, correct / total))

        # 记录训练loss
        writer.add_scalars('Loss_group', {'test_loss': loss_avg}, i)
        # 记录learning rate
        #writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)
        # 记录Accuracy
        writer.add_scalars('Accuracy_group', {'test_acc': correct / total}, i)


# ------------------------------------ step5: 保存模型 并且绘制混淆矩阵图 ------------------------------------
net_save_path = os.path.join(log_dir, 'net_params.pkl')
torch.save(net.state_dict(), net_save_path)

conf_mat_train, train_acc = validate(net, train_dataloader, 'train', classes_name)
conf_mat_valid, valid_acc = validate(net, val_dataloader, 'valid', classes_name)
conf_mat_test, test_acc = validate(net, test_dataloader, 'test', classes_name)

show_confMat(conf_mat_train, classes_name, 'train', log_dir)
show_confMat(conf_mat_valid, classes_name, 'valid', log_dir)
show_confMat(conf_mat_test, classes_name, 'test', log_dir)