# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 20:12:20 2020

@author: 盒子先生
"""


from __future__ import print_function, division
import time
import os
import numpy as np
import torch as t

import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

import torch.nn as nn
import torch.nn.functional as F
from torch import optim

use_gpu = t.cuda.is_available()
if use_gpu == True:
    print('gpu可用')
else:
    print('gpu不可用')



epochs = 50  # 训练次数
batch_size = 6  # 批处理大小
num_workers = 0 # 多线程的数目
model = 'model.pt'   # 把训练好的模型保存下来



# 对加载的图像作归一化处理， 全部改为[32x32x3]大小的图像
data_transform = transforms.Compose([
    transforms.Resize(32),
#    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 在训练集中，shuffle必须设置为True，表示次序是随机的
trainset = datasets.ImageFolder(root='datasets/train/', transform=data_transform)
trainloader = t.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# 在测试集中，shuffle必须设置为False，表示每个样本都只出现一次
testset = datasets.ImageFolder(root='datasets/test/', transform=data_transform)
testloader = t.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


classes = ('cardboard', 'glass', 'metal', 'paper','plastic','trash')



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #卷积
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        #池化
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #卷积
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        #全连接
        self.fc1 = nn.Linear(in_features=16 * 5 * 5,out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=6)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        #改变张量维度
        x = x.view(-1, 16 * 5 * 5)             
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
  '''  
#error cnn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.conv2 = nn.Conv2d(6, 16, 5)  
        self.fc1   = nn.Linear(16*5*5, 120)  
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x): 
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) 
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) 
        x = x.view(x.size()[0], -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        
        return x
'''
'''
#模型可视化
model = Net()
x = t.autograd.Variable(t.randn(1, 3, 32, 32))
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/CNN_error')
writer.add_graph(model, x)
'''
def train():

    net = Net()
    if use_gpu:
      net = net.cuda()
    
    print("开始训练")
    
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    loss_count = []
    test_accuracy_count = []
    train_accuracy_count = []
    diff_count = []
    correct = 0 # 预测正确的图片数
    total = 0 # 总共的图片数
    
    train_correct = 0
    train_total = 0
    
    
    t.set_num_threads(8)
    start = time.time()
    for epoch in range(epochs):  
        
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
    
      #      print('interations:',i)
            # 输入数据
            inputs, labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            # 梯度清零
            optimizer.zero_grad()
            
            # forward + backward 
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()   
            
            # 更新参数 
            optimizer.step()
            
            # 打印log信息
            # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
            running_loss += loss.item()
            if i % 50 == 49: # 每50个batch打印一下训练状态
                
                loss_count.append(running_loss / 50)
                
                
                for data in testloader:
                    images, labels = data
                    if use_gpu:
                        images, labels = Variable(images.cuda()), Variable(labels.cuda())
                    else:
                        images, labels = Variable(images), Variable(labels)
                    outputs = net(images)
                    _, predicted = t.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
               
                
                
                for train_data in trainloader:
                    images, labels = train_data
                    if use_gpu:
                        images, labels = Variable(images.cuda()), Variable(labels.cuda())
                    else:
                        images, labels = Variable(images), Variable(labels)
                    outputs = net(images)
                    _, predicted = t.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum()
               
                
                test_accuracy_count.append((100 * correct / total))
                train_accuracy_count.append((100 * train_correct / train_total))
                Diff = (100 * train_correct / train_total) - (100 * correct / total)
                diff_count.append(Diff)
                
                print('[%d, %5d] loss: %.3f  test_accuracy:%.3f  train_accuracy:%.3f' \
                      % (epoch+1, i+1, running_loss / 50, 100 * correct / total,100 * train_correct / train_total ) )  
                
                correct = 0
                total = 0
                train_correct = 0
                train_total = 0
                
                #print('[%d, %5d] loss: %.3f  ' \
                  #    % (epoch+1, i+1, running_loss / 50)  )
                running_loss = 0.0
                
    t.save(net, model)
    end = time.time()
    print("训练完毕！总耗时：%d 秒" % (end - start))           
                
    plt.figure(1)            
    plt.figure('CNN_Loss')
    plt.plot(loss_count,label='Loss')
    plt.legend()
    plt.show()
    
    
    plt.figure(2)            
    plt.figure('CNN_Aest_Accuracy')
    plt.plot(test_accuracy_count,label='Test_Accuracy')
    plt.legend()
    plt.show()
    
    plt.figure(3)            
    plt.figure('CNN_Train_Accuracy')
    plt.plot(train_accuracy_count,label='Train_Accuracy')
    plt.legend()
    plt.show()

    plt.figure(4)            
    plt.figure('CNN_Diff_Accuracy')
    plt.plot(diff_count,label='Diff_Accuracy')
    plt.legend()
    plt.show()
    



    t.save(net, model)

def test():
    
    correct = 0 # 预测正确的图片数
    total = 0 # 总共的图片数
    print("开始检测")
    net = t.load(model)
    net.eval()
    
    # 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
    with t.no_grad():
        for data in testloader:
            images, labels = data
            if use_gpu:
                images, labels = Variable(images.cuda()), Variable(labels.cuda())
            else:
                images, labels = Variable(images), Variable(labels)
            outputs = net(images)
            _, predicted = t.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    
    print('测试集中的准确率为: %d %%' % (100 * correct / total))


train()
test()
