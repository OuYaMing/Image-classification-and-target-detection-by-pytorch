# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:29:59 2020

@author: 盒子先生
"""




from __future__ import print_function, division
import time
import torch as t
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms, datasets
import torchvision as tv

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
num_workers = 0  # 多线程的数目
model = 'model.pt'   # 把训练好的模型保存下来



# 对加载的图像作归一化处理， 并裁剪为[224x224x3]大小的图像
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
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






class ResidualBlock(nn.Module):
    '''
    实现子module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        #Sequential 是一个特殊Module, 包含几个子module,前向传播时会将输入一层一层的传递下去
        self.left = nn.Sequential(
                #卷积层
                nn.Conv2d(inchannel,outchannel,3,stride, 1,bias=False),
                #在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，
                #这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
                nn.BatchNorm2d(outchannel),
                #激活函数采用ReLU
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
                nn.BatchNorm2d(outchannel) )
        self.right = shortcut
        
    #前向传播
    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class ResNet(nn.Module):
    '''
    实现主module：ResNet34
    ResNet34 包含多个layer，每个layer又包含多个residual block
    用子module来实现residual block，用_make_layer函数来实现layer
    '''
    def __init__(self, num_classes=6):
        super(ResNet, self).__init__()
        # 前几层图像转换
        self.pre = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                #最大池化
                nn.MaxPool2d(3, 2, 1))
        
        # 重复的layer，分别有3，4，6，3个residual block
        #共四层
        self.layer1 = self._make_layer( 64, 64, 3)
        self.layer2 = self._make_layer( 64, 128, 4, stride=2)
        self.layer3 = self._make_layer( 128, 256, 6, stride=2)
        self.layer4 = self._make_layer( 256, 512, 3, stride=2)

        #分类用的全连接
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self,  inchannel, outchannel, block_num, stride=1):
        
        #构建layer,包含多个residual block
        shortcut = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,1,stride, bias=False),
                nn.BatchNorm2d(outchannel))
        
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.pre(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #平均池化
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train():

    net = ResNet(6)
#    print(net)
    
    if use_gpu:
        net = net.cuda()
    
    
    
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
                #print('测试集中的准确率为: %d %%' % (100 * correct / total))
                
                
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
                #print('测试集中的准确率为: %d %%' % (100 * train_correct / train_total))
                
                
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
                
              #  print('[%d, %5d] loss: %.3f  ' \
               #       % (epoch+1, i+1, running_loss / 50)  )
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
