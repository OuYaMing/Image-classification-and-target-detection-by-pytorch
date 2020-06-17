# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:12:26 2020

@author: 盒子先生
"""


from torch import nn
import torch as t
from torch.nn import functional as F
from torchvision import models


#分析resnet模型，可以提取出重复的block，增强代码的复用性
#考虑到Residual block和layer出现了多次，我们可以把它们实现为一个子Module或函数。这里将Residual block实现为一个子moduke，
#而将layer实现为一个函数。
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
    def __init__(self, num_classes=1000):
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
    
#使用tensorboard可视化网络模型
#自己搭的深度残差网络   
my_model = ResNet()
#pytorch封装的现成模型
torch_model = models.resnet34()
print(my_model)
x = t.autograd.Variable(t.randn(1, 3, 224, 224))
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/Resnet')
writer.add_graph(my_model, x)
input = t.autograd.Variable(t.randn(3, 3, 224, 224))
o = torch_model(input)
print(o)
    

    