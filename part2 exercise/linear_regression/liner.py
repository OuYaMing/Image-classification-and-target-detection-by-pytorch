# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 19:26:15 2020

@author: 盒子先生
"""


import torch as t
from matplotlib import pyplot as plt
from IPython import display
t.manual_seed(1000)


def get_fake_data(batch_size=8):
    
    x = t.rand(batch_size,1)*20
    #y=2x+3+(随机噪声)
    y = x*2 + (1+t.randn(batch_size, 1))*3
    
    return x,y

x,y = get_fake_data()
plt.scatter(x.squeeze().numpy(),y.squeeze().numpy())

w=t.rand(1,1)
b=t.zeros(1,1)
lr = 0.001


for ii in range(3000):
    #获取y=2*x+3+(随机噪声)的数据集
    x,y = get_fake_data()
    #前向传播  y=wx+b
    y_pred = x.mm(w) + b.expand_as(y)
    #均方根误差
    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()
    #后向传播 手动计算梯度
    dloss = 1
    dy_pred = dloss * (y_pred - y)
    dw = x.t().mm(dy_pred)
    db = dy_pred.sum()
    #更新参数
    w.sub_(lr*dw)
    b.sub_(lr*db)
    if ii%500 == 0:
        #画图
        display.clear_output(wait = True)
        x = t.arange(0,20).view(-1,1)
 #       print(x.size(),w.size())
        
 
        y = t.mm(x.float(),w.float()) + b.expand_as(x)
        plt.plot(x.numpy(),y.numpy())
        
        x2, y2 = get_fake_data(20)
        plt.scatter(x2.numpy(), y2.numpy())
        
        plt.xlim(0,20)
        plt.ylim(0,41)
        plt.show()
        plt.pause(0.5)
        #打印学习到的参数
        print('第',ii+1,'次迭代')
        print('w  实际值：2   预测值',w)
        print('b  实际值：3   预测值',b)
        

        
    
    
    
    
    
    
    
