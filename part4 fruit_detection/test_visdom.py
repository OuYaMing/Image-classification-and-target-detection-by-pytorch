from visdom import Visdom
import numpy as np
import time

# 将窗口类实例化
viz = Visdom() 

# 创建窗口并初始化
viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))

for global_steps in range(10):
    # 随机获取loss值
    loss = 0.2 * np.random.randn() + 1
    # 更新窗口图像
    viz.line([loss], [global_steps], win='train_loss', update='append')
    time.sleep(0.5)