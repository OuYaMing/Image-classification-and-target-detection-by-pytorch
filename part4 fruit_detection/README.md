数据集：fruit-detection  (VOC格式)
当前训练checkpoint已删除
需重新下载vgg16预训练模型，重新训练

Step1: python creat_txt.py
Step2: python creat_data_list.py
#以下注意修改 checkpoint 的保存读取路径
Step3: python train.py
Step4: python eval.py（所有图片评估）
Step5: python detect.py(单张图片)