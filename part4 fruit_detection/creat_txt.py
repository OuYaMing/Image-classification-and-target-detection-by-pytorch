import os
import random

trainval_percent = 0.8
train_percent = 0.7
#xmlfilepath = './underwater_dataset/train_data/Annotations'
#txtsavepath = './underwater_dataset/train_data/ImageSets/Main'

xmlfilepath = './fruit-detection/Annotations'
txtsavepath = './fruit-detection/ImageSets/Main'

total_xml = os.listdir(xmlfilepath)

num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval= random.sample(list,tv)
train=random.sample(trainval,tr)

#ftrainval = open('./underwater_dataset/train_data/ImageSets/Main/trainval.txt', 'w')
#ftest = open('./underwater_dataset/train_data/ImageSets/Main/test.txt', 'w')
#ftrain = open('./underwater_dataset/train_data/ImageSets/Main/train.txt', 'w')
#fval = open('./underwater_dataset/train_data/ImageSets/Main/val.txt', 'w')

'''
ftrainval = open('./dataset/VOC2007/ImageSets/Main/trainval.txt', 'w')
ftest = open('./dataset/VOC2007/ImageSets/Main/test.txt', 'w')
ftrain = open('./dataset/VOC2007/ImageSets/Main/train.txt', 'w')
fval = open('./dataset/VOC2007/ImageSets/Main/val.txt', 'w')
'''
ftrainval = open('./fruit-detection/ImageSets/Main/trainval.txt', 'w')
ftest = open('./fruit-detection/ImageSets/Main/test.txt', 'w')
ftrain = open('./fruit-detection/ImageSets/Main/train.txt', 'w')
fval = open('./fruit-detection/ImageSets/Main/val.txt', 'w')

for i  in list:
    name=total_xml[i][:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest .close()