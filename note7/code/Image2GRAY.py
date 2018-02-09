# coding=utf-8
import os
from PIL import Image

def Image2GRAY(path):
    # 获取临时文件夹中的所有图像路径
    imgs = os.listdir(path)
    i = 0
    for img in imgs:
        # 每10个数据取一个作为测试数据，剩下的作为训练数据
        if i % 10 == 0:
            # 使图像灰度化并保存
            im = Image.open(path + '/' + img).convert('L')
            im = im.resize((180, 80), Image.ANTIALIAS)
            im.save('../data/test_data/' + img)
        else:
            # 使图像灰度化并保存
            im = Image.open(path + '/' + img).convert('L')
            im = im.resize((180, 80), Image.ANTIALIAS)
            im.save('../data/train_data/' + img)
        i = i + 1

if __name__ == '__main__':
    # 临时数据存放路径
    path = '../data/data_temp'
    Image2GRAY(path)