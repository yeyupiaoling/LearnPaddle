# coding=utf-8
import os
import re
import random

# 根据数据集的路径和年份和数据类型拼接路径
def get_dir(devkit_dir, year, type):
    return os.path.join(devkit_dir, 'VOC' + year, type)


def walk_dir(devkit_dir, year):
    # 获取所有类别的训练和测试的文件名的文件夹路径
    filelist_dir = get_dir(devkit_dir, year, 'ImageSets/Main')
    # 获取存放标注文件的路径
    annotation_dir = get_dir(devkit_dir, year, 'Annotations')
    # 获取存放图像文件夹的路径
    img_dir = get_dir(devkit_dir, year, 'JPEGImages')
    # 训练的数据list
    trainval_list = []
    # 测试的数据list
    test_list = []
    # 用于存放数据来检查是否已存在该路径的名称
    added = set()
    # 获取ImageSets/Main下的所有文件夹
    for _, _, files in os.walk(filelist_dir):
        # 获取类别的trainval.txt和测试的test.txt
        for fname in files:
            # 清空img_ann_list
            img_ann_list = []
            # 判断是测试数据还是训练数据
            if re.match('[a-z]+_trainval\.txt', fname):
                img_ann_list = trainval_list
            elif re.match('[a-z]+_test\.txt', fname):
                img_ann_list = test_list
            else:
                continue
            # 拼接路径，获得文件的相对路径
            fpath = os.path.join(filelist_dir, fname)
            # 读取文件中的内容
            for line in open(fpath):
                # 获取文件的名称
                name_prefix = line.strip().split()[0]
                # 判断是否已经存在该名称，如果存在就直接跳过下面的添加操作
                if name_prefix in added:
                    continue
                # 添加新数据，用于下次检查
                added.add(name_prefix)
                # 根据名称获取标注文件的相对路径
                ann_path = os.path.join(annotation_dir, name_prefix + '.xml')
                # 根据名称获取图像的相对路径
                img_path = os.path.join(img_dir, name_prefix + '.jpg')
                # 检查文件是否存在
                assert os.path.isfile(ann_path), 'file %s not found.' % ann_path
                assert os.path.isfile(img_path), 'file %s not found.' % img_path
                # 生成一个图像列表
                img_ann_list.append((img_path, ann_path))

    return trainval_list, test_list


def prepare_filelist(devkit_dir, years, output_dir):
    trainval_list = []
    test_list = []
    # 获取两个年份的数据
    for year in years:
        trainval, test = walk_dir(devkit_dir, year)
        trainval_list.extend(trainval)
        test_list.extend(test)
    # 打乱训练数据
    random.shuffle(trainval_list)
    # 保存训练图像列表
    with open(os.path.join(output_dir, 'trainval.txt'), 'w') as ftrainval:
        for item in trainval_list:
            ftrainval.write(item[0] + ' ' + item[1] + '\n')
    # 保存测试图像列表
    with open(os.path.join(output_dir, 'test.txt'), 'w') as ftest:
        for item in test_list:
            ftest.write(item[0] + ' ' + item[1] + '\n')

if __name__ == '__main__':
    # 数据存放的位置
    devkit_dir = 'VOCdevkit'
    # 数据的年份
    years = ['2007', '2012']
    prepare_filelist(devkit_dir, years, '.')
