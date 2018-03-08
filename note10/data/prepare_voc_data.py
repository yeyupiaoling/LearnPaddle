# coding=utf-8
import os
import random


def prepare_filelist(images_path, annotation_path, output_dir):
    # 训练数据list
    trainval_list = []
    # 测试数据list
    test_list = []
    # 获取所以图像
    all_images = os.listdir(images_path)
    # 当前数据量
    data_num = 1
    for images in all_images:
        trainval = []
        test = []
        if data_num % 10 == 0:
            # 没10张图像取一个做测试集
            name = images.split('.')[0]
            annotation = os.path.join(annotation_path, name + '.xml')
            # 如果该图像的标注文件不存在，就不添加到图像列表中
            if not os.path.exists(annotation):
                continue
            test.append(os.path.join(images_path, images))
            test.append(annotation)
            # 添加到总的测试数据中
            test_list.append(test)
        else:
            # 其他的的图像做训练数据集
            name = images.split('.')[0]
            annotation = os.path.join(annotation_path, name + '.xml')
            # 如果该图像的标注文件不存在，就不添加到图像列表中
            if not os.path.exists(annotation):
                continue
            trainval.append(os.path.join(images_path, images))
            trainval.append(annotation)
            # 添加到总的训练数据中
            trainval_list.append(trainval)
        data_num += 1

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
    # 图像所存在的路径
    images_path = 'plate_number/images'
    # 标注文件所存在的路径
    annotation_path = 'plate_number/annotation'
    # 输出图像列表的路径
    output_dir = './'
    prepare_filelist(images_path, annotation_path, output_dir)
