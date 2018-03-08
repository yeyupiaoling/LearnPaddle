# coding=utf-8
import os

def rename(images_dir):
    # 获取所有图像
    images = os.listdir(images_dir)
    i = 1
    for image in images:
        src_name = images_dir + image
        # 以六位数字命名，符合VOC数据集格式
        name = '%06d.jpg' % i
        dst_name = images_dir + name
        os.rename(src_name,dst_name)
        i += 1
    print '重命名完成'

if __name__ == '__main__':
    # 要重命名的文件所在的路径
    images_dir = '../data/plate_number/images/'
    rename(images_dir)