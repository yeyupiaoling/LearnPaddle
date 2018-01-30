# coding=utf-8
import os
import uuid

from PIL import Image


class YanZhenMaUtil():
    def __init__(self):
        pass

    def splitimage(self,src, dstpath):
        # 分割路径,并获得文件名
        name = src.split('/')
        name1 = name[name.__len__() - 1]
        name2 = name1.split('.')[0]
        # 加载四个文字的名字
        l1 = list(name2)
        img = Image.open(src)
        # 按照四张图片的大小裁剪
        box1 = (5, 0, 17, 27)
        box2 = (17, 0, 29, 27)
        box3 = (29, 0, 41, 27)
        box4 = (41, 0, 53, 27)
        # 为每一张图片提供自己的文件夹
        path1 = dstpath + '/%s' % l1[0]
        path2 = dstpath + '/%s' % l1[1]
        path3 = dstpath + '/%s' % l1[2]
        path4 = dstpath + '/%s' % l1[3]
        # 创建对应的文件夹
        if not os.path.exists(path1):
            os.makedirs(path1)
        if not os.path.exists(path2):
            os.makedirs(path2)
        if not os.path.exists(path3):
            os.makedirs(path3)
        if not os.path.exists(path4):
            os.makedirs(path4)
        # 裁剪图片并保存
        img.crop(box1).resize((36, 36), Image.ANTIALIAS).save(path1 + '/%s.png' % uuid.uuid1())
        img.crop(box2).resize((36, 36), Image.ANTIALIAS).save(path2 + '/%s.png' % uuid.uuid1())
        img.crop(box3).resize((36, 36), Image.ANTIALIAS).save(path3 + '/%s.png' % uuid.uuid1())
        img.crop(box4).resize((36, 36), Image.ANTIALIAS).save(path4 + '/%s.png' % uuid.uuid1())


if __name__ == '__main__':
    # 原图片路径
    root_path = '../images/src_yanzhengma/'
    # 裁剪后图片的路径
    dstpath = '../images/dst_yanzhengma/'
    # 获取所以图片
    imgs = os.listdir(root_path)
    yanZhenMaUtil = YanZhenMaUtil()
    # 开始裁剪
    for src in imgs:
        src = root_path + src
        yanZhenMaUtil.splitimage(src=src, dstpath=dstpath)
