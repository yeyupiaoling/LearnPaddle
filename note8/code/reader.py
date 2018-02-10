# coding=utf-8
import os
import cv2

from paddle.v2.image import load_image


class DataGenerator(object):
    def __init__(self, char_dict, image_shape):
        '''
        :param char_dict: 标签的字典类
        :type char_dict: class
        :param image_shape: 图像的固定形状
        :type image_shape: tuple
        '''
        self.image_shape = image_shape
        self.char_dict = char_dict

    def train_reader(self, file_list):
        '''
        训练读取数据
        :param file_list: 用预训练的图像列表，包含标签和图像路径
        :type file_list: list
        '''
        def reader():
            UNK_ID = self.char_dict['<unk>']
            for image_path, label in file_list:
                label = [self.char_dict.get(c, UNK_ID) for c in label]
                yield self.load_image(image_path), label

        return reader

    def infer_reader(self, file_list):
        '''
        Reader interface for inference.

        :param file_list: The path list of the image file for inference.
        :type file_list: list
        '''

        def reader():
            for image_path, label in file_list:
                yield self.load_image(image_path), label

        return reader

    def load_image(self, path):
        '''
        加载图像并将其转换为一维向量
        :param path: 图像数据的路径
        :type path: str
        '''
        image = load_image(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 将所有图像调整为固定形状
        if self.image_shape:
            image = cv2.resize(
                image, self.image_shape, interpolation=cv2.INTER_CUBIC)

        image = image.flatten() / 255.
        return image
