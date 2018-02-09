# coding=utf-8
import cv2
import paddle.v2 as paddle

class Reader(object):
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
                # 解决key为中文问题
                label2 = []
                for c in unicode(label, "utf-8"):
                    for dict1 in self.char_dict:
                        if c == dict1.decode('utf-8'):
                            label2.append(self.char_dict[dict1])
                yield self.load_image(image_path), label2
        return reader

    def load_image(self, path):
        '''
        加载图像并将其转换为一维向量
        :param path: 图像数据的路径
        :type path: str
        '''
        image = paddle.image.load_image(path,is_color=False)
        # 将所有图像调整为固定形状
        if self.image_shape:
            image = cv2.resize(
                image, self.image_shape, interpolation=cv2.INTER_CUBIC)
        image = image.flatten() / 255.
        return image
