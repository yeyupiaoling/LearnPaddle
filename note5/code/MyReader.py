# coding=utf-8
from multiprocessing import cpu_count

import paddle.v2 as paddle
import image


class MyReader:
    def __init__(self, imageSize):
        self.imageSize = imageSize

    def train_mapper(self, sample):
        img, label = sample
        # 我这里使用的是本地的image,如果你的paddlepaddle是最新的,也可以使用padd.v2.image
        # 因为是灰度图,所以is_color=False
        img = image.load_image(img, is_color=False)
        img = image.simple_transform(img, 38, self.imageSize, True, is_color=False)
        return img.flatten().astype('float32'), label

    def test_mapper(self, sample):
        img, label = sample
        # 我这里使用的是本地的image,如果你的paddlepaddle是最新的,也可以使用padd.v2.image
        # 因为是灰度图,所以is_color=False
        img = image.load_image(img, is_color=False)
        img = image.simple_transform(img, 38, self.imageSize, False, is_color=False)
        return img.flatten().astype('float32'), label

    def train_reader(self, train_list, buffered_size=1024):
        def reader():
            with open(train_list, 'r') as f:
                lines = [line.strip() for line in f]
                for line in lines:
                    img_path, lab = line.strip().split('\t')
                    yield img_path, int(lab)

        return paddle.reader.xmap_readers(self.train_mapper, reader,
                                          cpu_count(), buffered_size)

    def test_reader(self, test_list, buffered_size=1024):
        def reader():
            with open(test_list, 'r') as f:
                lines = [line.strip() for line in f]
                for line in lines:
                    img_path, lab = line.strip().split('\t')
                    yield img_path, int(lab)

        return paddle.reader.xmap_readers(self.test_mapper, reader,
                                          cpu_count(), buffered_size)
