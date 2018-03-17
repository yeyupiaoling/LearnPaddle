from multiprocessing import cpu_count
import paddle.v2 as paddle

class MyReader:
    def __init__(self,imageSize):
        self.imageSize = imageSize

    def train_mapper(self,sample):
        '''
        map image path to type needed by model input layer for the training set
        '''
        img, label = sample
        img = paddle.image.load_image(img)
        img = paddle.image.simple_transform(img, 256, self.imageSize, True)
        return img.flatten().astype('float32'), label

    def test_mapper(self,sample):
        '''
        map image path to type needed by model input layer for the test set
        '''
        img, label = sample
        img = paddle.image.load_image(img)
        img = paddle.image.simple_transform(img, 256, self.imageSize, False)
        return img.flatten().astype('float32'), label

    def train_reader(self,train_list, buffered_size=1024):
        def reader():
            with open(train_list, 'r') as f:
                lines = [line.strip() for line in f]
                for line in lines:
                    img_path, lab = line.strip().split('\t')
                    yield img_path, int(lab)

        return paddle.reader.xmap_readers(self.train_mapper, reader,
                                          cpu_count(), buffered_size)

    def test_reader(self,test_list, buffered_size=1024):
        def reader():
            with open(test_list, 'r') as f:
                lines = [line.strip() for line in f]
                for line in lines:
                    img_path, lab = line.strip().split('\t')
                    yield img_path, int(lab)

        return paddle.reader.xmap_readers(self.test_mapper, reader,
                                          cpu_count(), buffered_size)

