# encoding:utf-8
import numpy as np
import paddle.v2 as paddle
from PIL import Image
from cnn import convolutional_neural_network


class TestMNIST:
    def __init__(self):
        # 该模型运行在CUP上，CUP的数量为2
        paddle.init(use_gpu=False, trainer_count=2)


    # *****************获取参数********************************
    def get_parameters(self):
        with open("../model/model.tar", 'r') as f:
            parameters = paddle.parameters.Parameters.from_tar(f)
        return parameters


    # *****************获取你要预测的参数********************************
    def get_TestData(self ,path):
        def load_images(file):
            # 对图进行灰度化处理
            im = Image.open(file).convert('L')
            # 缩小到跟训练数据一样大小
            im = im.resize((28, 28), Image.ANTIALIAS)
            im = np.array(im).astype(np.float32).flatten()
            im = im / 255.0
            return im

        test_data = []
        test_data.append((load_images(path),))
        return test_data

    # *****************使用训练好的参数进行预测********************************
    def to_prediction(self, out, parameters, test_data):

        # 开始预测
        probs = paddle.infer(output_layer=out,
                             parameters=parameters,
                             input=test_data)
        # 处理预测结果并打印
        lab = np.argsort(-probs)
        print "预测结果为: %d" % lab[0][0]


if __name__ == "__main__":
    testMNIST = TestMNIST()
    # 开始预测
    out = convolutional_neural_network()
    parameters = testMNIST.get_parameters()
    test_data = testMNIST.get_TestData('../images/infer_3.png')
    testMNIST.to_prediction(out=out, parameters=parameters, test_data=test_data)
