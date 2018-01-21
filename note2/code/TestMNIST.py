# encoding:utf-8
from PIL import Image
import paddle.v2 as paddle
import numpy as np
import os
import sys


class TestMNIST:
    def __init__(self):
        # 该模型运行在CUP上，CUP的数量为2
        paddle.init(use_gpu=False, trainer_count=2)

    # 卷积神经网络LeNet-5,获取分类器
    def convolutional_neural_network(self, img):
        # 第一个卷积--池化层
        conv_pool_1 = paddle.networks.simple_img_conv_pool(input=img,
                                                           filter_size=5,
                                                           num_filters=20,
                                                           num_channel=1,
                                                           pool_size=2,
                                                           pool_stride=2,
                                                           act=paddle.activation.Relu())
        # 第二个卷积--池化层
        conv_pool_2 = paddle.networks.simple_img_conv_pool(input=conv_pool_1,
                                                           filter_size=5,
                                                           num_filters=50,
                                                           num_channel=20,
                                                           pool_size=2,
                                                           pool_stride=2,
                                                           act=paddle.activation.Relu())
        # 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
        predict = paddle.layer.fc(input=conv_pool_2,
                                  size=10,
                                  act=paddle.activation.Softmax())
        return predict

    # *****************获取分类器,也就是全连接层********************************
    def get_out(self):
        # 定义数据模型,数据大小是28*28,即784
        images = paddle.layer.data(name="pixel",
                                   type=paddle.data_type.dense_vector(784))

        # 获取全连接层,也是分类器
        out = self.convolutional_neural_network(img=images)  # LeNet-5卷积神经网络

        return out

    # *****************获取训练器********************************
    def get_trainer(self):

        # 获取分类器
        out = self.get_out()

        # 定义标签
        label = paddle.layer.data(name="label",
                                  type=paddle.data_type.integer_value(10))

        # 获取损失函数
        cost = paddle.layer.classification_cost(input=out, label=label)

        # 获取参数
        parameters = paddle.parameters.create(layers=cost)

        """
        定义优化方法
        learning_rate 迭代的速度
        momentum 跟前面动量优化的比例
        regularzation 正则化,防止过拟合
        :leng re
        """
        optimizer = paddle.optimizer.Momentum(learning_rate=0.1 / 128.0,
                                              momentum=0.9,
                                              regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128))
        '''
        创建训练器
        cost 分类器
        parameters 训练参数,可以通过创建,也可以使用之前训练好的参数
        update_equation 优化方法
        '''
        trainer = paddle.trainer.SGD(cost=cost,
                                     parameters=parameters,
                                     update_equation=optimizer)
        return trainer

    # *****************开始训练********************************
    def start_trainer(self):
        # 获取训练器
        trainer = self.get_trainer()

        # 定义训练事件
        def event_handler(event):
            lists = []
            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 100 == 0:
                    print "\nPass %d, Batch %d, Cost %f, %s" % (
                        event.pass_id, event.batch_id, event.cost, event.metrics)
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()
            if isinstance(event, paddle.event.EndPass):
                # 保存训练好的参数
                model_path = '../model'
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                with open(model_path + "/model.tar", 'w') as f:
                    trainer.save_parameter_to_tar(f=f)

                result = trainer.test(reader=paddle.batch(paddle.dataset.mnist.test(), batch_size=128))
                print "\nTest with Pass %d, Cost %f, %s\n" % (event.pass_id, result.cost, result.metrics)
                lists.append((event.pass_id, result.cost, result.metrics['classification_error_evaluator']))

        # 获取数据
        reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=20000),
                              batch_size=128)
        '''
        开始训练
        reader 训练数据
        num_passes 训练的轮数
        event_handler 训练的事件,比如在训练的时候要做一些什么事情
        '''
        trainer.train(reader=reader,
                      num_passes=200,
                      event_handler=event_handler)

    # *****************获取参数********************************
    def get_parameters(self):
        with open("../model/model.tar", 'r') as f:
            parameters = paddle.parameters.Parameters.from_tar(f)
        return parameters


    # *****************获取你要预测的参数********************************
    def get_TestData(self):
        def load_images(file):
            # 对图进行灰度化处理
            im = Image.open(file).convert('L')
            # 缩小到跟训练数据一样大小
            im = im.resize((28, 28), Image.ANTIALIAS)
            im = np.array(im).astype(np.float32).flatten()
            im = im / 255.0
            return im

        test_data = []
        test_data.append((load_images('../images/infer_3.png'),))
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
    # 开始训练
    testMNIST.start_trainer()

    # out = testMNIST.get_out()
    # parameters = testMNIST.get_parameters()
    # test_data = testMNIST.get_TestData()
    # # 开始预测
    # testMNIST.to_prediction(out=out, parameters=parameters, test_data=test_data)
