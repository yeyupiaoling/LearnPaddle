# encoding:utf-8
import os
import sys
import paddle.v2 as paddle
from cnn import convolutional_neural_network


class TestMNIST:
    def __init__(self):
        # 该模型运行在CUP上，CUP的数量为2
        paddle.init(use_gpu=False, trainer_count=2)

    # *****************获取训练器********************************
    def get_trainer(self):

        # 获取分类器
        out = convolutional_neural_network()

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
                # 使用测试进行测试
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
                      num_passes=100,
                      event_handler=event_handler)


if __name__ == "__main__":
    testMNIST = TestMNIST()
    # 开始训练
    testMNIST.start_trainer()
