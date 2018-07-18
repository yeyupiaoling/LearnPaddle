# coding:utf-8
import sys
import paddle.v2 as paddle
import os
from mobilenet import mobile_net
from vgg import vgg_bn_drop
import gzip
import time

class TestCIFAR:
    # ***********************初始化操作***************************************
    def __init__(self):
        # 初始化paddpaddle,
        paddle.init(use_gpu=True, trainer_count=1)

    # **********************获取参数***************************************
    def get_parameters(self, parameters_path=None, cost=None):
        if not parameters_path:
            # 使用cost创建parameters
            if not cost:
                print "请输入cost参数"
            else:
                # 根据损失函数创建参数
                parameters = paddle.parameters.create(cost)
                return parameters
        else:
            # 使用之前训练好的参数
            try:
                # 使用训练好的参数
                with gzip.open(parameters_path, 'r') as f:
                    parameters = paddle.parameters.Parameters.from_tar(f)
                return parameters
            except Exception as e:
                raise NameError("你的参数文件错误,具体问题是:%s" % e)

    # ***********************获取训练器***************************************
    def get_trainer(self):
        # 数据大小
        datadim = 3 * 32 * 32

        # 获得图片对于的信息标签
        lbl = paddle.layer.data(name="label",
                                type=paddle.data_type.integer_value(10))

        # 获取全连接层,也就是分类器
        out = mobile_net(datadim, 10, 1.0)

        # 获得损失函数
        cost = paddle.layer.classification_cost(input=out, label=lbl)

        # 使用之前保存好的参数文件获得参数
        # parameters = self.get_parameters(parameters_path="../model/mobile_net.tar.gz")
        # 使用损失函数生成参数
        parameters = self.get_parameters(cost=cost)

        '''
        定义优化方法
        learning_rate 迭代的速度
        momentum 跟前面动量优化的比例
        regularzation 正则化,防止过拟合
        '''
        momentum_optimizer = paddle.optimizer.Momentum(
            momentum=0.9,
            regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
            learning_rate=0.1 / 128.0,
            learning_rate_decay_a=0.1,
            learning_rate_decay_b=50000 * 100,
            learning_rate_schedule="discexp")

        '''
        创建训练器
        cost 分类器
        parameters 训练参数,可以通过创建,也可以使用之前训练好的参数
        update_equation 优化方法
        '''
        trainer = paddle.trainer.SGD(cost=cost,
                                     parameters=parameters,
                                     update_equation=momentum_optimizer)
        return trainer

    # ***********************开始训练***************************************
    def start_trainer(self):
        # 获得数据
        reader = paddle.batch(reader=paddle.reader.shuffle(reader=paddle.dataset.cifar.train10(),
                                                           buf_size=50000),
                              batch_size=128)

        # 指定每条数据和padd.layer.data的对应关系
        feeding = {"image": 0, "label": 1}

        lists = []
        # 定义训练事件，输出日志
        def event_handler(event):
            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 100 == 0:
                    print "\nPass %d, Batch %d, Cost %f, %s" % (
                        event.pass_id, event.batch_id, event.cost, event.metrics)
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()

            # 每一轮训练完成之后
            if isinstance(event, paddle.event.EndPass):
                # 保存训练好的参数
                model_path = '../model'
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                with gzip.open(model_path + '/mobile_net.tar.gz', 'w') as f:
                    trainer.save_parameter_to_tar(f)

                # 测试准确率
                result = trainer.test(reader=paddle.batch(reader=paddle.dataset.cifar.test10(),
                                                          batch_size=128),
                                      feeding=feeding)
                print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)
                lists.append((event.pass_id, result.cost,
                              result.metrics['classification_error_evaluator']))

        # 获取训练器
        trainer = self.get_trainer()

        '''
        开始训练
        reader 训练数据
        num_passes 训练的轮数
        event_handler 训练的事件,比如在训练的时候要做一些什么事情
        feeding 说明每条数据和padd.layer.data的对应关系
        '''
        trainer.train(reader=reader,
                      num_passes=50,
                      event_handler=event_handler,
                      feeding=feeding)

        # find the best pass
        best = sorted(lists, key=lambda list: float(list[1]))[0]
        print 'Best pass is %s, testing Avgcost is %s' % (best[0], best[1])
        print 'The classification accuracy is %.2f%%' % (100 - float(best[2]) * 100)


if __name__ == '__main__':
    # 开始训练
    testCIFAR = TestCIFAR()
    # 开始训练时间
    testCIFAR.start_trainer()
