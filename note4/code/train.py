# coding:utf-8
import os
import sys
import paddle.v2 as paddle
from MyReader import MyReader
from vgg import vgg_bn_drop


class PaddleUtil:
    # ***********************初始化操作***************************************
    def __init__(self):
        # 初始化paddpaddle,只是用CPU,把GPU关闭
        paddle.init(use_gpu=False, trainer_count=2)

    # **********************获取参数***************************************
    def get_parameters(self, parameters_path=None, cost=None):
        if not parameters_path:
            # 使用cost创建parameters
            if not cost:
                raise NameError('请输入cost参数')
            else:
                # 根据损失函数创建参数
                parameters = paddle.parameters.create(cost)
                print "cost"
                return parameters
        else:
            # 使用之前训练好的参数
            try:
                # 使用训练好的参数
                with open(parameters_path, 'r') as f:
                    parameters = paddle.parameters.Parameters.from_tar(f)
                print "使用parameters"
                return parameters
            except Exception as e:
                raise NameError("你的参数文件错误,具体问题是:%s" % e)

    # ***********************获取训练器***************************************
    # datadim 数据大小
    def get_trainer(self, datadim, type_size, parameters_path):
        # 获得图片对于的信息标签
        label = paddle.layer.data(name="label",
                                  type=paddle.data_type.integer_value(type_size))

        # 获取全连接层,也就是分类器
        out = vgg_bn_drop(datadim=datadim, type_size=type_size)

        # 获得损失函数
        cost = paddle.layer.classification_cost(input=out, label=label)

        # 获得参数
        if not parameters_path:
            parameters = self.get_parameters(cost=cost)
        else:
            parameters = self.get_parameters(parameters_path=parameters_path)

        '''
        定义优化方法
        learning_rate 迭代的速度
        momentum 跟前面动量优化的比例
        regularzation 正则化,防止过拟合
        '''
        optimizer = paddle.optimizer.Momentum(
            momentum=0.9,
            regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128),
            learning_rate=0.001 / 128,
            learning_rate_decay_a=0.1,
            learning_rate_decay_b=128000 * 35,
            learning_rate_schedule="discexp", )

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

    # ***********************开始训练***************************************
    def start_trainer(self, trainer, num_passes, save_parameters_name, trainer_reader, test_reader):
        # 获得数据
        reader = paddle.batch(reader=paddle.reader.shuffle(reader=trainer_reader,
                                                           buf_size=50000),
                              batch_size=128)
        # 保证保存模型的目录是存在的
        father_path = save_parameters_name[:save_parameters_name.rfind("/")]
        if not os.path.exists(father_path):
            os.makedirs(father_path)

        # 指定每条数据和padd.layer.data的对应关系
        feeding = {"image": 0, "label": 1}

        # 定义训练事件
        def event_handler(event):
            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 100 == 0:
                    print "\nPass %d, Batch %d, Cost %f, Error %s" % (
                        event.pass_id, event.batch_id, event.cost, event.metrics['classification_error_evaluator'])
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()

            # 每一轮训练完成之后
            if isinstance(event, paddle.event.EndPass):
                # 保存训练好的参数
                with open(save_parameters_name, 'w') as f:
                    trainer.save_parameter_to_tar(f)

                # 测试准确率
                result = trainer.test(reader=paddle.batch(reader=test_reader,
                                                          batch_size=128),
                                      feeding=feeding)
                print "\nTest with Pass %d, Classification_Error %s" % (
                event.pass_id, result.metrics['classification_error_evaluator'])

        '''
        开始训练
        reader 训练数据
        num_passes 训练的轮数
        event_handler 训练的事件,比如在训练的时候要做一些什么事情
        feeding 说明每条数据和padd.layer.data的对应关系
        '''
        trainer.train(reader=reader,
                      num_passes=num_passes,
                      event_handler=event_handler,
                      feeding=feeding)


if __name__ == '__main__':
    # 类别总数
    type_size = 3
    # 图片大小
    imageSize = 32
    # 总的分类名称
    all_class_name = 'vegetables'
    # 保存的model路径
    parameters_path = "../model/model.tar"
    # 数据的大小
    datadim = 3 * imageSize * imageSize
    paddleUtil = PaddleUtil()

    # *******************************开始训练**************************************
    myReader = MyReader(imageSize=imageSize)
    # # parameters_path设置为None就使用普通生成参数,
    trainer = paddleUtil.get_trainer(datadim=datadim, type_size=type_size, parameters_path=None)
    trainer_reader = myReader.train_reader(train_list="../data/%s/trainer.list" % all_class_name)
    test_reader = myReader.test_reader(test_list="../data/%s/test.list" % all_class_name)

    paddleUtil.start_trainer(trainer=trainer, num_passes=100, save_parameters_name=parameters_path,
                             trainer_reader=trainer_reader, test_reader=test_reader)
