# coding:utf-8
from paddle.v2.plot import Ploter
import sys
import paddle.v2 as paddle
from PIL import Image
import numpy as np
import os

step = 0

class TestCIFAR:
    # ***********************初始化操作***************************************
    def __init__(self):
        # 初始化paddpaddle,只是用CPU,把GPU关闭
        paddle.init(use_gpu=False, trainer_count=2)

    # ***********************定义VGG卷积神经网络模型***************************************
    def vgg_bn_drop(self, input):
        def conv_block(ipt, num_filter, groups, dropouts, num_channels=None):
            return paddle.networks.img_conv_group(
                input=ipt,
                num_channels=num_channels,
                pool_size=2,
                pool_stride=2,
                conv_num_filter=[num_filter] * groups,
                conv_filter_size=3,
                conv_act=paddle.activation.Relu(),
                conv_with_batchnorm=True,
                conv_batchnorm_drop_rate=dropouts,
                pool_type=paddle.pooling.Max())

        conv1 = conv_block(input, 64, 2, [0.3, 0], 3)
        conv2 = conv_block(conv1, 128, 2, [0.4, 0])
        conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
        conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
        conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

        drop = paddle.layer.dropout(input=conv5, dropout_rate=0.5)
        fc1 = paddle.layer.fc(input=drop, size=512, act=paddle.activation.Linear())
        bn = paddle.layer.batch_norm(input=fc1,
                                     act=paddle.activation.Relu(),
                                     layer_attr=paddle.attr.Extra(drop_rate=0.5))
        fc2 = paddle.layer.fc(input=bn, size=512, act=paddle.activation.Linear())
        return fc2

    # ***********************定义ResNet卷积神经网络模型***************************************
    def resnet_cifar10(self, ipt, depth=32):
        def conv_bn_layer(input, ch_out, filter_size, stride, padding, active_type=paddle.activation.Relu(),
                          ch_in=None):
            tmp = paddle.layer.img_conv(input=input,
                                        filter_size=filter_size,
                                        num_channels=ch_in,
                                        num_filters=ch_out,
                                        stride=stride,
                                        padding=padding,
                                        act=paddle.activation.Linear(),
                                        bias_attr=False)
            return paddle.layer.batch_norm(input=tmp, act=active_type)

        def shortcut(ipt, n_in, n_out, stride):
            if n_in != n_out:
                return conv_bn_layer(ipt, n_out, 1, stride, 0, paddle.activation.Linear())
            else:
                return ipt

        def basicblock(ipt, ch_out, stride):
            ch_in = ch_out * 2
            tmp = conv_bn_layer(ipt, ch_out, 3, stride, 1)
            tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, paddle.activation.Linear())
            short = shortcut(ipt, ch_in, ch_out, stride)
            return paddle.layer.addto(input=[tmp, short],
                                      act=paddle.activation.Relu())

        def layer_warp(block_func, ipt, features, count, stride):
            tmp = block_func(ipt, features, stride)
            for i in range(1, count):
                tmp = block_func(tmp, features, 1)
            return tmp

        assert (depth - 2) % 6 == 0
        n = (depth - 2) / 6
        nStages = {16, 64, 128}
        conv1 = conv_bn_layer(ipt, ch_in=3, ch_out=16, filter_size=3, stride=1, padding=1)
        res1 = layer_warp(basicblock, conv1, 16, n, 1)
        res2 = layer_warp(basicblock, res1, 32, n, 2)
        res3 = layer_warp(basicblock, res2, 64, n, 2)
        pool = paddle.layer.img_pool(
            input=res3, pool_size=8, stride=1, pool_type=paddle.pooling.Avg())
        return pool

    # ***********************获取全连接层,也就是分类器***************************************
    def get_out(self, datadim):
        # 获取输入数据大小
        image = paddle.layer.data(name="image",
                                  type=paddle.data_type.dense_vector(datadim))

        # 获得卷积神经模型模型
        net = self.vgg_bn_drop(image)
        # net = self.resnet_cifar10(image)

        # 通过神经网络模型再使用Softmax获得分类器(全连接)
        out = paddle.layer.fc(input=net,
                              size=10,
                              act=paddle.activation.Softmax())
        return out

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
                with open(parameters_path, 'r') as f:
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
        out = self.get_out(datadim=datadim)

        # 获得损失函数
        cost = paddle.layer.classification_cost(input=out, label=lbl)

        # 使用之前保存好的参数文件获得参数
        # parameters = self.get_parameters(parameters_path="../model/model.tar")
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
                with open(model_path + '/model.tar', 'w') as f:
                    trainer.save_parameter_to_tar(f)

                # 测试准确率
                result = trainer.test(reader=paddle.batch(reader=paddle.dataset.cifar.test10(),
                                                          batch_size=128),
                                      feeding=feeding)
                print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

        train_title = "Train cost"
        test_title = "Test cost"
        cost_ploter = Ploter(train_title, test_title)

        # 定义训练事件,画出折线图,该事件的图可以在notebook上显示，命令行不会正常输出
        def event_handler_plot(event):
            global step
            if isinstance(event, paddle.event.EndIteration):
                if step % 1 == 0:
                    cost_ploter.append(train_title, step, event.cost)
                    cost_ploter.plot()
                step += 1
            if isinstance(event, paddle.event.EndPass):
                # 保存训练好的参数
                model_path = '../model'
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                with open(model_path + '/model_%d.tar' % event.pass_id, 'w') as f:
                    trainer.save_parameter_to_tar(f)

                result = trainer.test(
                    reader=paddle.batch(
                        paddle.dataset.cifar.test10(), batch_size=128),
                    feeding=feeding)
                cost_ploter.append(test_title, step, result.cost)

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
                      num_passes=100,
                      event_handler=event_handler,
                      feeding=feeding)

    # ***********************使用训练好的参数进行预测***************************************
    def to_prediction(self, image_path, parameters, out):
        # 获取图片
        def load_image(file):
            im = Image.open(file)
            im = im.resize((32, 32), Image.ANTIALIAS)
            im = np.array(im).astype(np.float32)
            # PIL打开图片存储顺序为H(高度)，W(宽度)，C(通道)。
            # PaddlePaddle要求数据顺序为CHW，所以需要转换顺序。
            im = im.transpose((2, 0, 1))
            # CIFAR训练图片通道顺序为B(蓝),G(绿),R(红),
            # 而PIL打开图片默认通道顺序为RGB,因为需要交换通道。
            im = im[(2, 1, 0), :, :]  # BGR
            im = im.flatten()
            im = im / 255.0
            return im

        # 获得要预测的图片
        test_data = []
        test_data.append((load_image(image_path),))

        # 获得预测结果
        probs = paddle.infer(output_layer=out,
                             parameters=parameters,
                             input=test_data)
        # 处理预测结果
        lab = np.argsort(-probs)
        # 返回概率最大的值和其对应的概率值
        return lab[0][0], probs[0][(lab[0][0])]


if __name__ == '__main__':
    testCIFAR = TestCIFAR()
    # 开始训练
    testCIFAR.start_trainer()
    # 开始预测
    out = testCIFAR.get_out(3 * 32 * 32)
    parameters = testCIFAR.get_parameters("../model/model.tar")
    image_path = "../images/0/airplane1.png"
    result,probability = testCIFAR.to_prediction(image_path=image_path, out=out, parameters=parameters)
    print '预测结果为:%d,可信度为:%f' % (result,probability)
