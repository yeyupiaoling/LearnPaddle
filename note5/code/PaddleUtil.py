# coding:utf-8
import sys
import os
import numpy as np
import paddle.v2 as paddle
from MyReader import MyReader
from PIL import Image
import json


class PaddleUtil:
    # ***********************初始化操作***************************************
    def __init__(self):
        # 初始化paddpaddle,只是用CPU,把GPU关闭
        paddle.init(use_gpu=False, trainer_count=2)

    # 卷积神经网络LeNet-5,获取分类器
    def convolutional_neural_network(self, input, type_size):
        # 第一个卷积--池化层
        conv_pool_1 = paddle.networks.simple_img_conv_pool(input=input,
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
        out = paddle.layer.fc(input=conv_pool_2,
                              size=type_size,
                              act=paddle.activation.Softmax())
        return out

    # ***********************定义VGG卷积神经网络模型***************************************
    def vgg_bn_drop(self, input, type_size):
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

        conv1 = conv_block(input, 64, 2, [0.3, 0], 1)
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
        # 通过Softmax获得分类器
        out = paddle.layer.fc(input=fc2,
                              size=type_size,
                              act=paddle.activation.Softmax())
        return out

    # ***********************获取全连接层,也就是分类器***************************************
    def get_out(self, datadim, type_size):
        print datadim
        # 获取输入数据模式
        image = paddle.layer.data(name="image",
                                  type=paddle.data_type.dense_vector(datadim))

        # 获得卷积神经模型模型
        out = self.vgg_bn_drop(input=image, type_size=type_size)
        # out = self.convolutional_neural_network(input=image, type_size=type_size)
        return out

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
        out = self.get_out(datadim=datadim, type_size=type_size)

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
        # ********************如果使用VGG网络模型就用这个优化方法******************
        optimizer = paddle.optimizer.Momentum(
            momentum=0.9,
            regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128),
            learning_rate=0.001 / 128,
            learning_rate_decay_a=0.1,
            learning_rate_decay_b=128000 * 35,
            learning_rate_schedule="discexp", )
        # ********************如果使用LeNet-5网络模型就用这个优化方法******************
        # optimizer = paddle.optimizer.Momentum(learning_rate=0.1 / 128.0,
        #                                       momentum=0.9,
        #                                       regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128))

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

    # *****************获取你要预测的参数********************************
    def get_TestData(self, path, imageSize):
        def load_images(img):
            # 对图进行灰度化处理
            im = img.convert('L')
            # 缩小到跟训练数据一样大小
            im = im.resize((imageSize, imageSize), Image.ANTIALIAS)
            im = np.array(im).astype(np.float32).flatten()
            im = im / 255.0
            return im

        test_data = []
        img = Image.open(path)
        # 切割图片
        box1 = (5, 0, 17, 27)
        box2 = (17, 0, 29, 27)
        box3 = (29, 0, 41, 27)
        box4 = (41, 0, 53, 27)
        test_data.append((load_images(img.crop(box1)),))
        test_data.append((load_images(img.crop(box2)),))
        test_data.append((load_images(img.crop(box3)),))
        test_data.append((load_images(img.crop(box4)),))
        return test_data

    def lab_to_result(self, lab, json_str):
        myjson = json.loads(json_str)
        class_details = myjson['class_detail']
        for class_detail in class_details:
            if class_detail['class_label'] == lab:
                return class_detail['class_name']

    # ***********************使用训练好的参数进行预测***************************************
    def to_prediction(self, test_data, parameters, out, all_class_name):
        with open('../data/%s/readme.json' % all_class_name) as f:
            txt = f.read()
        # 获得预测结果
        probs = paddle.infer(output_layer=out,
                             parameters=parameters,
                             input=test_data)
        # 处理预测结果
        lab = np.argsort(-probs)
        # 返回概率最大的值和其对应的概率值
        result = ''
        for i in range(0, lab.__len__()):
            print '第%d张预测结果为:%d,可信度为:%f' % (i + 1, lab[i][0], probs[i][(lab[i][0])])
            result = result + self.lab_to_result(lab[i][0], txt)
        return str(result)


if __name__ == '__main__':
    # 类别总数
    type_size = 33
    # 图片大小
    imageSize = 28
    # 总的分类名称
    all_class_name = 'dst_yanzhengma'
    # 保存的model路径
    parameters_path = "../model/model.tar"
    # 数据的大小
    datadim = imageSize * imageSize
    paddleUtil = PaddleUtil()
    myReader = MyReader(imageSize=imageSize)
    # parameters_path设置为None就使用普通生成参数,
    # trainer = paddleUtil.get_trainer(datadim=datadim, type_size=type_size, parameters_path=None)
    # trainer_reader = myReader.train_reader(train_list="../data/%s/trainer.list" % all_class_name)
    # test_reader = myReader.test_reader(test_list="../data/%s/test.list" % all_class_name)
    #
    # paddleUtil.start_trainer(trainer=trainer, num_passes=1000, save_parameters_name=parameters_path,
    #                          trainer_reader=trainer_reader, test_reader=test_reader)

    # 添加数据
    test_data = paddleUtil.get_TestData("../images/src_yanzhengma/0alc.png", imageSize=imageSize)
    out = paddleUtil.get_out(datadim=datadim, type_size=type_size)
    parameters = paddleUtil.get_parameters(parameters_path=parameters_path)
    result = paddleUtil.to_prediction(test_data=test_data,
                                      parameters=parameters,
                                      out=out,
                                      all_class_name=all_class_name)
    print '预测结果为:%s' % result
