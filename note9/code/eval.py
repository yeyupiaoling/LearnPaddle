# coding=utf-8
import gzip
import os

import paddle.v2 as paddle
from pascal_voc_conf import cfg

import data_provider
import vgg_ssd_net


def eval(eval_file_list, batch_size, data_args, model_path):
    '''
    评估脚本，用于评估训好模型
    :param eval_file_list:要测试的图像列表
    :param batch_size:batch的大小
    :param data_args:数据的设置参数
    :param model_path:模型的路径
    :return:
    '''
    # 通过神经网络模型获取损失函数和额外层
    cost, detect_out = vgg_ssd_net.net_conf(mode='eval')
    # 检查模型模型路径是否正确
    assert os.path.isfile(model_path), 'Invalid model.'
    # 通过训练好的模型生成参数
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_path))
    # 创建优化方法
    optimizer = paddle.optimizer.Momentum()
    # 创建训练器
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 extra_layers=[detect_out],
                                 update_equation=optimizer)
    # 定义数据层之间的关系
    feeding = {'image': 0, 'bbox': 1}
    # 生成要训练的数据
    reader = paddle.batch(
        data_provider.test(data_args, eval_file_list), batch_size=batch_size)
    # 获取测试结果
    result = trainer.test(reader=reader, feeding=feeding)
    # 打印模型的测试信息
    print "TestCost: %f, Detection mAP=%g" % \
          (result.cost, result.metrics['detection_evaluator'])


if __name__ == "__main__":
    paddle.init(use_gpu=True, trainer_count=2)
    # 设置数据参数
    data_args = data_provider.Settings(
        data_dir='../data',
        label_file='../data/label_list',
        resize_h=cfg.IMG_HEIGHT,
        resize_w=cfg.IMG_WIDTH,
        mean_value=[104, 117, 124])
    # 开始评估
    eval(eval_file_list='../data/test.txt',
         batch_size=4,
         data_args=data_args,
         model_path='../models/params_pass.tar.gz')
