# coding=utf-8
import gzip
import os
import sys

import paddle.v2 as paddle
from pascal_voc_conf import cfg

import data_provider
import vgg_ssd_net


def train(train_file_list, dev_file_list, data_args, init_model_path):
    # 创建优化方法
    optimizer = paddle.optimizer.Momentum(
        momentum=cfg.TRAIN.MOMENTUM,
        learning_rate=cfg.TRAIN.LEARNING_RATE,
        regularization=paddle.optimizer.L2Regularization(
            rate=cfg.TRAIN.L2REGULARIZATION),
        learning_rate_decay_a=cfg.TRAIN.LEARNING_RATE_DECAY_A,
        learning_rate_decay_b=cfg.TRAIN.LEARNING_RATE_DECAY_B,
        learning_rate_schedule=cfg.TRAIN.LEARNING_RATE_SCHEDULE)

    # 通过神经网络模型获取损失函数和额外层
    cost, detect_out = vgg_ssd_net.net_conf('train')
    # 通过损失函数创建训练参数
    parameters = paddle.parameters.create(cost)
    # 如果有训练好的模型，可以使用训练好的模型再训练
    if not (init_model_path is None):
        assert os.path.isfile(init_model_path), 'Invalid model.'
        parameters.init_from_tar(gzip.open(init_model_path))
    # 创建训练器
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 extra_layers=[detect_out],
                                 update_equation=optimizer)
    # 定义数据层之间的关系
    feeding = {'image': 0, 'bbox': 1}
    # 创建训练数据
    train_reader = paddle.batch(
        data_provider.train(data_args, train_file_list),
        batch_size=cfg.TRAIN.BATCH_SIZE)
    # 创建测试数据
    dev_reader = paddle.batch(
        data_provider.test(data_args, dev_file_list),
        batch_size=cfg.TRAIN.BATCH_SIZE)

    # 定义训练事件
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "\nPass %d, Batch %d, TrainCost %f, Detection mAP=%f" % \
                        (event.pass_id,
                         event.batch_id,
                         event.cost,
                         event.metrics['detection_evaluator'])
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

        if isinstance(event, paddle.event.EndPass):
            with gzip.open('../models/params_pass.tar.gz', 'w') as f:
                trainer.save_parameter_to_tar(f)
            result = trainer.test(reader=dev_reader, feeding=feeding)
            print "\nTest with Pass %d, TestCost: %f, Detection mAP=%g" % \
                    (event.pass_id,
                     result.cost,
                     result.metrics['detection_evaluator'])
    # 开始训练
    trainer.train(
        reader=train_reader,
        event_handler=event_handler,
        num_passes=cfg.TRAIN.NUM_PASS,
        feeding=feeding)


if __name__ == "__main__":
    # 初始化PaddlePaddle
    paddle.init(use_gpu=True, trainer_count=2)
    # 设置数据参数
    data_args = data_provider.Settings(
        data_dir='../data',
        label_file='../data/label_list',
        resize_h=cfg.IMG_HEIGHT,
        resize_w=cfg.IMG_WIDTH,
        mean_value=[104, 117, 124])
    # 开始训练
    train(
        train_file_list='../data/trainval.txt',
        dev_file_list='../data/test.txt',
        data_args=data_args,
        init_model_path='../models/vgg_model.tar.gz')
