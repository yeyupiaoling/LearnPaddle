# coding=utf-8
import gzip
import os

import paddle.v2 as paddle
from network_conf import Model
from reader import Reader
from utils import get_file_list, build_label_dict, load_dict

# 图像的大小，格式(宽度,高度)
IMAGE_SHAPE = (72, 27)
# batch的大小
BATCH_SIZE = 10


def train(train_file_list_path, test_file_list_path, label_dict_path, model_save_dir):
    # 获取训练列表
    train_file_list = get_file_list(train_file_list_path)
    # 获取测试列表
    test_file_list = get_file_list(test_file_list_path)

    # 使用训练数据生成标记字典
    if not os.path.exists(label_dict_path):
        print(("Label dictionary is not given, the dictionary "
               "is automatically built from the training data."))
        build_label_dict(train_file_list, label_dict_path)

    # 获取标签字典
    char_dict = load_dict(label_dict_path)
    # 获取字典大小
    dict_size = len(char_dict)
    # 定义网络拓扑
    model = Model(dict_size, IMAGE_SHAPE, is_infer=False)

    # 初始化PaddlePaddle
    paddle.init(use_gpu=True, trainer_count=1)
    # 创建优化方法
    optimizer = paddle.optimizer.Momentum(momentum=0)
    # 创建训练参数
    params = paddle.parameters.create(model.cost)
    # 定义训练器
    trainer = paddle.trainer.SGD(cost=model.cost,
                                 parameters=params,
                                 update_equation=optimizer,
                                 extra_layers=model.eval)

    # 获取reader
    my_reader = Reader(char_dict=char_dict, image_shape=IMAGE_SHAPE)
    # 说明数据层之间的关系
    feeding = {'image': 0, 'label': 1}

    # 训练事件
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print("Pass %d, batch %d, Samples %d, Cost %f, Eval %s" %
                      (event.pass_id, event.batch_id, event.batch_id *
                       BATCH_SIZE, event.cost, event.metrics))

        if isinstance(event, paddle.event.EndPass):
            # 这里由于训练和测试数据共享相同的格式
            # 我们仍然使用reader.train_reader来读取测试数据
            test_reader = paddle.batch(
                my_reader.train_reader(test_file_list),
                batch_size=BATCH_SIZE)
            result = trainer.test(reader=test_reader, feeding=feeding)
            print("Test %d, Cost %f, Eval %s" % (event.pass_id, result.cost, result.metrics))
            # 检查保存model的路径是否存在，如果不存在就创建
            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)
            with gzip.open(
                    os.path.join(model_save_dir, "params_pass.tar.gz"), "w") as f:
                trainer.save_parameter_to_tar(f)

    # 获取训练数据的reader
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            my_reader.train_reader(train_file_list),
            buf_size=1000),
        batch_size=BATCH_SIZE)
    # 开始训练
    trainer.train(reader=train_reader,
                  feeding=feeding,
                  event_handler=event_handler,
                  num_passes=1000)


if __name__ == "__main__":
    # 训练列表的的路径
    train_file_list_path = '../data/train_data/trainer.list'
    # 测试列表的路径
    test_file_list_path = '../data/test_data/test.list'
    # 标签字典的路径
    label_dict_path = '../data/label_dict.txt'
    # 保存模型的路径
    model_save_dir = '../models'
    train(train_file_list_path, test_file_list_path, label_dict_path, model_save_dir)
