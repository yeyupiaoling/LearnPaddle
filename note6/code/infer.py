# coding=utf-8
import gzip

import paddle.v2 as paddle
from network_conf import Model
from reader import Reader
from decoder import ctc_greedy_decoder
from utils import load_dict, load_reverse_dict


def start_infer(inferer, test_batch, reversed_char_dict):
    # 获取初步预测结果
    infer_results = inferer.infer(input=test_batch)
    num_steps = len(infer_results) // len(test_batch)
    probs_split = [
        infer_results[i * num_steps:(i + 1) * num_steps]
        for i in range(0, len(test_batch))]
    # 最佳路径解码
    result = ''
    for i, probs in enumerate(probs_split):
        result = ctc_greedy_decoder(
            probs_seq=probs, vocabulary=reversed_char_dict)
    return result


def infer(img_path, model_path, image_shape, label_dict_path):
    # 获取标签字典
    char_dict = load_dict(label_dict_path)
    # 获取反转的标签字典
    reversed_char_dict = load_reverse_dict(label_dict_path)
    # 获取字典大小
    dict_size = len(char_dict)
    # 获取reader
    my_reader = Reader(char_dict=char_dict, image_shape=image_shape)
    # 初始化PaddlePaddle
    paddle.init(use_gpu=True, trainer_count=1)
    # 加载训练好的参数
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_path))
    # 获取网络模型
    model = Model(dict_size, image_shape, is_infer=True)
    # 获取预测器
    inferer = paddle.inference.Inference(output_layer=model.log_probs, parameters=parameters)
    # 加载数据
    test_batch = [[my_reader.load_image(img_path)]]
    # 开始预测
    return start_infer(inferer, test_batch, reversed_char_dict)


if __name__ == "__main__":
    # 要预测的图像
    img_path = '../data/test_data/4uqh.png'
    # 模型的路径
    model_path = '../models/params_pass.tar.gz'
    # 图像的大小
    image_shape = (72, 27)
    # 标签的路径
    label_dict_path = '../data/label_dict.txt'
    # 获取预测结果
    result = infer(img_path, model_path, image_shape, label_dict_path)
    print '预测结果：%s' % result
