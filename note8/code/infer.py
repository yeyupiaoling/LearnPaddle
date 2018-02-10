 # coding=utf-8
import gzip

import paddle.v2 as paddle
from network_conf import Model
from reader import DataGenerator
from decoder import ctc_greedy_decoder
from utils import get_file_list, load_dict, load_reverse_dict


def infer_batch(inferer, test_batch, labels, reversed_char_dict):
    # 获取初步预测结果
    infer_results = inferer.infer(input=test_batch)
    num_steps = len(infer_results) // len(test_batch)
    probs_split = [
        infer_results[i * num_steps:(i + 1) * num_steps]
        for i in xrange(0, len(test_batch))
    ]
    results = []
    # 最佳路径解码
    for i, probs in enumerate(probs_split):
        output_transcription = ctc_greedy_decoder(
            probs_seq=probs, vocabulary=reversed_char_dict)
        results.append(output_transcription)
    # 打印预测结果
    for result, label in zip(results, labels):
        print("\n预测结果: %s\n实际文字: %s" %(result, label))


def infer(model_path, image_shape, label_dict_path,infer_file_list_path):

    infer_file_list = get_file_list(infer_file_list_path)
    # 获取标签字典
    char_dict = load_dict(label_dict_path)
    # 获取反转的标签字典
    reversed_char_dict = load_reverse_dict(label_dict_path)
    # 获取字典大小
    dict_size = len(char_dict)
    # 获取reader
    data_generator = DataGenerator(char_dict=char_dict, image_shape=image_shape)
    # 初始化PaddlePaddle
    paddle.init(use_gpu=True, trainer_count=2)
    # 加载训练好的参数
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_path))
    # 获取网络模型
    model = Model(dict_size, image_shape, is_infer=True)
    # 获取预测器
    inferer = paddle.inference.Inference(output_layer=model.log_probs, parameters=parameters)
    # 开始预测
    test_batch = []
    labels = []
    for i, (image, label) in enumerate(data_generator.infer_reader(infer_file_list)()):
        test_batch.append([image])
        labels.append(label)
    infer_batch(inferer, test_batch, labels, reversed_char_dict)


if __name__ == "__main__":
    # 要预测的图像
    infer_file_list_path = '../data/test_data/Challenge2_Test_Task3_GT.txt'
    # 模型的路径
    model_path = '../models/params_pass.tar.gz'
    # 图像的大小
    image_shape = (173, 46)
    # 标签的路径
    label_dict_path = '../data/label_dict.txt'
    # 开始预测
    infer(model_path, image_shape, label_dict_path, infer_file_list_path)
