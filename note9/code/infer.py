# coding=utf-8
import gzip
import os

import numpy as np
import paddle.v2 as paddle
from PIL import Image
from pascal_voc_conf import cfg

import data_provider
import vgg_ssd_net


def _infer(inferer, infer_data, threshold):
    '''
    预测并返回预测结果
    :param inferer: 预测器
    :param infer_data: 要预测的数据
    :param threshold: 置信度阈值
    :return:
    '''
    ret = []
    # 获取预测原始结果
    infer_res = inferer.infer(input=infer_data)
    # 筛选预测结果
    keep_inds = np.where(infer_res[:, 2] >= threshold)[0]
    for idx in keep_inds:
        ret.append([
            infer_res[idx][0], infer_res[idx][1] - 1, infer_res[idx][2],
            infer_res[idx][3], infer_res[idx][4], infer_res[idx][5],
            infer_res[idx][6]
        ])
    print ret
    return ret


def save_batch_res(ret_res, img_w, img_h, fname_list, fout):
    '''
    保存预测结果
    :param ret_res: 所有的预测结果
    :param img_w: 图像的宽
    :param img_h: 图像的高
    :param fname_list: 图像路径的list
    :param fout: 保存预测结果的文件
    :return:
    '''
    for det_res in ret_res:
        # 获取图像的idx
        img_idx = int(det_res[0])
        # 获取图像的label
        label = int(det_res[1])
        # 获取预测的得分
        conf_score = det_res[2]
        # 获取目标的框
        xmin = det_res[3] * img_w[img_idx]
        ymin = det_res[4] * img_h[img_idx]
        xmax = det_res[5] * img_w[img_idx]
        ymax = det_res[6] * img_h[img_idx]
        # 将预测结果写入到文件中
        fout.write(fname_list[img_idx] + '\t' + str(label) + '\t' + str(
            conf_score) + '\t' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) +
                   ' ' + str(ymax))
        fout.write('\n')


def infer(eval_file_list, save_path, data_args, batch_size, model_path,threshold):
    '''
    预测图像
    :param eval_file_list: 指定图像路径列表
    :param save_path: 指定预测结果保存路径
    :param data_args: 配置数据预处理所需参数
    :param batch_size: 为每多少样本预测一次
    :param model_path: 指模型的位置
    :param threshold: 置信度阈值
    :return:
    '''
    # 通过网络模型获取输出层
    detect_out = vgg_ssd_net.net_conf(mode='infer')
    # 检查模型路径是否正确
    assert os.path.isfile(model_path), 'Invalid model.'
    # 加载训练好的参数
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_path))
    # 或预测器
    inferer = paddle.inference.Inference(
        output_layer=detect_out, parameters=parameters)
    # 获取预测数据
    reader = data_provider.infer(data_args, eval_file_list)
    all_fname_list = [line.strip() for line in open(eval_file_list).readlines()]

    # 预测的数据
    test_data = []
    # 预测的图像路径list
    fname_list = []
    # 图像的宽
    img_w = []
    # 图像的高
    img_h = []
    # 图像的idx
    idx = 0
    '''按批处理推理，
    bbox的坐标将根据图像大小进行缩放
    '''
    with open(save_path, 'w') as fout:
        for img in reader():
            test_data.append([img])
            fname_list.append(all_fname_list[idx])
            w, h = Image.open(os.path.join('../images', fname_list[-1])).size
            img_w.append(w)
            img_h.append(h)
            # 当数据达到一个batch后，就开始预测
            if len(test_data) == batch_size:
                ret_res = _infer(inferer, test_data, threshold)
                save_batch_res(ret_res, img_w, img_h, fname_list, fout)
                # 预测后要清空之前的数据
                test_data = []
                fname_list = []
                img_w = []
                img_h = []
            idx += 1

        # 剩下没有达到一个batch的在在最后也要预测
        if len(test_data) > 0:
            ret_res = _infer(inferer, test_data, threshold)
            save_batch_res(ret_res, img_w, img_h, fname_list, fout)


if __name__ == "__main__":
    paddle.init(use_gpu=True, trainer_count=2)
    # 设置数据参数
    data_args = data_provider.Settings(
        data_dir='../images',
        label_file='../data/label_list',
        resize_h=cfg.IMG_HEIGHT,
        resize_w=cfg.IMG_WIDTH,
        mean_value=[104, 117, 124])
    # 开始预测,batch_size只能设置为1，否则会数据丢失
    infer(
        eval_file_list='../images/infer.txt',
        save_path='../images/infer.res',
        data_args=data_args,
        batch_size=1,
        model_path='../models/params_pass.tar.gz',
        threshold=0.3)
