# coding=utf-8
from PIL import Image
import numpy as np
import os
from mobilenet import mobile_net
from vgg import vgg_bn_drop
import paddle.v2 as paddle
import time
import gzip


# ***********************使用训练好的参数进行预测***************************************
def to_prediction(image_path, parameters, out):
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

    # 开始预测时间
    start_infer = int(round(time.time() * 1000))

    # 获得要预测的图片
    test_data = []
    test_data.append((load_image(image_path),))

    # 获得预测结果
    probs = paddle.infer(output_layer=out,
                         parameters=parameters,
                         input=test_data)
    # 结束预测时间
    end_infer = int(round(time.time() * 1000))

    print '预测时间：', end_infer - start_infer, 'ms'

    # 处理预测结果
    lab = np.argsort(-probs)
    # 返回概率最大的值和其对应的概率值
    return lab[0][0], probs[0][(lab[0][0])]


if __name__ == '__main__':
    # 开始预测
    paddle.init(use_gpu=False, trainer_count=1)

    # VGG模型
    # out = vgg_bn_drop(3 * 32 * 32, 10)
    # with gzip.open("../model/vgg16.tar.gz", 'r') as f:
    #     parameters = paddle.parameters.Parameters.from_tar(f)

    # MobileNet模型
    out = mobile_net(3 * 32 * 32, 10)
    with gzip.open("../model/mobile_net.tar.gz", 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    image_path = "../images/truck1.png"
    # 总开始预测时间
    start_time = int(round(time.time() * 1000))
    for i in range(10):
        result, probability = to_prediction(image_path=image_path, out=out, parameters=parameters)
        print '预测结果为:%d,可信度为:%f' % (result, probability)

    # 总的预测时间
    end_time = int(round(time.time() * 1000))
    # 打印平均预测时间
    print '\n平均预测时间：', (end_time - start_time)/10, 'ms'
