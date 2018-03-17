# coding:utf-8
import numpy as np
import paddle.v2 as paddle

from vgg import vgg_bn_drop


# **********************获取参数***************************************
def get_parameters(parameters_path):
    with open(parameters_path, 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)
    return parameters


# ***********************使用训练好的参数进行预测***************************************
def to_prediction(image_paths, parameters, out, imageSize):
    # 获得要预测的图片
    test_data = []
    for image_path in image_paths:
        test_data.append((paddle.image.load_and_transform(image_path, 256, imageSize, False)
                          .flatten().astype('float32'),))

    # 获得预测结果
    probs = paddle.infer(output_layer=out,
                         parameters=parameters,
                         input=test_data)
    # 处理预测结果
    lab = np.argsort(-probs)
    # 返回概率最大的值和其对应的概率值
    all_result = []
    for i in range(0, lab.__len__()):
        all_result.append([lab[i][0], probs[i][(lab[i][0])]])
    return all_result


if __name__ == '__main__':
    paddle.init(use_gpu=False, trainer_count=2)
    # 类别总数
    type_size = 3
    # 图片大小
    imageSize = 32
    # 保存的model路径
    parameters_path = "../model/model.tar"
    # 数据的大小
    datadim = 3 * imageSize * imageSize

    # *******************************开始预测**************************************
    # 添加数据
    image_path = []
    image_path.append("../images/vegetables/cuke/1515826971850.jpg")
    image_path.append("../images/vegetables/lettuce/1515827012863.jpg")
    image_path.append("../images/vegetables/lotus_root/1515827059200.jpg")
    out = vgg_bn_drop(datadim=datadim, type_size=type_size)
    parameters = get_parameters(parameters_path=parameters_path)
    all_result = to_prediction(image_paths=image_path, parameters=parameters,
                                          out=out, imageSize=imageSize)
    for i in range(0, all_result.__len__()):
        print '预测结果为:%d,可信度为:%f' % (all_result[i][0], all_result[i][1])
