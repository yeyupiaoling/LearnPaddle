# coding:utf-8
import json

import numpy as np
import paddle.v2 as paddle
from PIL import Image

from vgg import vgg_bn_drop


# **********************获取参数***************************************
def get_parameters(parameters_path):
    with open(parameters_path, 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)
    return parameters


# *****************获取你要预测的参数********************************
def get_TestData(path, imageSize):
    test_data = []
    img = Image.open(path)
    # 切割图片并保存
    box1 = (5, 0, 17, 27)
    box2 = (17, 0, 29, 27)
    box3 = (29, 0, 41, 27)
    box4 = (41, 0, 53, 27)
    temp = '../images/temp'
    img.crop(box1).resize((32, 32), Image.ANTIALIAS).save(temp + '/1.png')
    img.crop(box2).resize((32, 32), Image.ANTIALIAS).save(temp + '/2.png')
    img.crop(box3).resize((32, 32), Image.ANTIALIAS).save(temp + '/3.png')
    img.crop(box4).resize((32, 32), Image.ANTIALIAS).save(temp + '/4.png')
    # 把图像加载到预测数据中
    test_data.append((paddle.image.load_and_transform(temp + '/1.png', 38, imageSize, False, is_color=False)
                      .flatten().astype('float32'),))
    test_data.append((paddle.image.load_and_transform(temp + '/2.png', 38, imageSize, False, is_color=False)
                      .flatten().astype('float32'),))
    test_data.append((paddle.image.load_and_transform(temp + '/3.png', 38, imageSize, False, is_color=False)
                      .flatten().astype('float32'),))
    test_data.append((paddle.image.load_and_transform(temp + '/4.png', 38, imageSize, False, is_color=False)
                      .flatten().astype('float32'),))
    return test_data


# *****************把预测的label对应的真实字符找到********************************
def lab_to_result(lab, json_str):
    myjson = json.loads(json_str)
    class_details = myjson['class_detail']
    for class_detail in class_details:
        if class_detail['class_label'] == lab:
            return class_detail['class_name']


# ***********************使用训练好的参数进行预测***************************************
def to_prediction(test_data, parameters, out, all_class_name):
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
        result = result + lab_to_result(lab[i][0], txt)
    return str(result)


if __name__ == '__main__':
    paddle.init(use_gpu=False, trainer_count=2)
    # 类别总数
    type_size = 33
    # 图片大小
    imageSize = 32
    # 总的分类名称
    all_class_name = 'dst_yanzhengma'
    # 保存的model路径
    parameters_path = "../model/model.tar"
    # 数据的大小
    datadim = imageSize * imageSize

    # *******************************开始预测**************************************
    out = vgg_bn_drop(datadim=datadim, type_size=type_size)
    parameters = get_parameters(parameters_path=parameters_path)
    # 添加数据
    test_data = get_TestData("../images/src_yanzhengma/0a13.png", imageSize=imageSize)
    result = to_prediction(test_data=test_data,
                           parameters=parameters,
                           out=out,
                           all_class_name=all_class_name)
    print '预测结果为:%s' % result
