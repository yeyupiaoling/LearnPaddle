# coding=utf-8
from PIL import Image
import numpy as np
import paddle.v2 as paddle
from cnn import convolutional_neural_network

paddle.init(use_gpu=False, trainer_count=2)

# 获取分类器
out = convolutional_neural_network()


# 加载模型参数和预测的拓扑生成一个预测器
with open('../models/param.tar', 'r') as param_f:
    params = paddle.parameters.Parameters.from_tar(param_f)


def load_image(img_path):
    im = Image.open(img_path).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32).flatten()
    im = im / 255.0
    return im

data = []
data.append((load_image('./data/infer_3.png'),))

result = paddle.infer(output_layer=out,
                      parameters=params,
                      input=data)

print result
