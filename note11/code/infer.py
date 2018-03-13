# coding=utf-8
import numpy
import paddle.fluid as fluid
import paddle.v2 as paddle
from PIL import Image
import numpy as np


def infer(image_file, use_cuda, model_path):
    # 是否使用GPU
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    # 生成调试器
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # 加载模型
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_path, exe)

        # 获取预测数据
        img = Image.open(image_file)
        img = img.resize((32, 32), Image.ANTIALIAS)
        test_data = np.array(img).astype("float32")
        test_data = np.transpose(test_data, (2, 0, 1))
        test_data = test_data[np.newaxis, :] / 255.0

        print test_data.shape
        # 开始预测
        results = exe.run(inference_program,
                          feed={feed_target_names[0]: test_data},
                          fetch_list=fetch_targets)

        results = np.argsort(-results[0])
        # 打印预测结果
        print "The images/horse4.png infer results label is: ", results[0][0]


if __name__ == '__main__':
    image_file = '../images/horse4.png'
    model_path = '../models/0/'
    infer(image_file, False, model_path)
