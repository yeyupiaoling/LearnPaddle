# coding=utf-8
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
from flask_cors import CORS
from Queue import Queue
import threading
import traceback
import paddle.v2 as paddle
from cnn import convolutional_neural_network

app = Flask(__name__)
CORS(app)
# 创建主队列
sendQ = Queue()


# 根路径，返回一个字符串
@app.route('/')
def hello_world():
    return 'Welcome to PaddlePaddle'


# 上传文件
@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files['img']
    img_path = './data/' + secure_filename(f.filename)
    print img_path
    f.save(img_path)
    return 'success'


# 错误请求
def errorResp(msg):
    return jsonify(code=-1, message=msg)


# 成功请求
def successResp(data):
    return jsonify(code=0, message="success", data=data)


@app.route('/infer', methods=['POST'])
def infer():
    # 获取上传的图像
    f = request.files['img']
    img_path = './data/' + secure_filename(f.filename)
    print img_path
    # 保存上传的图像
    f.save(img_path)
    # 把读取上传图像转成矢量
    data = []
    data.append((load_image(img_path),))
    # print '预测数据为：', data

    # 创建子队列
    recv_queue = Queue()
    # 使用主队列发送数据和子队列
    sendQ.put((data, recv_queue))
    # 获取子队列的结果
    success, resp = recv_queue.get()
    if success:
        # 如果成功返回预测结果
        return successResp(resp)
    else:
        # 如果失败返回错误信息
        return errorResp(resp)


# 获取数据
def load_image(img_path):
    im = Image.open(img_path).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32).flatten()
    im = im / 255.0
    return im


# 创建一个PaddlePaddle的预测线程
def worker():
    # 初始化PaddlePaddle
    paddle.init(use_gpu=False, trainer_count=2)

    # 加载模型参数和预测的拓扑生成一个预测器
    with open('../models/param.tar', 'r') as param_f:
        params = paddle.parameters.Parameters.from_tar(param_f)

    # 获取分类器
    out = convolutional_neural_network()

    while True:
        # 获取数据和子队列
        data, recv_queue = sendQ.get()
        try:
            # 获取预测结果
            result = paddle.infer(output_layer=out,
                                  parameters=params,
                                  input=data)

            # 处理预测结果
            lab = np.argsort(-result)
            print lab
            # 返回概率最大的值和其对应的概率值
            result = '{"result":%d,"possibility":%f}' % (lab[0][0], result[0][(lab[0][0])])
            print result
            recv_queue.put((True, result))
        except:
            # 通过子队列发送异常信息
            trace = traceback.format_exc()
            print trace
            recv_queue.put((False, trace))
            continue


if __name__ == '__main__':
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=80, threaded=True)
