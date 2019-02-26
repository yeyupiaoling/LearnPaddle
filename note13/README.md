# 目录
@[toc]
*本篇文章基于 PaddlePaddle 0.11.0、Python 2.7
# 前言
------
如果读者使用过百度等的一些图像识别的接口，比如百度的[细粒度图像识别](http://ai.baidu.com/docs#/ImageClassify-API/top "细粒度图像识别")接口，应该了解这个过程，省略其他的安全方面的考虑。这个接口大体的流程是，我们把图像上传到百度的网站上，然后服务器把这些图像转换成功矢量数据，最后就是拿这些数据传给深度学习的预测接口，比如是PaddlePaddle的预测接口，获取到预测结果，返回给客户端。这个只是简单的流程，真实的复杂性远远不止这些，但是我们只需要了解这些，然后去搭建属于我们的图像识别接口。

# 环境
------

 1. 系统是：64位 Ubuntu 16.04 
 2. 开发语言是：Python2.7 
 3. web框架是：flask 
 4. 预测接口是：图像识别

# flask的熟悉
----
## 安装flask
安装flask很简单，只要一条命令就可以了：
```shell
pip install flask
```
同时我们也使用到flask_cors，所以我们也要安装这个库
```shell
pip install flask_cors
```
主要安装的是这两个库，如果还缺少哪些库，可以使用pip命令安装，`*`代表读者缺少的库：
```shell
pip install *
```

## 测试flask框架
我们来编写一个简单的程序，来测试我们安装的框架，使用`@app.route('/')`是指定访问的路径：
```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Welcome to PaddlePaddle'

if __name__ == '__main__':
    app.run()
```
然后在浏览器上输入以下地址：
```text
http://127.0.0.1:5000
```
然后浏览器会返回之前写好的字符串：
```text
Welcome to PaddlePaddle
```
## 文件上传
我们来编写一个上传文件的程序，这个程序比上面复杂了一点点，我们要留意这些：
`secure_filename`是为了能够正常获取到上传文件的文件名
`flask_cors`可以实现跨越访问
`methods=['POST']`指定该路径只能使用POST方法访问
`f = request.files['img']`读取表单名称为`img`的文件
`f.save(img_path)`在指定路径保存该文件
```python
from werkzeug.utils import secure_filename
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files['img']
    img_path = './data/' + secure_filename(f.filename)
    print img_path
    f.save(img_path)
    return 'success'
```
然后我们编写一个HTML的网页`index.html`，方便我们测试这个接口：
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>预测图像</title>
</head>
<body>
<form action="http://127.0.0.1:5000/upload" enctype="multipart/form-data" method="post">
    选择要预测的图像：<input type="file" name="img"><br>
    <input type="submit" value="提交">
</form>
</body>
</html>
```
最后我们在浏览器上打开着网页，选择要上传的文件，点击`提交`，如果返回的是`success`，那代表我们已经上传成功了，我们可以去到保存的位置查看文件是否存在。

# 使用PaddlePaddle预测
----
## 获取预测模型
我们这次使用的是第二章的[MNIST手写数字识别](http://blog.csdn.net/qq_33200967/article/details/79095172)的例子，因为这个训练比较快，可以更快的获取到我们需要的预测模型，代码也是类似的，详细可以读到第二章的代码，只是添加了生成拓扑的功能
```python
# 保存预测拓扑图
inference_topology = paddle.topology.Topology(layers=out)
with open("../models/inference_topology.pkl", 'wb') as f:
    inference_topology.serialize_for_inference(f)
```
同时把测试部分去掉了，这样训练起来速度会更快：
```python
result = trainer.test(reader=paddle.batch(paddle.dataset.mnist.test(), batch_size=128))
print "\nTest with Pass %d, Cost %f, %s\n" % (event.pass_id, result.cost, result.metrics)
lists.append((event.pass_id, result.cost, result.metrics['classification_error_evaluator']))
```
最后会获取到这连个文件：

 1. `param.tar`模型参数文件
 2. `inference_topology.pkl`预测拓扑文件

## 把PaddlePaddle部署到服务器
首先我们要创建一个队列，我们要在队列中使用PaddlePaddle进行预测
```python
app = Flask(__name__)
CORS(app)
# 创建主队列
sendQ = Queue()
```
同样我们要编写一个上传文件的接口：
```python
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
```
对于上传文件和保存文件的介绍在上一部分已经讲，接下来就是把图像文件读取读取成矢量数据：
```python
data = []
data.append((load_image(img_path),))
```
`load_image()`这函数在之前使用的是一样的
```python
def load_image(img_path):
    im = Image.open(img_path).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32).flatten()
    im = im / 255.0
    return im
```
然后就是使用主队列发送图像的数据和子队列。使用子队列的作用是为了在PaddlePaddle的预测线程中把预测结果发送回来。


```
# 创建子队列
recv_queue = Queue()
# 使用主队列发送数据和子队列
sendQ.put((data, recv_queue))
```
下面就是我们的PaddlePaddle预测线程

 - 因为PaddlePaddle的初始化和加载模型只能执行一次，所以要放在循环的外面。
 - 在循环中，要从主队列中获取图像数据和子队列
 - 使用图像数据预测并获得结果
 - 使用`recv_queue`把预测结果返回

```python
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
```
回到`infer()`函数中，刚才已经是把数据发送出去了，并有预测结果发送回来，我们这里就接收预测数据，并把预测结果返回给客户端。
```python
# 获取子队列的结果
success, resp = recv_queue.get()
if success:
    # 如果成功返回预测结果
    return successResp(resp)
else:
    # 如果失败返回错误信息
    return errorResp(resp)
```
最后的两个函数是格式化返回的数据，生成的是一个json格式的数据。
```
# 错误请求
def errorResp(msg):
    return jsonify(code=-1, message=msg)

# 成功请求
def successResp(data):
    return jsonify(code=0, message="success", data=data)
```
最后就是启动我们的预测线程和服务了：
```python
if __name__ == '__main__':
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()
    # 已经把端口改成80
    app.run(host='0.0.0.0', port=80, threaded=True)
```
同样在浏览器上打开刚才创建的HTML网页`index.html`，要注意的是提交的`action`改成`http://127.0.0.1/infer`，选择要预测的图像，点击`提交`，便可以获取预测结果
```json
{
  "code": 0, 
  "data": "{\"result\":3,\"possibility\":1.000000}", 
  "message": "success"
}
```





# 项目代码
-----
GitHub地址:https://github.com/yeyupiaoling/LearnPaddle

# 参考资料
---------
 1. http://paddlepaddle.org/
 2. http://blog.csdn.net/u011054333/article/details/70151857
