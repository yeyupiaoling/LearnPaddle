# 目录
@[toc]
*本篇文章基于 PaddlePaddle 0.13.0、Python 2.7
# 前言
------
VisualDL是一个面向深度学习任务设计的可视化工具，包含了scalar、参数分布、模型结构、图像可视化等功能。可以这样说：“所见即所得”。我们可以借助VisualDL来观察我们训练的情况，方便我们对训练的模型进行分析，改善模型的收敛情况。

之前我们使用的`paddle.v2.plot`接口，也可以观察训练的情况，但是只是支持CSOT的折线图而已。而VisualDL可以支持一下这个功能：

 1. `scalar`，趋势图，可用于训练测试误差的展示 
![这里写图片描述](//img-blog.csdn.net/20180314105807560?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
 2. `image`, 图片的可视化，可用于卷积层或者其他参数的图形化展示 
![这里写图片描述](//img-blog.csdn.net/20180314105838309?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
 3. `histogram`, 用于参数分布及变化趋势的展示 
![这里写图片描述](//img-blog.csdn.net/20180314105859971?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
 4. `graph`，用于训练模型结构的可视化
![这里写图片描述](//img-blog.csdn.net/20180314105922862?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
以上的图像来自[VisualDL的Github](https://github.com/PaddlePaddle/VisualDL)

既然那么方便，那么我们就来尝试一下吧。VisualDL底层采用C++编写，但是它在提供C++ SDK的同时，也支持Python SDK，我们主要是使用Python的SDK。顺便说一下，VisualDL除了支持PaddlePaddle,之外，还支持pytorch, mxnet在内的大部分主流DNN平台。

# VisualDL的安装
本章只讲述在Ubuntu系统上的安装和使用，Mac的操作应该也差不多。

## 使用pip安装
使用pip安装非常简单，只要一条命令就够了，如下：
```shell
pip install --upgrade visualdl
```
测试一下是否安装成功了，运行一个例子下载日志文件：
```shell
# 在当前位置下载一个日志
vdl_create_scratch_log
# 如果提示命令不存在，那就使用下面这条命令
vdl_scratch.py
```
然后再输入，启动VisualDL并加载这个日志信息：
```shell
visualdl --logdir ./scratch_log --port 8080
```
这里说明一下，visualDL的参数：

 - `host` 设定IP
 - `port` 设定端口
 - `model_pb` 指定 ONNX 格式的模型文件，这木方我们还没要用到

**注意：** 如果是报以下的错误，那是因为protobuf版本过低的原因。
```
root@test:/home/test/VisualDL# visualdl --logdir ./scratch_log --port 8080
Traceback (most recent call last):
  File "/usr/local/bin/visualdl", line 29, in <module>
    import visualdl.server.graph as vdl_graph
  File "/usr/local/lib/python2.7/dist-packages/visualdl/server/graph.py", line 23, in <module>
    from . import onnx
  File "/usr/local/lib/python2.7/dist-packages/visualdl/server/onnx/__init__.py", line 8, in <module>
    from .onnx_pb2 import ModelProto
  File "/usr/local/lib/python2.7/dist-packages/visualdl/server/onnx/onnx_pb2.py", line 213, in <module>
    options=None, file=DESCRIPTOR),
TypeError: __init__() got an unexpected keyword argument 'file'
```

protobuf的版本要不小于3.5.0，如何小于这个版本可以使用以下命令升级：
```
pip install protobuf -U
```

然后在浏览器上输入：
```
http://127.0.0.1:8080
```
即可看到一个可视化的界面，如下：
![这里写图片描述](//img-blog.csdn.net/20180314124348701?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 使用源码安装
如果读者出于各种情况，使用pip安装不能满足需求，那可以考虑使用源码安装VisualDL，操作如下：
首先要安装依赖库：
```shell
# 安装npm
apt install npm
# 安装node
apt install nodejs-legacy
# 安装cmake
apt install cmake
# 安装unzip
apt install unzip
```
然后在GitHub上clone最新的源码并打开：
```shell
git clone https://github.com/PaddlePaddle/VisualDL.git
cd VisualDL
```
之后是编译生成`whl`安装包：
```shell
python setup.py bdist_wheel
```
生成`whl`安装包之后，就可以使用pip命令安装这个安装包了，`*`号对应的是visualdl版本号，读者要根据实际情况来安装：
```shell
pip install --upgrade dist/visualdl-*.whl
```
安装完成之后，同样可以使用在上一部分的[使用pip安装](http://mp.csdn.net/mdeditor/79127175#%E4%BD%BF%E7%94%A8pip%E5%AE%89%E8%A3%85)的测试方法测试安装是否成功。

# 简单使用VisualDL
我们编写下面这一小段的代码来学习VisualDL的使用，代码如下：
```python
# coding=utf-8
# 导入VisualDL的包
from visualdl import LogWriter

# 创建一个LogWriter，第一个参数是指定存放数据的路径，
# 第二个参数是指定多少次写操作执行一次内存到磁盘的数据持久化
logw = LogWriter("./random_log", sync_cycle=10000)

# 创建训练和测试的scalar图，
# mode是标注线条的名称，
# scalar标注的是指定这个组件的tag
with logw.mode('train') as logger:
    scalar0 = logger.scalar("scratch/scalar")

with logw.mode('test') as logger:
    scalar1 = logger.scalar("scratch/scalar")

# 读取数据
for step in range(1000):
    scalar0.add_record(step, step * 1. / 1000)
    scalar1.add_record(step, 1. - step * 1. / 1000)
```
运行Python代码之后，在终端上输入，从上面的代码可以看到我们定义的路径是`./random_log`：
```shell
visualDL --logdir ./random_log --port 8080
```
然后在浏览器上输入：
```
http://127.0.0.1:8080
```
然后就可以看到刚才编写Python代码生成的图像了：
![这里写图片描述](//img-blog.csdn.net/20180314123345810?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

经过这个例子，读者对VisualDL有了进一步的了解了，那么在接下来的我们就在实际的PaddlePaddle例子中使用我们的VisualDL。

# 在PaddlePaddle使用VisualDL
------
## 定义VisualDL组件
创建三个组件：`scalar`，`image`，`histogram`，并指定存放日志的路径
```
# 创建VisualDL，并指定当前该项目的VisualDL的路径
logdir = "../data/tmp"
logwriter = LogWriter(logdir, sync_cycle=10)

# 创建loss的趋势图
with logwriter.mode("train") as writer:
    loss_scalar = writer.scalar("loss")

# 创建acc的趋势图
with logwriter.mode("train") as writer:
    acc_scalar = writer.scalar("acc")

# 定义没多少次重新输出一遍
num_samples = 4
# 创建卷积层和输出图像的图形化展示
with logwriter.mode("train") as writer:
    conv_image = writer.image("conv_image", num_samples, 1)
    input_image = writer.image("input_image", num_samples, 1)

# 创建可视化的训练模型结构
with logwriter.mode("train") as writer:
    param1_histgram = writer.histogram("param1", 100)
```

## 编写PaddlePaddle代码
然后创建PaddlePaddle代码，我们使用的是PaddlePaddle的Fluid版本，如果对Fluid版本不熟悉的话，可以阅读笔者的上一篇文章[新版本Fluid的使用](http://blog.csdn.net/qq_33200967/article/details/79126897)，了解Fluid版本之后再继续阅读下面的代码，如果读者已经很熟悉Fluid版本的使用了，那就往下看。

定义`data`和`label`，代码如下：
```python
# 定义图像的类别数量
class_dim = 10
# 定义图像的通道数和大小
image_shape = [3, 32, 32]
# 定义输入数据大小，指定图像的形状，数据类型是浮点型
image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
# 定义标签，类型是整型
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
```
然后是获取分类器，这里跟上一篇有点不一样，这里还要提供第一层卷积，这是在训练的时候要使用到，使用它来获得卷积层的输出。
```python
# 获取神经网络
net, conv1 = vgg16_bn_drop(image)
# 获取全连接输出，获得分类器
predict = fluid.layers.fc(
    input=net,
    size=class_dim,
    act='softmax',
    param_attr=ParamAttr(name="param1", initializer=NormalInitializer()))
```
之后获取损失函数和batch_acc，在这些之后才能定义优化方法。
```python
# 获取损失函数
cost = fluid.layers.cross_entropy(input=predict, label=label)
# 定义平均损失函数
avg_cost = fluid.layers.mean(x=cost)

# 每个batch计算的时候能取到当前batch里面样本的个数，从而来求平均的准确率
batch_size = fluid.layers.create_tensor(dtype='int64')
print batch_size
batch_acc = fluid.layers.accuracy(input=predict, label=label, total=batch_size)

# 定义优化方法
optimizer = fluid.optimizer.Momentum(
    learning_rate=learning_rate,
    momentum=0.9,
    regularization=fluid.regularizer.L2Decay(5 * 1e-5))

opts = optimizer.minimize(avg_cost)
```
然后就开始创建调试器，并让其初始化。
```python
# 是否使用GPU
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
# 创建调试器
exe = fluid.Executor(place)
# 初始化调试器
exe.run(fluid.default_startup_program())
```
在训练之前，还有获取训的数据，这里没有使用到测试，所以就没有获取测试的数据。
```python
# 获取训练数据
train_reader = paddle.batch(
    paddle.dataset.cifar.train10(), batch_size=BATCH_SIZE)

# 指定数据和label的对于关系
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
```
这里多了一步，这是为了让调试器在训练的时候也输出参数的分布和变化趋势。
```python
step = 0
sample_num = 0
start_up_program = framework.default_startup_program()
param1_var = start_up_program.global_block().var("param1")
```
现在就可以开始训练了，一共输出的四个值：`loss`, `conv1_out`, `param1`, `acc`, `weight`，这些在图像输出上，我们都是用到的。
```python
accuracy = fluid.average.WeightedAverage()
# 开始训练，使用循环的方式来指定训多少个Pass
for pass_id in range(num_passes):
    # 从训练数据中按照一个个batch来读取数据
    accuracy.reset()
    for batch_id, data in enumerate(train_reader()):
        loss, conv1_out, param1, acc, weight = exe.run(fluid.default_main_program(),
                                                       feed=feeder.feed(data),
                                                       fetch_list=[avg_cost, conv1, param1_var, batch_acc,
                                                                   batch_size])
        accuracy.add(value=acc, weight=weight)
        pass_acc = accuracy.eval()
```

## 把数据都添加到VisualDL
加载卷积层和输入图像的数据加载到VisualDL中
```python
# 重新启动图形化展示组件
if sample_num == 0:
    input_image.start_sampling()
    conv_image.start_sampling()
# 获取taken
idx1 = input_image.is_sample_taken()
idx2 = conv_image.is_sample_taken()
# 保证它们的taken是一样的
assert idx1 == idx2
idx = idx1
if idx != -1:
    # 加载输入图像的数据数据
    image_data = data[0][0]
    input_image_data = np.transpose(
        image_data.reshape(image_shape), axes=[1, 2, 0])
    input_image.set_sample(idx, input_image_data.shape,
                           input_image_data.flatten())
    # 加载卷积数据
    conv_image_data = conv1_out[0][0]
    conv_image.set_sample(idx, conv_image_data.shape,
                          conv_image_data.flatten())

    # 完成输出一次
    sample_num += 1
    if sample_num % num_samples == 0:
        input_image.finish_sampling()
        conv_image.finish_sampling()
        sample_num = 0
```
加载趋势图的数据，这里包括了loss和平均错误率。
```python
# 加载趋势图的数据
loss_scalar.add_record(step, loss)
acc_scalar.add_record(step, acc)
```
加载参数变化的数据
```python
# 添加模型结构数据
param1_histgram.add_record(step, param1.flatten())
```
然后是运行项目，在运行项目的时候，会输出一下的日志信息：
```text
loss:[16.7996] acc:[0.0703125] pass_acc:[0.0703125]
loss:[15.192436] acc:[0.1171875] pass_acc:[0.09375]
loss:[14.519127] acc:[0.109375] pass_acc:[0.09895833]
loss:[15.262356] acc:[0.125] pass_acc:[0.10546875]
loss:[13.626783] acc:[0.078125] pass_acc:[0.1]
loss:[11.8960285] acc:[0.09375] pass_acc:[0.09895833]
```
同时运行我们的VisualDL，笔者把VisualDL的日志都存放在`data`目录下，所以我们要去到该目录，然后输入以下命令：
```shell
visualDL --logdir ./tmp --port 8080
```
然后在浏览器上输入：
```text
http://127.0.0.1:8080
```
即可看到我们项目的图像了：

  1. 我们训练的趋势图
![这里写图片描述](//img-blog.csdn.net/201803151907382?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

  2. 卷积和输入图像的可视化页面
![这里写图片描述](//img-blog.csdn.net/20180315190757430?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

  3. 训练参数的变化情况
![这里写图片描述](//img-blog.csdn.net/20180315190813420?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



# 项目代码
-------
GitHub地址:https://github.com/yeyupiaoling/LearnPaddle

# 参考资料
---------
 1. http://paddlepaddle.org/
 2. https://github.com/PaddlePaddle/VisualDL
