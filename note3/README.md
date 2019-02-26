# 目录
@[toc]
*本篇文章基于 PaddlePaddle 0.11.0、Python 2.7
# 数据集的介绍
----------
本次项目中使用的是一个32*32的彩色图像的数据集CIFAR-10,CIFAR-10数据集包含10个类的60000个32x32彩色图像，每个类有6000个图像。有50000个训练图像和10000个测试图像。数据集分为五个训练batch和一个测试batch，每个batch有10000个图像。测试batch包含来自每个类1000个随机选择的图像。训练batch按照随机顺序包含剩余的图像，但是一些训练batch可能包含比另一个更多的图像。在他们之间，训练的batch包含每个类别正好5000张图片。 
![这里写图片描述](http://img.blog.csdn.net/20180121135605029?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
下表是数据集文件内部的结构,如上所说,有五个训练batch和一个测试batch:
|文件名称|大小|说明|
| :---: |:---:| :---:| 
|test_batch|31.0M|10000个测试图像|
|data_batch_1|31.0M|10000个训练图像|
|data_batch_2|31.0M|10000个训练图像|
|data_batch_3|31.0M|10000个训练图像|
|data_batch_4|31.0M|10000个训练图像|
|data_batch_5|31.0M|10000个训练图像|

整个文件的说明:
|文件名称|大小|说明|
| :---: |:---:| :---:| 
|cifar-10-python.tar.gz|170.5M|10个类别,60000个彩色图像,其中50000个训练图像,10000个测试图像|

同样的,在训练时开发者不需要单独去下载该数据集,PaddlePaddle已经帮我们封装好了,在我们调用paddle.dataset.cifar的时候,会自动在下载到缓存目录/home/username/.cache/paddle/dataset/cifar下,当以后再使用的时候,可以直接在缓存中获取,就不会去下载了.

# 开始训练模型
--------
## 定义神经网络模型
我们这里使用的是VGG神经网络，这个模型是牛津大学VGG(Visual Geometry Group)组在2014年ILSVRC提出的，VGG神经模型的核心是五组卷积操作，每两组之间做Max-Pooling空间降维。同一组内采用多次连续的3X3卷积，卷积核的数目由较浅组的64增多到最深组的512，同一组内的卷积核数目是一样的。卷积之后接两层全连接层，之后是分类层。由于每组内卷积层的不同，有11、13、16、19层这几种模型，在本章文章中使用到的是VGG16。VGG神经网络也是在ImageNet上首次公开超过人眼识别的模型。

这个VGG不是原来设的VGG神经模型,由于CIFAR10图片大小和数量相比ImageNet数据小很多，因此这里的模型针对CIFAR10数据做了一定的适配，卷积部分引入了`BN`层和`Dropout`操作。`conv_with_batchnorm`可以设置是否说使用`BN`层。`BN`层全称为：`Batch Normalization`，在没有使用`BN`层之前：

 - 参数的更新，使得每层的输入输出分布发生变化，称作ICS（Internal Covariate Shift）
 - 差异hui会随着网络深度增大而增大
 - 需要更小的学习率和较好的参数进行初始化

加入了`BN`层之后：
 
  - 可以使用较大的学习率
  - 可以减少对参数初始化的依赖
  - 可以拟制梯度的弥散
  - 可以起到正则化的作用
  - 可以加速模型收敛速度
 
以下就是`vgg.py`的文件中定义VGG神经网络模型的Python代码：
```python
# coding=utf-8
import paddle.v2 as paddle

# ***********************定义VGG卷积神经网络模型***************************************
def vgg_bn_drop(datadim):
    # 获取输入数据大小
    img = paddle.layer.data(name="image",
                            type=paddle.data_type.dense_vector(datadim))

    def conv_block(ipt, num_filter, groups, dropouts, num_channels=None):
        return paddle.networks.img_conv_group(
            input=ipt,
            num_channels=num_channels,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act=paddle.activation.Relu(),
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type=paddle.pooling.Max())

    conv1 = conv_block(img, 64, 2, [0.3, 0], 3)
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = paddle.layer.dropout(input=conv5, dropout_rate=0.5)
    fc1 = paddle.layer.fc(input=drop, size=512, act=paddle.activation.Linear())
    bn = paddle.layer.batch_norm(input=fc1,
                                 act=paddle.activation.Relu(),
                                 layer_attr=paddle.attr.Extra(drop_rate=0.5))
    fc2 = paddle.layer.fc(input=bn, size=512, act=paddle.activation.Linear())

    # 通过神经网络模型再使用Softmax获得分类器(全连接)
    out = paddle.layer.fc(input=fc2,
                          size=10,
                          act=paddle.activation.Softmax())
    return out
```

然后创建一个`train.py`的Python文件来编写训练的代码
## 导入依赖包
首先要先导入依赖包,其中就包含了最重要的PaddlePaddle的V2包
```python
# coding:utf-8
import sys
import paddle.v2 as paddle
from PIL import Image
import numpy as np
import os
```
## 初始化Paddle
然后我们创建一个类,再在类中创建一个初始化函数,在初始化函数中来初始化我们的PaddlePaddle
```python
class TestCIFAR:
    # ***********************初始化操作***************************************
    def __init__(self):
        # 初始化paddpaddle,只是用CPU,把GPU关闭
        paddle.init(use_gpu=False, trainer_count=2)
```

## 获取参数
训练参数可以通过使用损失函数创建一个训练参数，也可以通过使用之前训练好的参数初始化训练参数，使用训练好的参数来初始化训练参数，不仅可以使用之前的训练好的参数作为在此之上再继续训练，而且在某种情况下还防止出现浮点异常，比如SSD神经网络很容易出现浮点异常，就可以使用预训练的参数作为初始化训练参数，来解决出现浮点异常的问题。

该函数可以通过输入是否是参数文件路径，或者是损失函数，如果是参数文件路径，就使用之前训练好的参数生产参数。如果不传入参数文件路径,那就使用传入的损失函数生成参数。
```python
# **********************获取参数***************************************
def get_parameters(self, parameters_path=None, cost=None):
    if not parameters_path:
        # 使用cost创建parameters
        if not cost:
            print "请输入cost参数"
        else:
            # 根据损失函数创建参数
            parameters = paddle.parameters.create(cost)
            return parameters
    else:
        # 使用之前训练好的参数
        try:
            # 使用训练好的参数
            with open(parameters_path, 'r') as f:
                parameters = paddle.parameters.Parameters.from_tar(f)
            return parameters
        except Exception as e:
            raise NameError("你的参数文件错误,具体问题是:%s" % e)
```
## 创建训练器
创建训练器要3个参数,分别是损失函数,参数,优化方法.通过图像的标签信息和分类器生成损失函数。
参数可以选择是使用之前训练好的参数,然后在此基础上再进行训练,又或者是使用损失函数生成初始化参数。
然后再生成优化方法.就可以创建一个训练器了.
```python
# ***********************获取训练器***************************************
def get_trainer(self):
    # 数据大小
    datadim = 3 * 32 * 32

    # 获得图片对于的信息标签
    lbl = paddle.layer.data(name="label",
                            type=paddle.data_type.integer_value(10))

    # 获取全连接层,也就是分类器
    out = vgg_bn_drop(datadim=datadim)

    # 获得损失函数
    cost = paddle.layer.classification_cost(input=out, label=lbl)

    # 使用之前保存好的参数文件获得参数
    # parameters = self.get_parameters(parameters_path="../model/model.tar")
    # 使用损失函数生成参数
    parameters = self.get_parameters(cost=cost)

    '''
    定义优化方法
    learning_rate 迭代的速度
    momentum 跟前面动量优化的比例
    regularzation 正则化,防止过拟合
    '''
    momentum_optimizer = paddle.optimizer.Momentum(
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
        learning_rate=0.1 / 128.0,
        learning_rate_decay_a=0.1,
        learning_rate_decay_b=50000 * 100,
        learning_rate_schedule="discexp")

    '''
    创建训练器
    cost 分类器
    parameters 训练参数,可以通过创建,也可以使用之前训练好的参数
    update_equation 优化方法
    '''
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=momentum_optimizer)
    return trainer
```
## 开始训练
要启动训练要4个参数,分别是训练数据,训练的轮数,训练过程中的事件处理,输入数据和标签的对应关系.
训练数据:PaddlePaddle已经有封装好的API,可以直接获取CIFAR的数据.
训练轮数:表示我们要训练多少轮,次数越多准确率越高,最终会稳定在一个固定的准确率上.不得不说的是这个会比MNIST数据集的速度慢很多
事件处理:训练过程中的一些事件处理,比如会在每个batch打印一次日志,在每个pass之后保存一下参数和测试一下测试数据集的预测准确率.
输入数据和标签的对应关系:说明输入数据是第0维度,标签是第1维度
```python
# ***********************开始训练***************************************
def start_trainer(self):
    # 获得数据
    reader = paddle.batch(reader=paddle.reader.shuffle(reader=paddle.dataset.cifar.train10(),
                                                       buf_size=50000),
                          batch_size=128)

    # 指定每条数据和padd.layer.data的对应关系
    feeding = {"image": 0, "label": 1}

    # 定义训练事件
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

        # 每一轮训练完成之后
        if isinstance(event, paddle.event.EndPass):
            # 保存训练好的参数
            model_path = '../model'
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            with open(model_path + '/model.tar', 'w') as f:
                trainer.save_parameter_to_tar(f)

            # 测试准确率
            result = trainer.test(reader=paddle.batch(reader=paddle.dataset.cifar.test10(),
                                                      batch_size=128),
                                  feeding=feeding)
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

    # 获取训练器
    trainer = self.get_trainer()

    '''
    开始训练
    reader 训练数据
    num_passes 训练的轮数
    event_handler 训练的事件,比如在训练的时候要做一些什么事情
    feeding 说明每条数据和padd.layer.data的对应关系
    '''
    trainer.train(reader=reader,
                  num_passes=100,
                  event_handler=event_handler,
                  feeding=feeding)
```
然后在`main`入口中调用该函数就可以开始训练了
```python
if __name__ == '__main__':
    testCIFAR = TestCIFAR()
    # 开始训练
    testCIFAR.start_trainer()
```
在训练过程中会输出这样的日志:
```
Pass 0, Batch 0, Cost 2.427227, {'classification_error_evaluator': 0.8984375}
...................................................................................................
Pass 0, Batch 100, Cost 2.115308, {'classification_error_evaluator': 0.78125}
...................................................................................................
Pass 0, Batch 200, Cost 2.081666, {'classification_error_evaluator': 0.8359375}
...................................................................................................
Pass 0, Batch 300, Cost 1.866330, {'classification_error_evaluator': 0.734375}
..........................................................................................
Test with Pass 0, {'classification_error_evaluator': 0.8687999844551086}
```
我们还可以使用PaddlePaddle提供的可视化日志输出接口`paddle.v2.plot`，以折线图的方式显示`Train cost`和`Test cost`，不过这个程序要在jupyter笔记本上运行，代码已在`train.ipynb`中提供。折线图如下，这张图是训练的56个pass之后的收敛情况。这个过程笔者为了使训练速度更快，笔者使用了2个GPU进行训练，训练56个pass共消耗6个小时，几乎已经完全收敛了：
![这里写图片描述](http://img.blog.csdn.net/20180225092404412?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

此时它测试输出的日志如下，可以看到预测错误率为`0.1477999985218048`：
```text
Test with Pass 56, {'classification_error_evaluator': 0.1477999985218048}
```


# 使用参数预测
-------
编写一个`infer.py`的Python程序文件编写下面的代码，用于测试数据。

在PaddlePaddle使用之前，都要初始化PaddlePaddle。
```python
def __init__(self):
    # 初始化paddpaddle,只是用CPU,把GPU关闭
    paddle.init(use_gpu=False, trainer_count=2)
```
然后加载训练是保存的模型，从保存的模型文件中读取模型参数。
```python
def get_parameters(self, parameters_path):
    with open(parameters_path, 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)
    return parameters

```
该函数需要输入3个参数：

 - 第一个是需要预测的图像,图像传入之后,会经过`load_image`函数处理,大小会变成32*32大小,训练是输入数据的大小一样.
 - 第二个就是训练好的参数
 - 第三个是通过神经模型生成的分类器
```python
def to_prediction(self, image_path, parameters, out):
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

    # 获得要预测的图片
    test_data = []
    test_data.append((load_image(image_path),))

    # 获得预测结果
    probs = paddle.infer(output_layer=out,
                         parameters=parameters,
                         input=test_data)
    # 处理预测结果
    lab = np.argsort(-probs)
    # 返回概率最大的值和其对应的概率值
    return lab[0][0], probs[0][(lab[0][0])]
```
在`main`入口中调用预测函数
```python
if __name__ == '__main__':
    testCIFAR = TestCIFAR()
    # 开始预测
    out = testCIFAR.get_out(3 * 32 * 32)
    parameters = testCIFAR.get_parameters("../model/model.tar")
    image_path = "../images/airplane1.png"
    result,probability = testCIFAR.to_prediction(image_path=image_path, out=out, parameters=parameters)
    print '预测结果为:%d,可信度为:%f' % (result,probability)
```
输出的预测结果是:
```
预测结果为:0,可信度为:0.965155
```

# 使用其他神经模型
---------
在上面的训练中,只是使用到了VGG神经模型,而目前的ResNet可以说最火的,因为该神经模型可以通过增加网络的深度达到提高识别率,而不会像其他过去的神经模型那样,当网络继续加深时,反而会损失精度.ResNet神经网络`resnet.py`定义如下：
```python
# coding=utf-8
import paddle.v2 as paddle

# ***********************定义ResNet卷积神经网络模型***************************************
def resnet_cifar10(datadim,depth=32):
    # 获取输入数据大小
    ipt = paddle.layer.data(name="image",
                              type=paddle.data_type.dense_vector(datadim))


    def conv_bn_layer(input, ch_out, filter_size, stride, padding, active_type=paddle.activation.Relu(),
                      ch_in=None):
        tmp = paddle.layer.img_conv(input=input,
                                    filter_size=filter_size,
                                    num_channels=ch_in,
                                    num_filters=ch_out,
                                    stride=stride,
                                    padding=padding,
                                    act=paddle.activation.Linear(),
                                    bias_attr=False)
        return paddle.layer.batch_norm(input=tmp, act=active_type)

    def shortcut(ipt, n_in, n_out, stride):
        if n_in != n_out:
            return conv_bn_layer(ipt, n_out, 1, stride, 0, paddle.activation.Linear())
        else:
            return ipt

    def basicblock(ipt, ch_out, stride):
        ch_in = ch_out * 2
        tmp = conv_bn_layer(ipt, ch_out, 3, stride, 1)
        tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, paddle.activation.Linear())
        short = shortcut(ipt, ch_in, ch_out, stride)
        return paddle.layer.addto(input=[tmp, short],
                                  act=paddle.activation.Relu())

    def layer_warp(block_func, ipt, features, count, stride):
        tmp = block_func(ipt, features, stride)
        for i in range(1, count):
            tmp = block_func(tmp, features, 1)
        return tmp

    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    nStages = {16, 64, 128}
    conv1 = conv_bn_layer(ipt, ch_in=3, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 64, n, 2)
    pool = paddle.layer.img_pool(
        input=res3, pool_size=8, stride=1, pool_type=paddle.pooling.Avg())

    # 通过神经网络模型再使用Softmax获得分类器(全连接)
    out = paddle.layer.fc(input=pool,
                          size=10,
                          act=paddle.activation.Softmax())
    return out
```
如果要使用上面的残差神经网络，只要把这行代码：
```python
out = vgg_bn_drop(datadim=datadim)
```
换成中残差神经网络中获取分类器就可以了：
```python
 out = resnet_cifar10(datadim=datadim)
```


# 所有代码
--------
为了让读者更直观阅读代码，这张贴出所有的代码。笔者这也代码同步到GitHub上，GitHub的地址章文章的最后，读者可以clone代码到自己的电脑阅读。
`vgg.py`，VGG16神经网络的代码：
```python
# coding=utf-8
import paddle.v2 as paddle

# ***********************定义VGG卷积神经网络模型***************************************
def vgg_bn_drop(datadim):
    # 获取输入数据大小
    img = paddle.layer.data(name="image",
                            type=paddle.data_type.dense_vector(datadim))

    def conv_block(ipt, num_filter, groups, dropouts, num_channels=None):
        return paddle.networks.img_conv_group(
            input=ipt,
            num_channels=num_channels,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act=paddle.activation.Relu(),
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type=paddle.pooling.Max())

    conv1 = conv_block(img, 64, 2, [0.3, 0], 3)
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = paddle.layer.dropout(input=conv5, dropout_rate=0.5)
    fc1 = paddle.layer.fc(input=drop, size=512, act=paddle.activation.Linear())
    bn = paddle.layer.batch_norm(input=fc1,
                                 act=paddle.activation.Relu(),
                                 layer_attr=paddle.attr.Extra(drop_rate=0.5))
    fc2 = paddle.layer.fc(input=bn, size=512, act=paddle.activation.Linear())

    # 通过神经网络模型再使用Softmax获得分类器(全连接)
    out = paddle.layer.fc(input=fc2,
                          size=10,
                          act=paddle.activation.Softmax())
    return out
```

`resnet.py`，残差神经网络的代码：
```python
# coding=utf-8
import paddle.v2 as paddle

# ***********************定义ResNet卷积神经网络模型***************************************
def resnet_cifar10(datadim,depth=32):
    # 获取输入数据大小
    ipt = paddle.layer.data(name="image",
                              type=paddle.data_type.dense_vector(datadim))


    def conv_bn_layer(input, ch_out, filter_size, stride, padding, active_type=paddle.activation.Relu(),
                      ch_in=None):
        tmp = paddle.layer.img_conv(input=input,
                                    filter_size=filter_size,
                                    num_channels=ch_in,
                                    num_filters=ch_out,
                                    stride=stride,
                                    padding=padding,
                                    act=paddle.activation.Linear(),
                                    bias_attr=False)
        return paddle.layer.batch_norm(input=tmp, act=active_type)

    def shortcut(ipt, n_in, n_out, stride):
        if n_in != n_out:
            return conv_bn_layer(ipt, n_out, 1, stride, 0, paddle.activation.Linear())
        else:
            return ipt

    def basicblock(ipt, ch_out, stride):
        ch_in = ch_out * 2
        tmp = conv_bn_layer(ipt, ch_out, 3, stride, 1)
        tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, paddle.activation.Linear())
        short = shortcut(ipt, ch_in, ch_out, stride)
        return paddle.layer.addto(input=[tmp, short],
                                  act=paddle.activation.Relu())

    def layer_warp(block_func, ipt, features, count, stride):
        tmp = block_func(ipt, features, stride)
        for i in range(1, count):
            tmp = block_func(tmp, features, 1)
        return tmp

    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    nStages = {16, 64, 128}
    conv1 = conv_bn_layer(ipt, ch_in=3, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 64, n, 2)
    pool = paddle.layer.img_pool(
        input=res3, pool_size=8, stride=1, pool_type=paddle.pooling.Avg())

    # 通过神经网络模型再使用Softmax获得分类器(全连接)
    out = paddle.layer.fc(input=pool,
                          size=10,
                          act=paddle.activation.Softmax())
    return out
```

`train.py`，训练模型的代码：
```python
# coding:utf-8
import os
import sys
import paddle.v2 as paddle
from vgg import vgg_bn_drop
from resnet import resnet_cifar10

class TestCIFAR:
    # ***********************初始化操作***************************************
    def __init__(self):
        # 初始化paddpaddle,只是用CPU,把GPU关闭
        paddle.init(use_gpu=False, trainer_count=2)

    # **********************获取参数***************************************
    def get_parameters(self, parameters_path=None, cost=None):
        if not parameters_path:
            # 使用cost创建parameters
            if not cost:
                print "请输入cost参数"
            else:
                # 根据损失函数创建参数
                parameters = paddle.parameters.create(cost)
                return parameters
        else:
            # 使用之前训练好的参数
            try:
                # 使用训练好的参数
                with open(parameters_path, 'r') as f:
                    parameters = paddle.parameters.Parameters.from_tar(f)
                return parameters
            except Exception as e:
                raise NameError("你的参数文件错误,具体问题是:%s" % e)

    # ***********************获取训练器***************************************
    def get_trainer(self):
        # 数据大小
        datadim = 3 * 32 * 32

        # 获得图片对于的信息标签
        lbl = paddle.layer.data(name="label",
                                type=paddle.data_type.integer_value(10))

        # 获取全连接层,也就是分类器
        #
        out = vgg_bn_drop(datadim=datadim)
        # out = resnet_cifar10(datadim=datadim)

        # 获得损失函数
        cost = paddle.layer.classification_cost(input=out, label=lbl)

        # 使用之前保存好的参数文件获得参数
        # parameters = self.get_parameters(parameters_path="../model/model.tar")
        # 使用损失函数生成参数
        parameters = self.get_parameters(cost=cost)

        '''        定义优化方法
        learning_rate 迭代的速度
        momentum 跟前面动量优化的比例
        regularzation 正则化,防止过拟合
        '''
        momentum_optimizer = paddle.optimizer.Momentum(
            momentum=0.9,
            regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
            learning_rate=0.1 / 128.0,
            learning_rate_decay_a=0.1,
            learning_rate_decay_b=50000 * 100,
            learning_rate_schedule="discexp")

        '''
        创建训练器
        cost 分类器
        parameters 训练参数,可以通过创建,也可以使用之前训练好的参数
        update_equation 优化方法
        '''
        trainer = paddle.trainer.SGD(cost=cost,
                                     parameters=parameters,
                                     update_equation=momentum_optimizer)
        return trainer

    # ***********************开始训练***************************************
    def start_trainer(self):
        # 获得数据
        reader = paddle.batch(reader=paddle.reader.shuffle(reader=paddle.dataset.cifar.train10(),
                                                           buf_size=50000),
                              batch_size=128)

        # 指定每条数据和padd.layer.data的对应关系
        feeding = {"image": 0, "label": 1}

        # 定义训练事件，输出日志
        def event_handler(event):
            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 100 == 0:
                    print "\nPass %d, Batch %d, Cost %f, %s" % (
                        event.pass_id, event.batch_id, event.cost, event.metrics)
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()

            # 每一轮训练完成之后
            if isinstance(event, paddle.event.EndPass):
                # 保存训练好的参数
                model_path = '../model'
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                with open(model_path + '/model.tar', 'w') as f:
                    trainer.save_parameter_to_tar(f)

                # 测试准确率
                result = trainer.test(reader=paddle.batch(reader=paddle.dataset.cifar.test10(),
                                                          batch_size=128),
                                      feeding=feeding)
                print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

        # 获取训练器
        trainer = self.get_trainer()

        '''
        开始训练
        reader 训练数据
        num_passes 训练的轮数
        event_handler 训练的事件,比如在训练的时候要做一些什么事情
        feeding 说明每条数据和padd.layer.data的对应关系
        '''
        trainer.train(reader=reader,
                      num_passes=100,
                      event_handler=event_handler,
                      feeding=feeding)

if __name__ == '__main__':
    testCIFAR = TestCIFAR()
    # 开始训练
    testCIFAR.start_trainer()
```

`train.ipynb`，在jupyter中使用的代码，会输出训练时cost的折线图：
```python
# coding:utf-8
import os
import sys
import paddle.v2 as paddle
from paddle.v2.plot import Ploter

from vgg import vgg_bn_drop

step = 0

class TestCIFAR:
    # ***********************初始化操作***************************************
    def __init__(self):
        # 初始化paddpaddle,只是用CPU,把GPU关闭
        paddle.init(use_gpu=False, trainer_count=2)

    # **********************获取参数***************************************
    def get_parameters(self, parameters_path=None, cost=None):
        if not parameters_path:
            # 使用cost创建parameters
            if not cost:
                print "请输入cost参数"
            else:
                # 根据损失函数创建参数
                parameters = paddle.parameters.create(cost)
                return parameters
        else:
            # 使用之前训练好的参数
            try:
                # 使用训练好的参数
                with open(parameters_path, 'r') as f:
                    parameters = paddle.parameters.Parameters.from_tar(f)
                return parameters
            except Exception as e:
                raise NameError("你的参数文件错误,具体问题是:%s" % e)

    # ***********************获取训练器***************************************
    def get_trainer(self):
        # 数据大小
        datadim = 3 * 32 * 32

        # 获得图片对于的信息标签
        lbl = paddle.layer.data(name="label",
                                type=paddle.data_type.integer_value(10))

        # 获取全连接层,也就是分类器
        out = vgg_bn_drop(datadim=datadim)

        # 获得损失函数
        cost = paddle.layer.classification_cost(input=out, label=lbl)

        # 使用之前保存好的参数文件获得参数
        # parameters = self.get_parameters(parameters_path="../model/model.tar")
        # 使用损失函数生成参数
        parameters = self.get_parameters(cost=cost)

        '''
        定义优化方法
        learning_rate 迭代的速度
        momentum 跟前面动量优化的比例
        regularzation 正则化,防止过拟合
        '''
        momentum_optimizer = paddle.optimizer.Momentum(
            momentum=0.9,
            regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
            learning_rate=0.1 / 128.0,
            learning_rate_decay_a=0.1,
            learning_rate_decay_b=50000 * 100,
            learning_rate_schedule="discexp")

        '''
        创建训练器
        cost 分类器
        parameters 训练参数,可以通过创建,也可以使用之前训练好的参数
        update_equation 优化方法
        '''
        trainer = paddle.trainer.SGD(cost=cost,
                                     parameters=parameters,
                                     update_equation=momentum_optimizer)
        return trainer

    # ***********************开始训练***************************************
    def start_trainer(self):
        # 获得数据
        reader = paddle.batch(reader=paddle.reader.shuffle(reader=paddle.dataset.cifar.train10(),
                                                           buf_size=50000),
                              batch_size=128)

        # 指定每条数据和padd.layer.data的对应关系
        feeding = {"image": 0, "label": 1}

        train_title = "Train cost"
        test_title = "Test cost"
        cost_ploter = Ploter(train_title, test_title)

        # 定义训练事件,画出折线图,该事件的图可以在notebook上显示，命令行不会正常输出
        def event_handler_plot(event):
            global step
            if isinstance(event, paddle.event.EndIteration):
                if step % 1 == 0:
                    cost_ploter.append(train_title, step, event.cost)
                    cost_ploter.plot()
                step += 1
            if isinstance(event, paddle.event.EndPass):
                # 保存训练好的参数
                model_path = '../model'
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                with open(model_path + '/model_%d.tar' % event.pass_id, 'w') as f:
                    trainer.save_parameter_to_tar(f)

                result = trainer.test(
                    reader=paddle.batch(
                        paddle.dataset.cifar.test10(), batch_size=128),
                    feeding=feeding)
                cost_ploter.append(test_title, step, result.cost)

        # 获取训练器
        trainer = self.get_trainer()

        '''
        开始训练
        reader 训练数据
        num_passes 训练的轮数
        event_handler 训练的事件,比如在训练的时候要做一些什么事情
        feeding 说明每条数据和padd.layer.data的对应关系
        '''
        trainer.train(reader=reader,
                      num_passes=100,
                      event_handler=event_handler_plot,
                      feeding=feeding)

if __name__ == '__main__':
    testCIFAR = TestCIFAR()
    # 开始训练
    testCIFAR.start_trainer()
```

`infer.py`，使用训练好的模型预测数据的代码：
```python
# coding:utf-8
from paddle.v2.plot import Ploter
import sys
import paddle.v2 as paddle
from PIL import Image
import numpy as np
import os
from vgg import vgg_bn_drop

class TestCIFAR:
    # ***********************初始化操作***************************************
    def __init__(self):
        # 初始化paddpaddle,只是用CPU,把GPU关闭
        paddle.init(use_gpu=False, trainer_count=2)

    # **********************获取参数***************************************
    def get_parameters(self, parameters_path):
        with open(parameters_path, 'r') as f:
            parameters = paddle.parameters.Parameters.from_tar(f)
        return parameters

    # ***********************使用训练好的参数进行预测***************************************
    def to_prediction(self, image_path, parameters, out):
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

        # 获得要预测的图片
        test_data = []
        test_data.append((load_image(image_path),))

        # 获得预测结果
        probs = paddle.infer(output_layer=out,
                             parameters=parameters,
                             input=test_data)
        # 处理预测结果
        lab = np.argsort(-probs)
        # 返回概率最大的值和其对应的概率值
        return lab[0][0], probs[0][(lab[0][0])]


if __name__ == '__main__':
    testCIFAR = TestCIFAR()
    # 开始预测
    out = vgg_bn_drop(3 * 32 * 32)
    parameters = testCIFAR.get_parameters("../model/model.tar")
    image_path = "../images/airplane1.png"
    result,probability = testCIFAR.to_prediction(image_path=image_path, out=out, parameters=parameters)
    print '预测结果为:%d,可信度为:%f' % (result,probability)
```



# 项目代码
-----
GitHub地址:[https://github.com/yeyupiaoling/LearnPaddle](https://github.com/yeyupiaoling/LearnPaddle)

# 参考资料
---------
1. http://paddlepaddle.org/
2. https://www.cs.toronto.edu/~kriz/cifar.html
3. https://arxiv.org/abs/1405.3531
