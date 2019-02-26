# 目录
@[toc]
*本篇文章基于 PaddlePaddle 0.13.0、Python 2.7
*Fluid版本的使用可以学习笔者的新系列教程：[《PaddlePaddle从入门到炼丹》](https://blog.csdn.net/qq_33200967/column/info/28685)
# 前言
------
PaddlePaddle的Fluid是0.11.0提出的，Fluid 是设计用来让用户像Pytorch和Tensorflow Eager Execution一样执行程序。在这些系统中，不再有模型这个概念，应用也不再包含一个用于描述Operator图或者一系列层的符号描述，而是像通用程序那样描述训练或者预测的过程。而Fluid与PyTorch或Eager Execution的区别在于Fluid不依赖Python提供的控制流，例如 if-else-then或者for，而是提供了基于C++实现的控制流并暴露了对应的用with语法实现的Python接口。例如我们会在例子中使用到的代码片段：
```python
with fluid.program_guard(inference_program):
    test_accuracy = fluid.evaluator.Accuracy(input=out, label=label)
    test_target = [avg_cost] + test_accuracy.metrics + test_accuracy.states
    inference_program = fluid.io.get_inference_program(test_target)
```
在Fluid版本中，不再使用`trainer`来训练和测试模型了，而是使用了一个C++类`Executor`用于运行一个Fluid程序，`Executor`类似一个解释器，Fluid将会使用这样一个解析器来训练和测试模型，如：
```python
loss, acc = exe.run(fluid.default_main_program(),
                    feed=feeder.feed(data),
                    fetch_list=[avg_cost] + accuracy.metrics)
```
对于这个Fluid版本，我们在此之前都没有使用过，那么接下来就让我们去使用Fluid版本，同时对比一下之前所写的，探讨Fluid版本的改变。

# 训练模型
------
## 定义神经网络
我们这次使用的是比较熟悉的VGG16神经模型，这个模型在之前的[CIFAR彩色图像识别](http://blog.csdn.net/qq_33200967/article/details/79095224)，为了方便比较，我们也是使用CIFAR10数据集，以下代码就是Paddle 1和Fluid版本的VGG16的定义，把它们都拿出来对比，看看Fluid版本的改动。

通过对比这个两神经网络的定义可以看到`img_conv_group`的接口位置已经不一样了，Fluid的相关接口都在`fluid`下了。同时我们看到改变最大的是Fluid取消了`num_channels`图像的通道数。

在Fluid版本中使用的激活函数不再是调用一个函数了，而是传入一个字符串，比如在BN层指定一个Relu激活函数`act='relu'`，在Paddle 1版本中是这样的：`act=paddle.activation.Relu()`

Paddle 1的VGG16
```python
def vgg_bn_drop(input,class_dim):
    # 定义卷积块
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
    # 定义一个VGG16的卷积组
    conv1 = conv_block(input, 64, 2, [0.3, 0], 3)
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])
    # 定义第一个drop层
    drop = paddle.layer.dropout(input=conv5, dropout_rate=0.5)
    # 定义第一层全连接层
    fc1 = paddle.layer.fc(input=drop, size=512, act=paddle.activation.Linear())
    # 定义BN层
    bn = paddle.layer.batch_norm(input=fc1,
                                 act=paddle.activation.Relu(),
                                 layer_attr=paddle.attr.Extra(drop_rate=0.5))
    # 定义第二层全连接层
    fc2 = paddle.layer.fc(input=bn, size=512, act=paddle.activation.Linear())
    # 获取全连接输出，获得分类器
    predict = paddle.layer.fc(input=fc2,
                          size=class_dim,
                          act=paddle.activation.Softmax())
    return predict
```
Fluid版本的VGG16
```python
def vgg16_bn_drop(input):
    # 定义卷积块
    def conv_block(input, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=input,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')
    # 定义一个VGG16的卷积组
    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])
    # 定义第一个drop层
    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    # 定义第一层全连接层
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    # 定义BN层
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    # 定义第二层全连接层
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    # 定义第二层全连接层
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    # 获取全连接输出，获得分类器
    predict = fluid.layers.fc(
        input=fc2,
        size=class_dim,
        act='softmax',
        param_attr=ParamAttr(name="param1", initializer=NormalInitializer()))
    return predict

```
通过上面获取的全连接，可以生成一个分类器
```python
# 定义图像的类别数量
class_dim = 10
# 获取神经网络的分类器
predict = vgg16_bn_drop(image,class_dim)
```
## 定义数据
在数据定义方式上，Fluid和之前的Paddle 1定义方式有了很大的差别，比如不再是根据图像的大小定义的，而是传图像的形状，包括通道数，同时指定数据的类型。

Fluid版本的定义方式
```python
# 定义图像的通道数和大小
image_shape = [3, 32, 32]
# 定义输入数据大小，指定图像的形状，数据类型是浮点型
image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
# 定义标签，类型是整型
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
```
Paddle 1的定义方式
```python
# 获取输入数据模式
image = paddle.layer.data(name="image",
                          type=paddle.data_type.dense_vector(datadim))
# 获得图片对于的信息标签
label = paddle.layer.data(name="label",
                          type=paddle.data_type.integer_value(type_size))
```
## 定义batch平均错误
在Fluid版本中，多了一个`batch_acc`的程序，这个是在训练过程或者是测试中计算平均错误率的。这个需要定义在优化方法之前。
```python
# 每个batch计算的时候能取到当前batch里面样本的个数，从而来求平均的准确率
batch_size = fluid.layers.create_tensor(dtype='int64')
print batch_size
batch_acc = fluid.layers.accuracy(input=predict, label=label, total=batch_size)
```

## 定义测试程序
这个一个定义预测的一个程序，这个是在主程序中获取的一个程序，专门用来做测试的，这个定义要放在定义方法之前，因为测试程序是训练程序的前半部分（不包括优化器和backward），所以要定义在优化方法之前。
```python
# 测试程序
inference_program = fluid.default_main_program().clone(for_test=True)
```

## 定义优化方法
在优化方法的定义上也有很大的不同，Fluid把`learning_rate`相关的都放在一起了，以下是两个优化方法的定义，这不是本章项目使用到的`optimizer`，本章使用的`optimizer`比较简单，差别不大。
Fluid版本的定义优化方法
```python
optimizer = fluid.optimizer.Momentum(
    learning_rate=fluid.layers.exponential_decay(
        learning_rate=learning_rate,
        decay_steps=40000,
        decay_rate=0.1,
        staircase=True),
    momentum=0.9,
    regularization=fluid.regularizer.L2Decay(0.0005), )
opts = optimizer.minimize(loss)
```
Paddle 1版本的定义优化方法
```python
momentum_optimizer = paddle.optimizer.Momentum(
    momentum=0.9,
    regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
    learning_rate=0.1 / 128.0,
    learning_rate_decay_a=0.1,
    learning_rate_decay_b=50000 * 100,
    learning_rate_schedule='discexp')
```
## 测试和训练
### 定义调试器
在前言有讲到，在Fluid版本中，不会在有`trainer`了，Paddle 1用 `trainer.train(...)`，Fluid用`fluid.Executor(place).Run(...)`，所以在Fluid起关键作用的是调试器
```python
# 是否使用GPU
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
# 创建调试器
exe = fluid.Executor(place)
# 初始化调试器
exe.run(fluid.default_startup_program())
```
如果要指定GPU个数和编号的话，可以在终端输入以下命令：
```text
export CUDA_VISIBLE_DEVICES=0,1
```
使用上面的方法如果是换一个终端，就没有上面的效果了，如果想设计持久化，就要在`~/.bashrc`的最后加上以下代码：
```text
cudaid=${cudaid_num:=0,1}
export CUDA_VISIBLE_DEVICES=$cudaid
```

### 获取数据
在读取数据成`reader`上没有什么区别，这要说的是`feeder`，这这里定义的更之前的` feeding = {"image": 0, "label": 1}`差距有点大了。不过这样看起了更加明了。
```python 
# 获取训练数据
train_reader = paddle.batch(
        paddle.dataset.cifar.train10(), batch_size=BATCH_SIZE)
# 获取测试数据
test_reader = paddle.batch(
        paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE)

# 指定数据和label的对于关系
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
```
### 开始训练和测试
在这里就有很大的不一样了，在Paddle 1中，使用的是`trainer`，通过`num_passes`来指定训练的Pass，而Fluid的是使用一个循环来处理的，这样就大大方便了我们在训练过程中所做的一些操作了，而在此之前是使用一个`event`训练时间的，虽然也可以做到一些操作，不过相对循环来说，笔者还是觉得循环用起来比较方便。
```python
accuracy = fluid.average.WeightedAverage()
test_accuracy = fluid.average.WeightedAverage()
# 开始训练，使用循环的方式来指定训多少个Pass
for pass_id in range(num_passes):
# 从训练数据中按照一个个batch来读取数据
accuracy.reset()
for batch_id, data in enumerate(train_reader()):
    loss, acc, weight = exe.run(fluid.default_main_program(),
                        feed=feeder.feed(data),
                        fetch_list=[avg_cost, batch_acc, batch_size])
    accuracy.add(value=acc, weight=weight)
    print("Pass {0}, batch {1}, loss {2}, acc {3}".format(
        pass_id, batch_id, loss[0], acc[0]))

# 测试模型
test_accuracy.reset()
for data in test_reader():
    loss, acc, weight = exe.run(inference_program,
                        feed=feeder.feed(data),
                        fetch_list=[avg_cost, batch_acc, batch_size])
    test_accuracy.add(value=acc, weight=weight)


# 输出相关日志
pass_acc = accuracy.eval()
test_pass_acc = test_accuracy.eval()
print("End pass {0}, train_acc {1}, test_acc {2}".format(
    pass_id, pass_acc, test_pass_acc))

# 每一个Pass就保存一次模型
# 指定保存模型的路径
model_path = os.path.join(model_save_dir, str(pass_id))
# 如果保存路径不存在就创建
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print 'save models to %s' % (model_path)
# 保存模型
fluid.io.save_inference_model(model_path, ['image'], [predict], exe)

```
在训练过程中，会输出类型下的日志信息：
```text
Pass 0, batch 0, loss 16.5825138092, acc 0.09375
Pass 0, batch 1, loss 15.7055978775, acc 0.1484375
Pass 0, batch 2, loss 15.8206882477, acc 0.0546875
Pass 0, batch 3, loss 14.6004362106, acc 0.1953125
Pass 0, batch 4, loss 14.9484052658, acc 0.1171875
Pass 0, batch 5, loss 13.0915336609, acc 0.078125
```
### 保存预测模型
在Fluid版本中，保存模型虽然复杂一点点，但是对于之后的预测是极大的方便了，因为在预测中，不需要再定义神经网络模型了，可以直接使用保存好的模型进行预测。还有要说一下的是，这个保存模型的格式跟之前的不一样，这个保存模型是不会压缩的。
Fluid版本的保存模型
```python
# 指定保存模型的路径
model_path = os.path.join(model_save_dir, str(pass_id))
# 如果保存路径不存在就创建
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print 'save models to %s' % (model_path)
# 保存预测模型
fluid.io.save_inference_model(model_path, ['image'], [net], exe)
```
Paddle 1的保存模型
```python
with open(save_parameters_name, 'w') as f:
         trainer.save_parameter_to_tar(f)
```

# 预测模型
------
## 获取调试器
在预测中，以前的Paddle 1是要使用到预测器`infer`的，而在Fluid中还是使用调试器，定义调试器如下
```python
# 是否使用GPU
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
# 生成调试器
exe = fluid.Executor(place)
```
在预测中，所有的预测都要在这个控制流中执行
```python
inference_scope = fluid.core.Scope()
with fluid.scope_guard(inference_scope):
```
## 加载训练好的模型
加载模型，在这里，加载模型跟之前的差距也很大，在Paddle 1的是`parameters = paddle.parameters.Parameters.from_tar(f)`,因为之前使用的是参数，而在Fluid没有使用到参数这个概念。
```python
# 加载模型
[inference_program, feed_target_names,fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)
```
## 获取预测结果
获取预测数据
```python
# 获取预测数据
img = Image.open(image_file)
img = img.resize((32, 32), Image.ANTIALIAS)
test_data = np.array(img).astype("float32")
test_data = np.transpose(test_data, (2, 0, 1))
test_data = test_data[np.newaxis, :] / 255
```
开始预测并打印结果
```python
# 开始预测
results = exe.run(inference_program,
                  feed={feed_target_names[0]: test_data},
                  fetch_list=fetch_targets)

results = np.argsort(-results[0])
# 打印预测结果
print "The images/horse4.png infer results label is: ", results[0][0]

```
调用预测函数
```python
if __name__ == '__main__':
    image_file = '../images/horse4.png'
    model_path = '../models/0/'
    infer(image_file, False, model_path)
```
输出结果如下：
```text
The images/horse4.png infer results label is:  7
```
# 项目代码
------
GitHub地址:https://github.com/yeyupiaoling/LearnPaddle

# 参考资料
---------
 1. http://paddlepaddle.org/
 2. https://github.com/PaddlePaddle/Paddle/blob/develop/RELEASE.cn.md
