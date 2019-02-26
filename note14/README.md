# 目录
@[toc]
*本篇文章基于 PaddlePaddle 0.11.0、Python 2.7
# 前言
------
PaddlePaddle还可以迁移到Android或者Linux设备上，在这些部署了PaddlePaddle的设备同样可以做深度学习的预测。在这篇文章中我们就介绍如何把PaddlePaddle迁移到Android手机上，并在Android的APP中使用PaddlePaddle。

# 编译PaddlePaddle库
------
## 使用Docker编译PaddlePaddle库
使用Docker编译PaddlePaddle真的会很方便，如果你对比了下面部分的**使用Linux编译PaddlePaddle库**，你就会发现使用Docker会少很多麻烦，比如安装一些依赖库等等。而且Docker是跨平台的，不管读者使用的是Windows，Linux，还是Mac，都可以使用Docker。以下操作方法都是在64位的Ubuntu 16.04上实现的。
首先安装Docker，在Ubuntu上安装很简单，只要一条命令就可以了
```text
sudo apt install docker.io
```
安装完成之后，可以使用`docker --version`命令查看是否安装成功，如果安装成功会输出Docker的版本信息。

然后是在GitHub上克隆PaddlePaddle源码，命令如下：
```text
git clone https://github.com/PaddlePaddle/Paddle.git
```
克隆完成PaddlePaddle源码之后，就可以使用PaddlePaddle源码创建可以编译给Android使用的PaddlePaddle库的Docker容器了：
```python
# 切入到源码目录
cd Paddle
# 创建Docker容器
docker build -t mypaddle/paddle-android:dev . -f Dockerfile.android
```
### 可能会出现的问题
值得注意的是如果读者的电脑不能科学上网的，会在下载https://storage.googleapis.com/golang/go1.8.1.linux-amd64.tar.gz的时候报错，可以修改其下载路径。使用的文件是在Paddle目录下的`Dockerfile.android`，所以要在这个文件中修改，具体在25行，将其修改成https://dl.google.com/go/go1.8.1.linux-amd64.tar.gz，如果可以科学上网，就不必理会。
如果再不行就干脆去掉GO语言依赖，因为编译Android的PaddlePaddle库根本就不用GO语言依赖库，具体操作如下：
修改`Paddle/CMakeLists.txt`下的22行
```text
 project(paddle CXX C Go) 
```
去掉Go依赖，修改成如下
```text
project(paddle CXX C)
```
删除`Paddle/Dockerfile.android`的Go语言配置
```text
# Install Go and glide
RUN wget -qO- go.tgz https://storage.googleapis.com/golang/go1.8.1.linux-amd64.tar.gz | \
    tar -xz -C /usr/local && \
    mkdir /root/gopath && \
    mkdir /root/gopath/bin && \
    mkdir /root/gopath/src
ENV GOROOT=/usr/local/go GOPATH=/root/gopath
# should not be in the same line with GOROOT definition, otherwise docker build could not find GOROOT.
ENV PATH=${PATH}:${GOROOT}/bin:${GOPATH}/bin
```

### 使用官方的Docker容器
如果读者不想使用源码创建Docker容器，PaddlePaddle官方也提供了创建好的Docker容器，读者可以直接拉到本地就可以使用了，命令如下：
```text
docker pull paddlepaddle/paddle:latest-dev-android
```
以上是国外的镜像，如果pull的速度慢，可以使用国内的镜像
```text
docker pull docker.paddlepaddlehub.com/paddle:latest-dev-android
```
### 开始编译PaddlePaddle库
编译`armeabi-v7a`，`Android API 21`的PaddlePaddle库，命令如下，创建PaddlePaddle的配置可以使用`e`命令设置。在命令的最后可以看到使用的容器是我们自己创建的Docker容器`mypaddle/paddle-android:dev`，如果换成官方提供的，把Docker名称修改成`paddlepaddle/paddle:latest-dev-android`即可。
```text
docker run -it --rm -v $PWD:/paddle -e "ANDROID_ABI=armeabi-v7a" -e "ANDROID_API=21" mypaddle/paddle-android:dev
```
当编译完成之后，在`$PWD/install_android`目录下创建以下三个目录，`$PWD`表示当前目录，笔者当前目录为`/home/work/android/docker/`。这些文件就是我们之后在Android的APP上会使用的的文件：

 - `include`是C-API的头文件
 - `lib`是Android ABI的PaddlePaddle库
 - `third_party`是所依赖的所有第三方库

上面的是编译`armeabi-v7a`，`Android API 21`的PaddlePaddle库，如果读者想编译`arm64-v8a`，`Android API 21`的PaddlePaddle库，只要修改命令参数就可以了，具体命令如下：
```text
docker run -it --rm -v $PWD:/paddle -e "ANDROID_ABI=arm64-v8a" -e "ANDROID_API=21" mypaddle/paddle-android:dev
```

## 使用Linux编译PaddlePaddle库
如果读者不习惯与使用Docker，或者想进一步了解编译PaddlePaddle库的流程，想使用Linux编译PaddlePaddle库，这也是没有问题的，只是步骤比较复杂一些。

### 安装依赖环境
首先要安装编译的依赖库，安装`gcc 4.9`，命令如下。安装完成之后使用`gcc --version`查看安装是否安装成功。
```text
sudo apt-get install gcc-4.9
```
安装`clang 3.8`命令如下。同样安装完成之后使用`clang --version`查看安装是否安装成功
```text
apt install clang
```
安装GO语言环境
```text
apt-get install golang
```
安装`CMake`，最好安装版本为3.8以上的。首先下载`CMake`源码。
```text
wget https://cmake.org/files/v3.8/cmake-3.8.0.tar.gz
```
解压`CMake`源码
```text
tar -zxvf cmake-3.8.0.tar.gz
```
依次执行下面的代码
```python
# 进入解压后的目录
cd cmake-3.8.0
# 执行当前目录的bootstrap程序
./bootstrap
# make一下，使用12个线程
make -j12
# 开始安装
sudo make install
```
### 配置编译环境
下载`Android NDK`，`Android NDK`是Android平台上使用的C/C++交叉编译工具链，`Android NDK`中包含了所有Android API级别、所有架构（arm/arm64/x86/mips）需要用到的编译工具和系统库。下载命令如下：
```text
wget https://dl.google.com/android/repository/android-ndk-r14b-linux-x86_64.zip
```
笔者当前的目录为`/home/work/android/linux/`，然后让它解压到当前目录，命令如下：
```text
unzip android-ndk-r14b-linux-x86_64.zip
```
如果读者没有安装解压工具，还要先安装解压工具`unzip`，安装命令如下：
```text
apt install unzip
```

然后构建`armeabi-v7a`、 `Android API 21`的独立工具链，命令如下，使用的脚步是刚下载的`Android NDK`的`android-ndk-r14b/build/tools/make-standalone-toolchain.sh`，生成的独立工具链存放在`/home/work/android/linux/arm_standalone_toolchain`：
```text
/home/work/android/linux/android-ndk-r14b/build/tools/make-standalone-toolchain.sh \
        --arch=arm --platform=android-21 --install-dir=/home/work/android/linux/arm_standalone_toolchain
```
切入到Paddle目录下，并创建build目录
```python
# 切入到Paddle源码中
cd Paddle
# 创建一个build目录，在此编译
mkdir build
# 切入到build目录
cd build
```
在`build`目录下配置交叉编译参数，编译的`Android ABI`是`armeabi-v7a`，使用的工具链是上一面生成的工具链`/home/work/android/linux/arm_standalone_toolchain`，设置存放编译好的文件存放在`/home/work/android/linux/install`，具体命令如下，不要少了最后的`..`，这个是说在上一个目录使用`CMake`文件：
```text
cmake -DCMAKE_SYSTEM_NAME=Android \
      -DANDROID_STANDALONE_TOOLCHAIN=/home/work/android/linux/arm_standalone_toolchain \
      -DANDROID_ABI=armeabi-v7a \
      -DANDROID_ARM_NEON=ON \
      -DANDROID_ARM_MODE=ON \
      -DUSE_EIGEN_FOR_BLAS=ON \
      -DCMAKE_INSTALL_PREFIX=/home/work/android/linux/install \
      -DWITH_C_API=ON \
      -DWITH_SWIG_PY=OFF \
      ..
```
### 编译和安装
CMake配置完成后，执行以下命令，PaddlePaddle将自动下载和编译所有第三方依赖库、编译和安装PaddlePaddle预测库。在`make`前应保证PaddlePaddle的源码目录是干净的，也就是没有编译过其他平台的PaddlePaddle库，又或者已经删除了之前编译生成的文件。
```python
# 使用12线程make
make -j12
# 开始安装
make install
```
当编译完成之后，在`/home/work/android/linux/install`目录下创建以下三个目录。这些文件就是我们之后在Android的APP上会使用的的文件，这些文件跟我们之前使用Docker编译的结果是一样的：

 - `include`是C-API的头文件
 - `lib`是Android ABI的PaddlePaddle库
 - `third_party`是所依赖的所有第三方库

同样，上面的流程是生成`armeabi-v7a`，`Android API 21`的PaddlePaddle库。如果要编译`arm64-v8a`，`Android API 21`的PaddlePaddle库要修改两处的参数。
第一处构建独立工具链的时候：
```text
/home/work/android/linux/android-ndk-r14b/build/tools/make-standalone-toolchain.sh \
        --arch=arm64 --platform=android-21 --install-dir=/home/work/android/linux/arm64_standalone_toolchain
```
第二处是配置交叉编译参数的时候：
```text
cmake -DCMAKE_SYSTEM_NAME=Android \
      -DANDROID_STANDALONE_TOOLCHAIN=/home/work/android/linux/arm64_standalone_toolchain \
      -DANDROID_ABI=arm64-v8a \
      -DUSE_EIGEN_FOR_BLAS=OFF \
      -DCMAKE_INSTALL_PREFIX=/home/work/android/linux/install \
      -DWITH_C_API=ON \
      -DWITH_SWIG_PY=OFF \
      ..
```
如果读者不想操作以上的步骤，也可以直接下载官方编译好的PaddlePaddle库，可以在PaddlePaddle开源社区的[wiki](https://github.com/PaddlePaddle/Mobile/wiki)下载

# 训练模型
-------
我们要使用PaddlePadad预先训练我们的神经网络模型才能进行下一步操作。我们这次使用的是mobilenet神经网络，这个网络更它的名字一样，是为了移植到移动设备上的一个神经网络，虽然我们第三章的[CIFAR彩色图像识别](http://blog.csdn.net/qq_33200967/article/details/79095224)使用的是VGG神经模型，但是使用的流程基本上是一样的。
## 定义神经网络
创建一个`mobilenet.py`的Python文件，来定义我的mobilenet神经网络模型。mobilenet是Google针对手机等嵌入式设备提出的一种轻量级的深层神经网络，它的核心思想就是卷积核的巧妙分解，可以有效减少网络参数，从而达到减小训练时网络的模型。因为太大的模型参数是不利于移植到移动设备上的，比如我们使用的VGG在训练CIFAR10的时候，模型会有58M那么大，这样的模型如下移植到Android应用上，那会大大增加apk的大小，这样是不利于应用的推广的。
```python
# edit-mode: -*- python -*-
import paddle.v2 as paddle


def conv_bn_layer(input,
                  filter_size,
                  num_filters,
                  stride,
                  padding,
                  channels=None,
                  num_groups=1,
                  active_type=paddle.activation.Relu(),
                  layer_type=None):
    """
    A wrapper for conv layer with batch normalization layers.
    Note:
    conv layer has no activation.
    """
    tmp = paddle.layer.img_conv(
        input=input,
        filter_size=filter_size,
        num_channels=channels,
        num_filters=num_filters,
        stride=stride,
        padding=padding,
        groups=num_groups,
        act=paddle.activation.Linear(),
        bias_attr=False,
        layer_type=layer_type)
    return paddle.layer.batch_norm(input=tmp, act=active_type)


def depthwise_separable(input, num_filters1, num_filters2, num_groups, stride,
                        scale):
    """
    """
    tmp = conv_bn_layer(
        input=input,
        filter_size=3,
        num_filters=int(num_filters1 * scale),
        stride=stride,
        padding=1,
        num_groups=int(num_groups * scale),
        layer_type='exconv')

    tmp = conv_bn_layer(
        input=tmp,
        filter_size=1,
        num_filters=int(num_filters2 * scale),
        stride=1,
        padding=0)
    return tmp


def mobile_net(img_size, class_num, scale=1.0):

    img = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(img_size))

    # conv1: 112x112
    tmp = conv_bn_layer(
        img,
        filter_size=3,
        channels=3,
        num_filters=int(32 * scale),
        stride=2,
        padding=1)

    # 56x56
    tmp = depthwise_separable(
        tmp,
        num_filters1=32,
        num_filters2=64,
        num_groups=32,
        stride=1,
        scale=scale)
    tmp = depthwise_separable(
        tmp,
        num_filters1=64,
        num_filters2=128,
        num_groups=64,
        stride=2,
        scale=scale)
    # 28x28
    tmp = depthwise_separable(
        tmp,
        num_filters1=128,
        num_filters2=128,
        num_groups=128,
        stride=1,
        scale=scale)
    tmp = depthwise_separable(
        tmp,
        num_filters1=128,
        num_filters2=256,
        num_groups=128,
        stride=2,
        scale=scale)
    # 14x14
    tmp = depthwise_separable(
        tmp,
        num_filters1=256,
        num_filters2=256,
        num_groups=256,
        stride=1,
        scale=scale)
    tmp = depthwise_separable(
        tmp,
        num_filters1=256,
        num_filters2=512,
        num_groups=256,
        stride=2,
        scale=scale)
    # 14x14
    for i in range(5):
        tmp = depthwise_separable(
            tmp,
            num_filters1=512,
            num_filters2=512,
            num_groups=512,
            stride=1,
            scale=scale)
    # 7x7
    tmp = depthwise_separable(
        tmp,
        num_filters1=512,
        num_filters2=1024,
        num_groups=512,
        stride=2,
        scale=scale)
    tmp = depthwise_separable(
        tmp,
        num_filters1=1024,
        num_filters2=1024,
        num_groups=1024,
        stride=1,
        scale=scale)

    tmp = paddle.layer.img_pool(
        input=tmp, pool_size=7, stride=1, pool_type=paddle.pooling.Avg())
    out = paddle.layer.fc(
        input=tmp, size=class_num, act=paddle.activation.Softmax())

    return out


if __name__ == '__main__':
    img_size = 3 * 32 * 32
    data_dim = 10
    out = mobile_net(img_size, data_dim, 1.0)
```
## 编写训练代码
然后我们编写一个`trian.py`的文件来编写接下来的Python代码。
### 初始化PaddlePaddle
我们创建一个`TestCIFAR`的类来做我们的训练，在初始化的时候，我们就让PaddlePaddle初始化，这里使用4个GPU来训练，在PaddlePaddle使用之前，都要初始化PaddlePaddle，但是不能重复初始化。
```python
class TestCIFAR:
    def __init__(self):
        # 初始化paddpaddle,
        paddle.init(use_gpu=True, trainer_count=4)
```
### 获取训练参数
然后是编写获取训练参数的代码，这个提供了两个获取参数的方法，一个是从损失函数中创建一个训练参数，另一个是使用之前训练好的训练参数：
```python
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
            with gzip.open(parameters_path, 'r') as f:
                parameters = paddle.parameters.Parameters.from_tar(f)
            return parameters
        except Exception as e:
            raise NameError("你的参数文件错误,具体问题是:%s" % e)
```
### 获取训练器
通过损失函数、训练参数、优化方法可以创建一个训练器。

 - `cost`，损失函数，通过神经网络的分类器和分类的标签可以获取损失函数。
 - `parameters`，训练参数，这个在上已经讲过了，这里就不重复了。
 - `optimizer`，优化方法，这个优化方法是设置学习率和加正则的。
```python
def get_trainer(self):
    # 数据大小
    datadim = 3 * 32 * 32

    # 获得图片对于的信息标签
    lbl = paddle.layer.data(name="label",
                            type=paddle.data_type.integer_value(10))

    # 获取全连接层,也就是分类器
    out = mobile_net(datadim, 10, 1.0)

    # 获得损失函数
    cost = paddle.layer.classification_cost(input=out, label=lbl)

    # 使用之前保存好的参数文件获得参数
    # parameters = self.get_parameters(parameters_path="../model/mobile_net.tar.gz")
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
### 开始训练
有了训练器之前，再加上训练数据就可以进行训练了，我们还是使用我们比较熟悉的CIFAR10数据集，PaddlePaddle提供了下载接口，只要调用PaddlePaddle的数据接口就可以了。

同时我们也定义了一个训练事件，通过这个事件可以输出训练的日志，也可以保存我们训练的参数，比如我们在每一个Pass之后，都会保存训练参。同时也记录了训练和测试的cost和分类错误，方便输出图像观察训练效果。
```python
    def start_trainer(self):
        # 获得数据
        reader = paddle.batch(reader=paddle.reader.shuffle(reader=paddle.dataset.cifar.train10(),
                                                           buf_size=50000),
                              batch_size=128)

        # 指定每条数据和padd.layer.data的对应关系
        feeding = {"image": 0, "label": 1}

        saveCost = SaveCost()

        lists = []
        # 定义训练事件，输出日志
        def event_handler(event):
            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 1 == 0:
                    print "\nPass %d, Batch %d, Cost %f, %s" % (
                        event.pass_id, event.batch_id, event.cost, event.metrics)
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()

                # 保存训练的cost,用于生成折线图,便于观察
                saveCost.save_trainer_cost(cost=event.cost)
                saveCost.save_trainer_classification_error(error=event.metrics['classification_error_evaluator'])

            # 每一轮训练完成之后
            if isinstance(event, paddle.event.EndPass):
                # 保存训练好的参数
                model_path = '../model'
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                with gzip.open(model_path + '/mobile_net.tar.gz', 'w') as f:
                    trainer.save_parameter_to_tar(f)

                # 测试准确率
                result = trainer.test(reader=paddle.batch(reader=paddle.dataset.cifar.test10(),
                                                          batch_size=128),
                                      feeding=feeding)
                print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)
                lists.append((event.pass_id, result.cost,
                              result.metrics['classification_error_evaluator']))
                # 保存训练的cost,用于生成折线图,便于观察
                saveCost.save_test_cost(cost=result.cost)
                saveCost.save_test_classification_error(error=result.metrics['classification_error_evaluator'])

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
                      num_passes=50,
                      event_handler=event_handler,
                      feeding=feeding)

        # find the best pass
        best = sorted(lists, key=lambda list: float(list[1]))[0]
        print 'Best pass is %s, testing Avgcost is %s' % (best[0], best[1])
        print 'The classification accuracy is %.2f%%' % (100 - float(best[2]) * 100)
```
我们启动训练，一共是训练50个Pass，训练次数还是比较少的，但是这个模型比较深，训练数据非常慢，笔者使用4个GPU训练，大约训练了39个小时才训练完成，可以说是非常久的。
```python
if __name__ == '__main__':
    # 开始训练
    start_train = time.time()
    testCIFAR = TestCIFAR()
    # 开始训练时间
    testCIFAR.start_trainer()
    # 结束时间
    end_train = time.time()
    print '训练时间为：', end_train - start_train, 'ms'
```
训练的时候会输出类似以下的日志：
```text
Pass 49, Batch 385, Cost 0.172634, {'classification_error_evaluator': 0.046875}
Pass 49, Batch 386, Cost 0.238134, {'classification_error_evaluator': 0.109375}
Pass 49, Batch 387, Cost 0.182165, {'classification_error_evaluator': 0.0546875}
Pass 49, Batch 388, Cost 0.259370, {'classification_error_evaluator': 0.1484375}
Pass 49, Batch 389, Cost 0.221146, {'classification_error_evaluator': 0.0859375}
```

## 编写预测代码
我们这里预测不是真正这样应用，我们使用Python在电脑上测试预测的结果和预测时间，跟之后在Android上的预测做一些对比。
```python
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

    # 获得要预测的图片
    test_data = []
    test_data.append((load_image(image_path),))

    # 开始预测时间
    start_infer = time.time()

    # 获得预测结果
    probs = paddle.infer(output_layer=out,
                         parameters=parameters,
                         input=test_data)
    # 结束预测时间
    end_infer = time.time()

    print '预测时间：', end_infer - start_infer, 'ms'

    # 处理预测结果
    lab = np.argsort(-probs)
    # 返回概率最大的值和其对应的概率值
    return lab[0][0], probs[0][(lab[0][0])]
```
然后在程序入口处调用预测函数，别忘了在使用PaddlePaddle前要初始化PaddlePaddle，我们这里使用的是1一个CPU来预测，同时还要从神经网络中获取分类器和加载上一步训练好的模型参数：
```python
if __name__ == '__main__':
    paddle.init(use_gpu=False, trainer_count=2)
    # 开始预测
    out = mobile_net(3 * 32 * 32, 10)
    with gzip.open("../model/mobile_net.tar.gz", 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)
    image_path = "../images/airplane1.png"
    result, probability = to_prediction(image_path=image_path, out=out, parameters=parameters)
    print '预测结果为:%d,可信度为:%f' % (result, probability)
```
预测结果为，在电脑上预测可以说是相当快的，这里只是统计预测时间，不包括初始化PaddlePaddle和加载神经网络的时间：
```text
预测时间： 0.132810115814 ms
预测结果为:0,可信度为:0.868770
```
# 合并模型
------
## 准备文件
合并模型是指把神经网络和训练好的模型参数合并生成一个可是直接使用的网络模型，合并模型需要两个文件：

 - **模型配置文件：** 用于推断任务的模型配置文件，就是我们用了训练模型时使用到的神经网络，必须只包含`inference`网络，即不能包含训练网络中需要的`label`、`loss`以及`evaluator`层。我们的这里的模型配置文件就是之前定义的`mobilenet.py`的mobilenet神经网络的Python文件。

 - **参数文件：** 使用训练时保存的模型参数，因为`paddle.utils.merge_model`合并模型时只读取`.tar.gz`，所以保存网络参数是要注意保存的格式。如果保存的格式为`.tar`，也没有关系，可以把里面的所有文件提取出来再压缩为`.tar.gz`的文件，压缩的时候要注意不需要为这些参数文件创建文件夹，直接压缩就可以，否则程序会找不到参数文件。保存参数文件程序如下：
```python
with open(model_path + '/model.tar.gz', 'w') as f:
    trainer.save_parameter_to_tar(f)
```

## 开始合并
编写一个Python程序文件`merge_model.py`来合并模型，代码如下：
```python
# coding=utf-8
from paddle.utils.merge_model import merge_v2_model

# 导入mobilenet神经网络
from mobilenet import mobile_net

if __name__ == "__main__":
    # 图像的大小
    img_size = 3 * 32 * 32
    # 总分类数
    class_dim = 10
    net = mobile_net(img_size, class_dim)
    param_file = '../model/mobile_net.tar.gz'
    output_file = '../model/mobile_net.paddle'
    merge_v2_model(net, param_file, output_file)
```
成功合并模型后会输出一下日志，同时会生成`mobile_net.paddle`文件。
```text
Generate  ../model/mobile_net.paddle  success!
```

# 移植到Android
------
使用最新的Android Studio创建一个可以支持C++开发的Android项目`TestPaddle2`。
## 加载PaddlePaddle库
我们在`项目根目录/app/`下创建一个`paddle-android`文件夹，把第一步编译好的PaddlePaddle库的三个文件都存放在这里，它们分别是：`include`，`lib`，`third_party`。

把文件存放在`paddle-android`这里之后，项目还不能直接使用，还要Android Studio把它们编译到项目中，我们使用的是`项目根目录/app/CMakeLists.txt`，我们介绍一下它都加载了哪些库：

 - `set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/")`：设置.cmake文件查找的路径
 - `set(PADDLE_ROOT ${CMAKE_SOURCE_DIR}/paddle-android)`：设置`paddle-android`库的路径，在`项目根目录/app/FindPaddle.cmake`里面需要用到,，该文件加载PaddlePaddle库的，因为该文件代码比较多，笔者就不展示了，可以自行查看源码。
 - `find_package(Paddle)`：查找paddle-android库的头文件和库文件是否存在
`set(SRC_FILES src/main/cpp/image_recognizer.cpp)`：项目中所有C++源码文件
 - `add_library(paddle_image_recognizer SHARED ${SRC_FILES})`：生成动态库即.so文件

加载完成PaddlePaddle之后，我们还有加载一个文件，那就是我们第二步合并的模型，这个模型是我们要用来预测图像的，所以接下来我们就看看如何处理我们的合并的模型。

## 加载合并模型
我们把合并的模型`mobile_net.paddle`存放在`项目根目录/app/src/main/assets/model.include`，然后通过调用PaddlePaddle的接口就可以加载完成合并模型，把路径`model/include/mobile_net.paddle`传入即可，还是听方便的。
```cpp
long size;
void* buf = BinaryReader()(merged_model_path, &size);

ECK(paddle_gradient_machine_create_for_inference_with_parameters(
      &gradient_machine_, buf, size));
```
为什么我们可以直接这样传路径，而不用带前面的路径呢，这是因为我们在`app`下的`build.gradle`做了一些设置，在`android`增加了这几行代码：
```text
sourceSets {
        main {
            manifest.srcFile "src/main/AndroidManifest.xml"
            java.srcDirs = ["src/main/java"]
            assets.srcDirs = ["src/main/assets"]
            jni.srcDirs = ["src/main/cpp"]
            jniLibs.srcDirs = ["paddle-android/lib"]
        }
    }
```
这样只要在传路径之前，把上下文传给`BinaryReader`即可：
```cpp
AAssetManager *aasset_manager = AAssetManager_fromJava(env, jasset_manager);
BinaryReader::set_aasset_manager(aasset_manager);
```
这个部分在这里不细讲，到下一部分笔者再把这个流程再讲一下。

## 开发Android程序
加载完成PaddlePaddle库之后，就可以使用PaddlePaddle来做我们的Android开发了，接下来我们就开始开发Android应用吧。

这里对于Android的开发笔者不会细讲，因为这里主要是讲在Android应用PaddlePaddle，所以笔者只会讲一些关键的代码。

在应用启动是，我们就应该让它初始化和加载模型：
### 初始化PaddlePaddle
这个跟我们在Python上的初始化是差不多的，在初始化是指定是否使用GPU，通过`paddle_init`CAPI接口初始化PaddlePaddle：
```c
JNIEXPORT void
Java_com_yeyupiaoling_testpaddle_ImageRecognition_initPaddle(JNIEnv *env, jobject thiz) {
    static bool called = false;
    if (!called) {
        // Initalize Paddle
        char* argv[] = {const_cast<char*>("--use_gpu=False"),
                        const_cast<char*>("--pool_limit_size=0")};
        CHECK(paddle_init(2, (char**)argv));
        called = true;
    }
}
```
这个C++的函数对应的是Java中`ImageRecognition`类的方法
```java
// CPP中初始化PaddlePaddle
public native void initPaddle();
```
这个Java类主要是用来给`MainActivity.java`调用C++函数的，同`ImageRecognition`的`native`方法，其他的Java类就可以调用自己写的C++函数了，但是不要忘了，要在`ImageRecognition`这个列中加载我们编写的C++程序：
```
static {
      System.loadLibrary("image_recognition");
}
```
### 加载合并模型
因为我们使用的是合并模型，所以跟之前在Python上使用的有点不一样，在Python的时候，我们要使用到升级网络输出的分类器`out`和训练是保存的模型参数`parameters`。而在这里，我们使用到的是合并模型，这个合并模型已经包含了分类器和模型参数了，所以只要这一个文件就可以了。
```c
JNIEXPORT void
Java_com_yeyupiaoling_testpaddle_ImageRecognition_loadModel(JNIEnv *env,
                                                            jobject thiz,
                                                            jobject jasset_manager,
                                                            jstring modelPath) {
    //加载上下文
    AAssetManager *aasset_manager = AAssetManager_fromJava(env, jasset_manager);
    BinaryReader::set_aasset_manager(aasset_manager);

    const char *merged_model_path = env->GetStringUTFChars(modelPath, 0);
    // Step 1: Reading merged model.
    LOGI("merged_model_path = %s", merged_model_path);
    long size;
    void *buf = BinaryReader()(merged_model_path, &size);
    // Create a gradient machine for inference.
    CHECK(paddle_gradient_machine_create_for_inference_with_parameters(
            &gradient_machine_, buf, size));
    // 释放空间
    env->ReleaseStringUTFChars(modelPath, merged_model_path);
    LOGI("加载模型成功");
    free(buf);
    buf = nullptr;
}
```
而这个方法就对应`ImageRecognition`类的方法：
```java
// CPP中加载预测合并模型
public native void loadModel(AssetManager assetManager, String modelPath);
```
这一步和上面的初始化PaddlePaddle都是要在`activity`加的时候就应该执行了：
```java
imageRecognition = new ImageRecognition();
imageRecognition.initPaddle();
imageRecognition.loadModel(this.getAssets(), "model/include/mobile_net.paddle");
```

### 预测图像
这个是我们的预测CPP程序，这个调用了PaddlePaddle的CAPI，通过这些接口来让模型做一个向前的计算，通过这个计算来获取到我们的预测结果。
因为PaddlePaddle读取的数据是float数组，而我们传过来的只是字节数组，所以我们要对数据进行转换，加了一个把字节数的`jpixels`的转成float数组的`array`。最后我们获得的结果也是一个float数组的`array`，这个是每个类别对于的概率：
```c
JNIEXPORT jfloatArray
Java_com_yeyupiaoling_testpaddle_ImageRecognition_infer(JNIEnv *env,
                                                        jobject thiz,
                                                        jbyteArray jpixels) {

    //网络的输入和输出被组织为paddle_arguments对象
    //在C-API中。在下面的评论中，“argument”具体指的是一个输入
    //PaddlePaddle C-API中的神经网络。
    paddle_arguments in_args = paddle_arguments_create_none();

    //调用函数来创建一个参数。
    CHECK(paddle_arguments_resize(in_args, 1));

    //每个参数需要一个矩阵或一个ivector（整数向量，稀疏
    //索引输入，通常用于NLP任务）来保存真实的输入数据。
    //在下面的评论中，“matrix”具体指的是需要的对象
    //参数来保存数据。这里我们为上面创建的矩阵创建
    //储存测试样品的存量。
    paddle_matrix mat = paddle_matrix_create(1, 3072, false);

    paddle_real *array;
    //获取指向第一行开始地址的指针
    //创建矩阵。
    CHECK(paddle_matrix_get_row(mat, 0, &array));

    //获取字节数组转换成浮点数组
    unsigned char *pixels =
            (unsigned char *) env->GetByteArrayElements(jpixels, 0);
    // RGB/RGBA -> RGB
    size_t index = 0;
    std::vector<float> means;
    means.clear();
    for (size_t i = 0; i < 3; ++i) {
        means.push_back(0.0f);
    }
    for (size_t c = 0; c < 3; ++c) {
        for (size_t h = 0; h < 32; ++h) {
            for (size_t w = 0; w < 32; ++w) {
                array[index] =
                        static_cast<float>(
                                pixels[(h * 32 + w) * 3 + c]) - means[c];
                index++;
            }
        }
    }
    env->ReleaseByteArrayElements(jpixels, (jbyte *) pixels, 0);

    //将矩阵分配给输入参数。
    CHECK(paddle_arguments_set_value(in_args, 0, mat));

    //创建输出参数。
    paddle_arguments out_args = paddle_arguments_create_none();

    //调用向前计算。
    CHECK(paddle_gradient_machine_forward(gradient_machine_, in_args, out_args, false));

    //创建矩阵来保存神经网络的向前结果。
    paddle_matrix prob = paddle_matrix_create_none();
    //访问输出参数的矩阵，预测结果存储在哪个。
    CHECK(paddle_arguments_get_value(out_args, 0, prob));

    uint64_t height;
    uint64_t width;
    //获取矩阵的大小
    CHECK(paddle_matrix_get_shape(prob, &height, &width));
    //获取预测结果矩阵
    CHECK(paddle_matrix_get_row(prob, 0, &array));

    jfloatArray result = env->NewFloatArray(height * width);
    env->SetFloatArrayRegion(result, 0, height * width, array);

    // 清空内存
    CHECK(paddle_matrix_destroy(prob));
    CHECK(paddle_arguments_destroy(out_args));
    CHECK(paddle_matrix_destroy(mat));
    CHECK(paddle_arguments_destroy(in_args));

    return result;
}
```
这个方法对应`ImageRecognition`类的方法：
```java
// CPP中获取预测结果
private native float[] infer(byte[] pixels);
```
在Java中，我们要获取到图像数据，我们从相册中获取图像：
```java
//打开相册
private void getPhoto() {
    Intent intent = new Intent(Intent.ACTION_PICK);
    intent.setType("image/*");
    startActivityForResult(intent, 1);
}
```
如果读者的手机是Android 6.0以上的，我们还有做一个动态获取权限的操作：
```java
//从相册获取照片
getPhotoBtn.setOnClickListener(new View.OnClickListener() {
    @Override
    public void onClick(View v) {
        if (ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        } else {
            getPhoto();
        }
    }
});
```
然后要在权限回调中也要做相应的操作，比如申请权限成功之后要打开相册，申请权限失败要提示用户打开相册失败：
```java
// 动态申请权限回调
@Override
public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                       @NonNull int[] grantResults) {
    switch (requestCode) {
        case 1:
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                getPhoto();
            } else {
                toastUtil.showToast("你拒绝了授权");
            }
            break;
    }
}
```
最后当用户选择图像只是，在回调中可以获取该图像的URI：
```java
// 相册获取照片回调
@Override
protected void onActivityResult(int requestCode, int resultCode, Intent data) {
    if (resultCode == Activity.RESULT_OK) {
        switch (requestCode) {
            case 1:
                Uri uri = data.getData();
                break;
        }
    }
}
```
然后编写一个工具类来把URI转成图像的路径：
```java
//获取图片的路径
public static String getRealPathFromURI(Context context, Uri uri) {
    String result;
    Cursor cursor = context.getContentResolver().query(uri, null, null, null, null);
    if (cursor == null) {
        result = uri.getPath();
    } else {
        cursor.moveToFirst();
        int idx = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);
        result = cursor.getString(idx);
        cursor.close();
    }
    return result;
}
```
之后通过调用这个方法就可以获取到图像的路径了：
```java
String imagePath = CameraUtil.getRealPathFromURI(MainActivity.this, uri);
```
最后在调用预测方法，获取到预测结果：
```java
String resutl = imageRecognition.infer(imagePath);
```
这里要注意，这个的`infer`方法不是我们的真正调用C++函数的方法，我们C++的预测函数传入的是一个字节数组：
```java
private native float[] infer(byte[] pixels);
```
所以我们要把获得的图像转换成字节数组，再去调用预测的C++接口：
```java
public String infer(String img_path) {
	//把图像读取成一个Bitmap对象
	Bitmap bitmap = BitmapFactory.decodeFile(img_path);
	Bitmap mBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
	mBitmap.setWidth(32);
	mBitmap.setHeight(32);
	int width = mBitmap.getWidth();
	int height = mBitmap.getHeight();
	int channel = 3;
	//把图像生成一个数组
	byte[] pixels = getPixelsBGR(mBitmap);
	// 获取预测结果
	float[] result = infer(pixels, width, height, channel);
	// 把概率最大的结果提取出来
	float max = 0;
	int number = 0;
	for (int i = 0; i < result.length; i++) {
		if (result[i] > max) {
			max = result[i];
			number = i;
		}
	}
	String msg = "类别为：" + clasName[number] + "，可信度为：" + max;
	Log.i("ImageRecognition", msg);
	return msg;
}
```

其中我们调用了一个`getPixelsBGR()`方法，这个CIFAR图片在训练时的通道顺序为B(蓝)、G(绿)、R(红)，而我们使用Bitmap读取图像的通道是RGB顺序的，所以我们还有转换一下它们的通道顺序，转换方法如下：
```java
public byte[] getPixelsBGR(Bitmap bitmap) {
	// 计算我们的图像包含多少字节
	int bytes = bitmap.getByteCount();

	ByteBuffer buffer = ByteBuffer.allocate(bytes);
	// 将字节数据移动到缓冲区
	bitmap.copyPixelsToBuffer(buffer);

	// 获取包含数据的基础数组
	byte[] temp = buffer.array();

	byte[] pixels = new byte[(temp.length/4) * 3];
	// 进行像素复制
	for (int i = 0; i < temp.length/4; i++) {
		pixels[i * 3] = temp[i * 4 + 2]; //B
		pixels[i * 3 + 1] = temp[i * 4 + 1]; //G
		pixels[i * 3 + 2] = temp[i * 4 ]; //R
	}
	return pixels;
}
```

这个我们的预测结果的截图：
![这里写图片描述](//img-blog.csdn.net/20180318104637275?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjAwOTY3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

# 项目代码
--------
GitHub地址:https://github.com/yeyupiaoling/LearnPaddle

# 参考资料
---------
 1. http://paddlepaddle.org/
 2. https://github.com/PaddlePaddle/Mobile/tree/develop/Demo/Android/AICamera
 3. http://blog.csdn.net/wfei101/article/details/78310226
 4. https://arxiv.org/abs/1704.04861
