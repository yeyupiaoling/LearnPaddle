# 目录
@[toc]
*本篇文章基于 PaddlePaddle 0.11.0、Python 2.7
# 数据集介绍
-------
如果我们要训练自己的数据集的话,就需要先建立图像列表文件,下面的代码是`Myreader.py`读取图像数据集的一部分,从这些代码中可以看出,图像列表中,图像的路径和标签是以`\t`来分割的,所以我们在生成这个列表的时候,使用`\t`就可以了.

```python
def train_reader(self,train_list, buffered_size=1024):
    def reader():
        with open(train_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab)

    return paddle.reader.xmap_readers(self.train_mapper, reader,
                                      cpu_count(), buffered_size)
```
生成的图像列表的结构是这样的:
```
../images/vegetables/lotus_root/1515827057517.jpg	2
../images/vegetables/lotus_root/1515827057582.jpg	2
../images/vegetables/lotus_root/1515827057616.jpg	2
../images/vegetables/lettuce/1515827015922.jpg	1
../images/vegetables/lettuce/1515827015983.jpg	1
../images/vegetables/lettuce/1515827016045.jpg	1
../images/vegetables/cuke/1515827008337.jpg	0
../images/vegetables/cuke/1515827008370.jpg	0
../images/vegetables/cuke/1515827008402.jpg	0
```

# 生成图像列表
--------
所以我们要编写一个`CreateDataList.py`程序可以为我们生成这样的图像列表
在这个程序中,我们只要把一个大类的文件夹路径传进去就可以了,该程序会把里面的每个小类别都迭代,生成固定格式的列表.比如我们把蔬菜类别的根目录传进去`../images/vegetables`
```python
# coding=utf-8
import os
import json

class CreateDataList:
    def __init__(self):
        pass

    def createDataList(self, data_root_path):
        # # 把生产的数据列表都放在自己的总类别文件夹中
        data_list_path = ''
        # 所有类别的信息
        class_detail = []
        # 获取所有类别
        class_dirs = os.listdir(data_root_path)
        # 类别标签
        class_label = 0
        # 获取总类别的名称
        father_paths = data_root_path.split('/')
        while True:
            if father_paths[father_paths.__len__() - 1] == '':
                del father_paths[father_paths.__len__() - 1]
            else:
                break
        father_path = father_paths[father_paths.__len__() - 1]

        all_class_images = 0
        # 读取每个类别
        for class_dir in class_dirs:
            # 每个类别的信息
            class_detail_list = {}
            test_sum = 0
            trainer_sum = 0
            # 把生产的数据列表都放在自己的总类别文件夹中
            data_list_path = "../data/%s/" % father_path
            # 统计每个类别有多少张图片
            class_sum = 0
            # 获取类别路径
            path = data_root_path + "/" + class_dir
            # 获取所有图片
            img_paths = os.listdir(path)
            for img_path in img_paths:
                # 每张图片的路径
                name_path = path + '/' + img_path
                # 如果不存在这个文件夹,就创建
                isexist = os.path.exists(data_list_path)
                if not isexist:
                    os.makedirs(data_list_path)
                # 每10张图片取一个做测试数据
                if class_sum % 10 == 0:
                    test_sum += 1
                    with open(data_list_path + "test.list", 'a') as f:
                        f.write(name_path + "\t%d" % class_label + "\n")
                else:
                    trainer_sum += 1
                    with open(data_list_path + "trainer.list", 'a') as f:
                        f.write(name_path + "\t%d" % class_label + "\n")
                class_sum += 1
                all_class_images += 1
            class_label += 1
            # 说明的json文件的class_detail数据
            class_detail_list['class_name'] = class_dir
            class_detail_list['class_label'] = class_label
            class_detail_list['class_test_images'] = test_sum
            class_detail_list['class_trainer_images'] = trainer_sum
            class_detail.append(class_detail_list)
        # 获取类别数量
        all_class_sum = class_dirs.__len__()
        # 说明的json文件信息
        readjson = {}
        readjson['all_class_name'] = father_path
        readjson['all_class_sum'] = all_class_sum
        readjson['all_class_images'] = all_class_images
        readjson['class_detail'] = class_detail
        jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
        with open(data_list_path + "readme.json",'w') as f:
            f.write(jsons)


if __name__ == '__main__':
    createDataList = CreateDataList()
    createDataList.createDataList('../images/vegetables')
```

运行这个程序之后,会生成在data文件夹中生成一个单独的大类文件夹,比如我们这次是使用到蔬菜类,所以我生成一个`vegetables`文件夹,在这个文件夹下有3个文件:
|文件名|作用|
|:---:|:---:|
|trainer.list|用于训练的图像列表|
|test.list|用于测试的图像列表|
|readme.json|该数据集的json格式的说明,方便以后使用|

`readme.json`文件的格式如下,可以很清楚看到整个数据的图像数量,总类别名称和类别数量,还有每个类对应的标签,类别的名字,该类别的测试数据和训练数据的数量:
```json
{
    "all_class_images": 3300,
    "all_class_name": "vegetables",
    "all_class_sum": 3,
    "class_detail": [
        {
            "class_label": 1,
            "class_name": "cuke",
            "class_test_images": 110,
            "class_trainer_images": 990
        },
        {
            "class_label": 2,
            "class_name": "lettuce",
            "class_test_images": 110,
            "class_trainer_images": 990
        },
        {
            "class_label": 3,
            "class_name": "lotus_root",
            "class_test_images": 110,
            "class_trainer_images": 990
        }
    ]
}
```

# 读取数据
---------
通过`MyReader.py`这个程序可以将上一部分的图像列表读取,生成训练和测试使用的reader,在生成reader前,要传入一个图像的大小,PaddlePaddle会帮我们按照这个大小随机裁剪一个方形的图像,这是种随机裁剪也是数据增强的一种方式.
```python
from multiprocessing import cpu_count
import paddle.v2 as paddle

class MyReader:
    def __init__(self,imageSize):
        self.imageSize = imageSize

    def train_mapper(self,sample):
        '''
        map image path to type needed by model input layer for the training set
        '''
        img, label = sample
        img = paddle.image.load_image(img)
        img = paddle.image.simple_transform(img, 70, self.imageSize, True)
        return img.flatten().astype('float32'), label

    def test_mapper(self,sample):
        '''
        map image path to type needed by model input layer for the test set
        '''
        img, label = sample
        img = paddle.image.load_image(img)
        img = paddle.image.simple_transform(img, 70, self.imageSize, False)
        return img.flatten().astype('float32'), label

    def train_reader(self,train_list, buffered_size=1024):
        def reader():
            with open(train_list, 'r') as f:
                lines = [line.strip() for line in f]
                for line in lines:
                    img_path, lab = line.strip().split('\t')
                    yield img_path, int(lab)

        return paddle.reader.xmap_readers(self.train_mapper, reader,
                                          cpu_count(), buffered_size)

    def test_reader(self,test_list, buffered_size=1024):
        def reader():
            with open(test_list, 'r') as f:
                lines = [line.strip() for line in f]
                for line in lines:
                    img_path, lab = line.strip().split('\t')
                    yield img_path, int(lab)

        return paddle.reader.xmap_readers(self.test_mapper, reader,
                                          cpu_count(), buffered_size)
```

# 定义神经网络
编写一个`vgg.py`来定义VGG神经网络，这里使用的是VGG神经网络,跟上一篇文章用到的VGG又有一点不同,这里可以看到`conv_with_batchnorm=False`，我是把`BN`关闭了，这是因为启用BN层的同时，也会使用`Dropout`层，因为数据集比较小，再使用`Dropout`就更小了，导致模型无法收敛。如果读者一定要启动`BN`层的话，可以单独关闭`Dropout`，把`drop_rate`全部设置为0。如果数据集大的话，就可以不用这样处理。
```python
# coding:utf-8
import paddle.v2 as paddle

def vgg_bn_drop(datadim, type_size):
    # 获取输入数据模式
    image = paddle.layer.data(name="image",
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
            conv_with_batchnorm=False,
            conv_batchnorm_drop_rate=dropouts,
            pool_type=paddle.pooling.Max())

    conv1 = conv_block(image, 64, 2, [0.3, 0], 3)
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
    # 通过Softmax获得分类器
    out = paddle.layer.fc(input=fc2,
                          size=type_size,
                          act=paddle.activation.Softmax())
    return out
```

# 使用PaddlePaddle开始训练
---------
编写`train.py`文件训练模型。
## 导入依赖包
首先要先导入依赖包,其中有PaddlePaddle的V2包和上面定义的`Myreader.py`读取数据的程序
```python
# coding:utf-8
import os
import sys
import paddle.v2 as paddle
from MyReader import MyReader
from vgg import vgg_bn_drop
```
## 初始化Paddle
然后我们创建一个类,再在类中创建一个初始化函数,在初始化函数中来初始化我们的PaddlePaddle
```python
class PaddleUtil:
    def __init__(self):
        # 初始化paddpaddle,只是用CPU,把GPU关闭
        paddle.init(use_gpu=False, trainer_count=2)
```

## 获取参数
该函数可以通过输入是否是参数文件路径,或者是损失函数,如果是参数文件路径,就使用之前训练好的参数生产参数.如果不传入参数文件路径,那就使用传入的损失函数生成参数
```python
def get_parameters(self, parameters_path=None, cost=None):
    if not parameters_path:
        # 使用cost创建parameters
        if not cost:
            raise NameError('请输入cost参数')
        else:
            # 根据损失函数创建参数
            parameters = paddle.parameters.create(cost)
            print "cost"
            return parameters
    else:
        # 使用之前训练好的参数
        try:
            # 使用训练好的参数
            with open(parameters_path, 'r') as f:
                parameters = paddle.parameters.Parameters.from_tar(f)
            print "使用parameters"
            return parameters
        except Exception as e:
            raise NameError("你的参数文件错误,具体问题是:%s" % e)
```

## 创建训练器
创建训练器要3个参数,分别是损失函数,参数,优化方法.通过图像的标签信息和分类器生成损失函数.参数可以选择是使用之前训练好的参数,然后在此基础上再进行训练,又或者是使用损失函数生成初始化参数.然后再生成优化方法.就可以创建一个训练器了.
```python
# datadim 数据大小
def get_trainer(self, datadim, type_size, parameters_path):
    # 获得图片对于的信息标签
    label = paddle.layer.data(name="label",
                              type=paddle.data_type.integer_value(type_size))

    # 获取全连接层,也就是分类器
    out = vgg_bn_drop(datadim=datadim, type_size=type_size)

    # 获得损失函数
    cost = paddle.layer.classification_cost(input=out, label=label)

    # 获得参数
    if not parameters_path:
        parameters = self.get_parameters(cost=cost)
    else:
        parameters = self.get_parameters(parameters_path=parameters_path)

    '''
    定义优化方法
    learning_rate 迭代的速度
    momentum 跟前面动量优化的比例
    regularzation 正则化,防止过拟合
    '''
    optimizer = paddle.optimizer.Momentum(
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128),
        learning_rate=0.001 / 128,
        learning_rate_decay_a=0.1,
        learning_rate_decay_b=128000 * 35,
        learning_rate_schedule="discexp", )

    '''
    创建训练器
    cost 分类器
    parameters 训练参数,可以通过创建,也可以使用之前训练好的参数
    update_equation 优化方法
    '''
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer)
    return trainer
```

## 开始训练
要启动训练要4个参数,分别是训练数据,训练的轮数,训练过程中的事件处理,输入数据和标签的对应关系. 
训练数据:这次的训练数据是我们自定义的数据集. 
训练轮数:表示我们要训练多少轮,次数越多准确率越高,最终会稳定在一个固定的准确率上.不得不说的是这个会比MNIST数据集的速度慢很多 
事件处理:训练过程中的一些事件处理,比如会在每个batch打印一次日志,在每个pass之后保存一下参数和测试一下测试数据集的预测准确率. 
输入数据和标签的对应关系:说明输入数据是第0维度,标签是第1维度
```python
# ***********************开始训练***************************************
def start_trainer(self, trainer, num_passes, save_parameters_name, trainer_reader, test_reader):
    # 获得数据
    reader = paddle.batch(reader=paddle.reader.shuffle(reader=trainer_reader,
                                                       buf_size=50000),
                          batch_size=128)
    # 保证保存模型的目录是存在的
    father_path = save_parameters_name[:save_parameters_name.rfind("/")]
    if not os.path.exists(father_path):
        os.makedirs(father_path)

    # 指定每条数据和padd.layer.data的对应关系
    feeding = {"image": 0, "label": 1}

    # 定义训练事件
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "\nPass %d, Batch %d, Cost %f, Error %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics['classification_error_evaluator'])
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

        # 每一轮训练完成之后
        if isinstance(event, paddle.event.EndPass):
            # 保存训练好的参数
            with open(save_parameters_name, 'w') as f:
                trainer.save_parameter_to_tar(f)

            # 测试准确率
            result = trainer.test(reader=paddle.batch(reader=test_reader,
                                                      batch_size=128),
                                  feeding=feeding)
            print "\nTest with Pass %d, Classification_Error %s" % (
            event.pass_id, result.metrics['classification_error_evaluator'])

    '''
    开始训练
    reader 训练数据
    num_passes 训练的轮数
    event_handler 训练的事件,比如在训练的时候要做一些什么事情
    feeding 说明每条数据和padd.layer.data的对应关系
    '''
    trainer.train(reader=reader,
                  num_passes=num_passes,
                  event_handler=event_handler,
                  feeding=feeding)
```
然后在`main`中调用相应的函数,开始训练,可以看到通过`myReader.train_reader`来生成一个reader
```python
if __name__ == '__main__':
    # 类别总数
    type_size = 3
    # 图片大小
    imageSize = 64
    # 总的分类名称
    all_class_name = 'fruits'
    # 保存的model路径
    parameters_path = "../model/model.tar"
    # 数据的大小
    datadim = 3 * imageSize * imageSize
    paddleUtil = PaddleUtil()
    myReader = MyReader(imageSize=imageSize)
    # parameters_path设置为None就使用普通生成参数,
    trainer = paddleUtil.get_trainer(datadim=datadim, type_size=type_size, parameters_path=None)
    trainer_reader = myReader.train_reader(train_list="../data/%s/trainer.list" % all_class_name)
    test_reader = myReader.test_reader(test_list="../data/%s/test.list" % all_class_name)

    paddleUtil.start_trainer(trainer=trainer, num_passes=100, save_parameters_name=parameters_path,
                             trainer_reader=trainer_reader, test_reader=test_reader)
```
输出日志如下:'
```
Pass 0, Batch 0, Cost 1.162887, Error 0.6171875
.....................
Test with Pass 0, Classification_Error 0.353333324194
```

**提示：**如果报以下错误：
```text
  File "/usr/local/lib/python2.7/dist-packages/paddle/v2/image.py", line 159, in load_image
    im = cv2.imread(file, flag)
AttributeError: 'NoneType' object has no attribute 'imread'
```

解决办法如下，首先升级以下CV2：
```
sudo pip install opencv-python -U
```

然后安装CV2的库：
```
sudo apt install libopencv-dev
```


# 使用PaddlePaddle预测
---------
编写一个`infer.py`来预测我们的数据。
先定义一个获取模型参数的函数：
```python
def get_parameters(parameters_path):
    with open(parameters_path, 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)
    return parameters
```
定义预测函数，该函数需要输入3个参数, 
第一个是需要预测的图像,图像传入之后,会经过load_image函数处理,大小会变成32*32大小,训练是输入数据的大小一样. 
第二个就是训练好的参数 
第三个是通过神经模型生成的分类器
```python
def to_prediction(image_paths, parameters, out, imageSize):

    # 获得要预测的图片
    test_data = []
    for image_path in image_paths:
        test_data.append((paddle.image.load_and_transform(image_path, 70, imageSize, False)
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
```
然后在`main`中调用相应的函数，开始预测,这个可以同时传入多个数据，可以同时预测，最后别忘了在使用PaddlePaddle前初始化PaddlePaddle。
```python
if __name__ == '__main__':
    paddle.init(use_gpu=False, trainer_count=2)
    # 类别总数
    type_size = 3
    # 图片大小
    imageSize = 64
    # 保存的model路径
    parameters_path = "../model/model.tar"
    # 数据的大小
    datadim = 3 * imageSize * imageSize

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
```
输出的结果是:
```
预测结果为:0,可信度为:0.699004
预测结果为:0,可信度为:0.546674
预测结果为:2,可信度为:0.756389
```

# 所有代码
---------
`train.py`，训练代码：
```python
# coding:utf-8
import os
import sys
import paddle.v2 as paddle
from MyReader import MyReader
from vgg import vgg_bn_drop


class PaddleUtil:
    # ***********************初始化操作***************************************
    def __init__(self):
        # 初始化paddpaddle,只是用CPU,把GPU关闭
        paddle.init(use_gpu=False, trainer_count=2)

    # **********************获取参数***************************************
    def get_parameters(self, parameters_path=None, cost=None):
        if not parameters_path:
            # 使用cost创建parameters
            if not cost:
                raise NameError('请输入cost参数')
            else:
                # 根据损失函数创建参数
                parameters = paddle.parameters.create(cost)
                print "cost"
                return parameters
        else:
            # 使用之前训练好的参数
            try:
                # 使用训练好的参数
                with open(parameters_path, 'r') as f:
                    parameters = paddle.parameters.Parameters.from_tar(f)
                print "使用parameters"
                return parameters
            except Exception as e:
                raise NameError("你的参数文件错误,具体问题是:%s" % e)

    # ***********************获取训练器***************************************
    # datadim 数据大小
    def get_trainer(self, datadim, type_size, parameters_path):
        # 获得图片对于的信息标签
        label = paddle.layer.data(name="label",
                                  type=paddle.data_type.integer_value(type_size))

        # 获取全连接层,也就是分类器
        out = vgg_bn_drop(datadim=datadim, type_size=type_size)

        # 获得损失函数
        cost = paddle.layer.classification_cost(input=out, label=label)

        # 获得参数
        if not parameters_path:
            parameters = self.get_parameters(cost=cost)
        else:
            parameters = self.get_parameters(parameters_path=parameters_path)

        '''
        定义优化方法
        learning_rate 迭代的速度
        momentum 跟前面动量优化的比例
        regularzation 正则化,防止过拟合
        '''
        optimizer = paddle.optimizer.Momentum(
            momentum=0.9,
            regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128),
            learning_rate=0.001 / 128,
            learning_rate_decay_a=0.1,
            learning_rate_decay_b=128000 * 35,
            learning_rate_schedule="discexp", )

        '''
        创建训练器
        cost 分类器
        parameters 训练参数,可以通过创建,也可以使用之前训练好的参数
        update_equation 优化方法
        '''
        trainer = paddle.trainer.SGD(cost=cost,
                                     parameters=parameters,
                                     update_equation=optimizer)
        return trainer

    # ***********************开始训练***************************************
    def start_trainer(self, trainer, num_passes, save_parameters_name, trainer_reader, test_reader):
        # 获得数据
        reader = paddle.batch(reader=paddle.reader.shuffle(reader=trainer_reader,
                                                           buf_size=50000),
                              batch_size=128)
        # 保证保存模型的目录是存在的
        father_path = save_parameters_name[:save_parameters_name.rfind("/")]
        if not os.path.exists(father_path):
            os.makedirs(father_path)

        # 指定每条数据和padd.layer.data的对应关系
        feeding = {"image": 0, "label": 1}

        # 定义训练事件
        def event_handler(event):
            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 100 == 0:
                    print "\nPass %d, Batch %d, Cost %f, Error %s" % (
                        event.pass_id, event.batch_id, event.cost, event.metrics['classification_error_evaluator'])
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()

            # 每一轮训练完成之后
            if isinstance(event, paddle.event.EndPass):
                # 保存训练好的参数
                with open(save_parameters_name, 'w') as f:
                    trainer.save_parameter_to_tar(f)

                # 测试准确率
                result = trainer.test(reader=paddle.batch(reader=test_reader,
                                                          batch_size=128),
                                      feeding=feeding)
                print "\nTest with Pass %d, Classification_Error %s" % (
                event.pass_id, result.metrics['classification_error_evaluator'])

        '''
        开始训练
        reader 训练数据
        num_passes 训练的轮数
        event_handler 训练的事件,比如在训练的时候要做一些什么事情
        feeding 说明每条数据和padd.layer.data的对应关系
        '''
        trainer.train(reader=reader,
                      num_passes=num_passes,
                      event_handler=event_handler,
                      feeding=feeding)


if __name__ == '__main__':
    # 类别总数
    type_size = 3
    # 图片大小
    imageSize = 64
    # 总的分类名称
    all_class_name = 'vegetables'
    # 保存的model路径
    parameters_path = "../model/model.tar"
    # 数据的大小
    datadim = 3 * imageSize * imageSize
    paddleUtil = PaddleUtil()

    # *******************************开始训练**************************************
    myReader = MyReader(imageSize=imageSize)
    # # parameters_path设置为None就使用普通生成参数,
    trainer = paddleUtil.get_trainer(datadim=datadim, type_size=type_size, parameters_path=None)
    trainer_reader = myReader.train_reader(train_list="../data/%s/trainer.list" % all_class_name)
    test_reader = myReader.test_reader(test_list="../data/%s/test.list" % all_class_name)

    paddleUtil.start_trainer(trainer=trainer, num_passes=100, save_parameters_name=parameters_path,
                             trainer_reader=trainer_reader, test_reader=test_reader)
```
`infer.py`，预测代码：
```python
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
        test_data.append((paddle.image.load_and_transform(image_path, 70, imageSize, False)
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
    imageSize = 64
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
```

`DownloadImages.py`,下载图片的代码：
这个程序可以从百度图片中下载图片,可以多个类别一起下载,还可以指定下载数量
```python
# -*- coding:utf-8 -*-
import re
import uuid
import requests
import os


class DownloadImages:
    def __init__(self,download_max,key_word):
        self.download_sum = 0
        self.download_max = download_max
        self.key_word = key_word
        self.save_path = '../images/download/' + key_word

    def start_download(self):
        self.download_sum = 0
        gsm = 80
        str_gsm = str(gsm)
        pn = 0
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        while self.download_sum < self.download_max:
            str_pn = str(self.download_sum)
            url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&' \
                  'word=' + self.key_word + '&pn=' + str_pn + '&gsm=' + str_gsm + '&ct=&ic=0&lm=-1&width=0&height=0'
            print url
            result = requests.get(url)
            self.downloadImages(result.text)
        print '下载完成'

    def downloadImages(self,html):
        img_urls = re.findall('"objURL":"(.*?)",', html, re.S)
        print '找到关键词:' + self.key_word + '的图片，现在开始下载图片...'
        for img_url in img_urls:
            print '正在下载第' + str(self.download_sum + 1) + '张图片，图片地址:' + str(img_url)
            try:
                pic = requests.get(img_url, timeout=50)
                pic_name = self.save_path + '/' + str(uuid.uuid1()) + '.' + str(img_url).split('.')[-1]
                with open(pic_name, 'wb') as f:
                    f.write(pic.content)
                self.download_sum += 1
                if self.download_sum >= self.download_max:
                    break
            except  Exception, e:
                print '【错误】当前图片无法下载，%s' % e
                continue


if __name__ == '__main__':
    key_word_max = input('请输入你要下载几个类别:')
    key_words = []
    for sum in range(key_word_max):
        key_words.append(raw_input('请输入第%s个关键字:' % str(sum+1)))
    max_sum = input('请输入每个类别下载的数量:')
    for key_word in key_words:
        downloadImages = DownloadImages(max_sum, key_word)
        downloadImages.start_download()
```

# 项目代码
-------
GitHub地址:[https://github.com/yeyupiaoling/LearnPaddle](https://github.com/yeyupiaoling/LearnPaddle)


# 参考资料
---------
1. http://paddlepaddle.org/
