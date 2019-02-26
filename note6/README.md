# 目录
@[toc]
*本篇文章基于 PaddlePaddle 0.10.0、Python 2.7
# 前言
--------
在上一篇文章中介绍了验证码的识别，但是使用的传统的验证码分割，然后通过图像分类的方法来实现验证码的识别的，这中方法比较繁琐，工作量比较多。在本篇文章会介绍验证码端到端的识别，直接一步到位，不用图像分割那么麻烦了。好吧，现在开始吧！

# 数据集介绍
--------
在本篇文章中同样是使用方正系统的验证码，该数据集在上一篇文章[《我的PaddlePaddle学习之路》笔记五——验证码的识别](http://blog.csdn.net/qq_33200967/article/details/79095295#%E6%95%B0%E6%8D%AE%E9%9B%86%E4%BB%8B%E7%BB%8D)已有介绍，在这里我就不介绍了，需要了解的可以点击链接去到上一篇文章查看。

# 获取验证码
-------
下载验证码和修改验证码同样在上一篇文章有介绍，如果读者需要同样可以回到上一篇文章查看。
验证码我们有了，有看过上一篇文章的读者会第一反应说还缺图像列表。没错，训练和测试都需要一个图像列表


# 把图像转成灰度图
--------
在生成列表之前，我们还有对图像做一些处理，就是把图像灰度化。
**注意：**在此之前应该把图像文件命名，文件名为验证码对应的字符，并把所有的验证码放在`data_temp`
然后执行以下的程序批量处理
```python
# coding=utf-8
import os
from PIL import Image

def Image2GRAY(path):
    # 获取临时文件夹中的所有图像路径
    imgs = os.listdir(path)
    i = 0
    for img in imgs:
        # 每10个数据取一个作为测试数据，剩下的作为训练数据
        if i % 10 == 0:
            # 使图像灰度化并保存
            im = Image.open(path + '/' + img).convert('L')
            im.save('data/test_data/' + img)
        else:
            # 使图像灰度化并保存
            im = Image.open(path + '/' + img).convert('L')
            im.save('data/train_data/' + img)
        i = i + 1

if __name__ == '__main__':
    # 临时数据存放路径
    path = 'data/data_temp'
    Image2GRAY(path)
```

# 生成图像列表
--------
经过上面一步，在`data/train_data`我们有了训练数据集，`data/test_data`测试数据集。然后就在这两个文件夹下生成对应的图像列表。
首先我们要了解图像列表的格式要求，我们来看看它的格式是怎样的
```text
10iw.png	10iw
218j.png	218j
28hi.png	28hi
3n1g.png	3n1g
47q7.png	47q7
4ju5.png	4ju5
4uqh.png	4uqh
```
这个图像类别是以Tab键区分路径和label的，了解图像列表的格式要求之后，那么我们就编写一个程序来生成这样格式的一个图像列表。代码如下：
```python
# coding=utf-8
import os

class CreateDataList:
    def __init__(self):
        pass

    def createDataList(self, data_path, isTrain):
        # 判断生成的列表是训练图像列表还是测试图像列表
        if isTrain:
            list_name = 'trainer.list'
        else:
            list_name = 'test.list'
        list_path = os.path.join(data_path, list_name)
        # 判断该列表是否存在，如果存在就删除，避免在生成图像列表时把该路径也写进去了
        if os.path.exists(list_path):
            os.remove(list_path)
        # 读取所有的图像路径，此时图像列表不存在，就不用担心写入非图像文件路径了
        imgs = os.listdir(data_path)
        for img in imgs:
            name = img.split('.')[0]
            with open(list_path, 'a') as f:
                # 写入图像路径和label，用Tab隔开
                f.write(img + '\t' + name + '\n')

if __name__ == '__main__':
    createDataList = CreateDataList()
    # 生成训练图像列表
    createDataList.createDataList('data/train_data/', True)
    # 生成测试图像列表
    createDataList.createDataList('data/test_data/', False)
```
经过上面的程序，会在`data/train_data`生成图像列表`trainer.list`，会在`data/test_data`生成图像列表`test.list`。到这里，我们的数据集已经准备好了，准备开始使用数据集训练了。

# 数据的读取
-------
## 读取数据成list
数据列表是有了，但是我们使用它就要用到文件读取，生成一个我们方便使用的的数据格式。在本例子项目中，我把图像的路径和label生成是一个list。读取方式如下：
```python
def get_file_list(image_file_list):
    '''
    生成用于训练和测试数据的文件列表。
    :param image_file_list: 图像文件和列表文件的路径
    :type image_file_list: str
    '''
    dirname = os.path.dirname(image_file_list)
    path_list = []
    with open(image_file_list) as f:
        for line in f:
            # 使用Tab键分离路径和label
            line_split = line.strip().split('\t')
            filename = line_split[0].strip()
            path = os.path.join(dirname, filename)
            label = line_split[1].strip()
            if label:
                path_list.append((path, label))

    return path_list
```
有了这个程序，我们就可以轻松拿到训练数据和测试数据的list了，如下：
```python
# 获取训练列表
train_file_list = get_file_list(train_file_list_path)
# 获取测试列表
test_file_list = get_file_list(test_file_list_path)
```

## 生成和读取标签字典
在这个项目中，要使用到我们之前没有使用过的文件：标签字典。这个标签字典是训练数据集中出现的字符，如：
```
r	81
4	77
h	75
i	74
2	72
```
通过每个字符的key就可以找到对应的字符了。
我们要编写一个从训练数据集的list中获取所有的字符，并生成一个标签字典
```python
def build_label_dict(file_list, save_path):
    """
    从训练数据建立标签字典
    :param file_list: 包含标签的训练数据列表
    :type file_list: list
    :params save_path: 保存标签字典的路径
    :type save_path: str
    """
    values = defaultdict(int)
    for path, label in file_list:
        for c in label:
            if c:
                values[c] += 1

    values['<unk>'] = 0
    with open(save_path, "w") as f:
        for v, count in sorted(
                values.iteritems(), key=lambda x: x[1], reverse=True):
            f.write("%s\t%d\n" % (v, count))
```
然后只要传入在上一步读取到的`train_file_list`和保存标签字典的路径就可以生成标签字典了。
```python
build_label_dict(train_file_list, label_dict_path)
```
保存字典之后，我们还要使用到这个字典。所以我们还要编写一个程序来读取标签字典，代码如下：
```python
def load_dict(dict_path):
    """
    从字典路径加载标签字典
    :param dict_path: 标签字典的路径
    :type dict_path: str
    """
    return dict((line.strip().split("\t")[0], idx)
                for idx, line in enumerate(open(dict_path, "r").readlines()))
```
然后通过传入标签字典的路径就可以读取标签字典内容了，如下：
```python
# 获取标签字典
char_dict = load_dict(label_dict_path)
```
## 读取训练和测试的数据
如果学习前面几个例子的，应该会知道trainer传入的数据是`reader`的，在上面获取的训练数据和测试数据都是list类型的，我们要把它转成`reader`类型的。同下面的程序，把训练和测试的数据根据其路径来加载成一维向量
```python
# coding=utf-8
import cv2
import paddle.v2 as paddle

class Reader(object):
    def __init__(self, char_dict, image_shape):
        '''
        :param char_dict: 标签的字典类
        :type char_dict: class
        :param image_shape: 图像的固定形状
        :type image_shape: tuple
        '''
        self.image_shape = image_shape
        self.char_dict = char_dict

    def train_reader(self, file_list):
        '''
        训练读取数据
        :param file_list: 用预训练的图像列表，包含标签和图像路径
        :type file_list: list
        '''
        def reader():
            UNK_ID = self.char_dict['<unk>']
            for image_path, label in file_list:
                label = [self.char_dict.get(c, UNK_ID) for c in label]
                yield self.load_image(image_path), label
        return reader

    def load_image(self, path):
        '''
        加载图像并将其转换为1维矢量
        :param path: 图像数据的路径
        :type path: str
        '''
        image = paddle.image.load_image(path,is_color=False)
        # 将所有图像调整为固定形状
        if self.image_shape:
            image = cv2.resize(
                image, self.image_shape, interpolation=cv2.INTER_CUBIC)
        image = image.flatten() / 255.
        return image
```
我们通过传入标签字典和图像的大小(宽度,高度)获取reader
```python
my_reader = Reader(char_dict=char_dict, image_shape=IMAGE_SHAPE)
```
然后通过执行下面的方法，同时传入训练的list：`train_file_list`和测试的list：`test_file_list`就可以生成reader了。

```python
# 获取测试数据的reader
test_reader = paddle.batch(
    my_reader.train_reader(test_file_list),
    batch_size=BATCH_SIZE)

# 获取训练数据的reader
train_reader = paddle.batch(
    paddle.reader.shuffle(
        my_reader.train_reader(train_file_list),
        buf_size=1000),
    batch_size=BATCH_SIZE)
```

# 定义网络模型
-------
这次使用的网络模型不是单纯的CNN模型了，还有结合了RNN来映射字符的分布和使用CTC来计算CTC任务的成本，具体是如何定义的呢，请往下细看。
跟之前一样，我们同样要定义数据的和label，更之前不一样的是这次我们定义数据的时候指定了宽度和高度，因为我们这个数据集只长方形的。
在定义label的时候，之前我们要传入类别的总数，我们这次还是同样的道理。还记得上一步获得的标签字典吧，标签字典就是我们训练集的所有出现过字符，只要获取字符的大小就可以了。
```python
# 获取字典大小
dict_size = len(char_dict)
```
以下就是类初始化的数据和定义数据和label的操作：
```python
class Model(object):
    def __init__(self, num_classes, shape, is_infer=False):
        '''
        :param num_classes: 字符字典的大小
        :type num_classes: int
        :param shape: 输入图像的大小
        :type shape: tuple of 2 int
        :param is_infer: 是否用于预测
        :type shape: bool
        '''
        self.num_classes = num_classes
        self.shape = shape
        self.is_infer = is_infer
        self.image_vector_size = shape[0] * shape[1]

        self.__declare_input_layers__()
        self.__build_nn__()

    def __declare_input_layers__(self):
        '''
        定义输入层
        '''
        # 图像输入为一个浮动向量
        self.image = paddle.layer.data(
            name='image',
            type=paddle.data_type.dense_vector(self.image_vector_size),
            # shape是(宽度,高度)
            height=self.shape[1],
            width=self.shape[0])

        # 将标签输入为ID列表
        if not self.is_infer:
            self.label = paddle.layer.data(
                name='label',
                type=paddle.data_type.integer_value_sequence(self.num_classes))
```
定义网络模型，该网络模型
首先是通过CNN获取图像的特征，
然后使用这些特征来输出展开成一系列特征向量，
然后使用RNN向前和向后捕获序列信息，
然后将RNN的输出映射到字符分布，
最后使用扭曲CTC来计算CTC任务的成本，获得了cost和额外层。
```python
def __build_nn__(self):
    '''
    建立网络拓扑
    '''
    # 通过CNN获取图像特征
    def conv_block(ipt, num_filter, groups, num_channels=None):
        return paddle.networks.img_conv_group(
            input=ipt,
            num_channels=num_channels,
            conv_padding=1,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act=paddle.activation.Relu(),
            conv_with_batchnorm=True,
            pool_size=2,
            pool_stride=2, )

    # 因为是灰度图所以最后一个参数是1
    conv1 = conv_block(self.image, 16, 2, 1)
    conv2 = conv_block(conv1, 32, 2)
    conv3 = conv_block(conv2, 64, 2)
    conv_features = conv_block(conv3, 128, 2)

    # 将CNN的输出展开成一系列特征向量。
    sliced_feature = paddle.layer.block_expand(
        input=conv_features,
        num_channels=128,
        stride_x=1,
        stride_y=1,
        block_x=1,
        block_y=11)

    # 使用RNN向前和向后捕获序列信息。
    gru_forward = paddle.networks.simple_gru(
        input=sliced_feature, size=128, act=paddle.activation.Relu())
    gru_backward = paddle.networks.simple_gru(
        input=sliced_feature,
        size=128,
        act=paddle.activation.Relu(),
        reverse=True)

    # 将RNN的输出映射到字符分布。
    self.output = paddle.layer.fc(input=[gru_forward, gru_backward],
                                  size=self.num_classes + 1,
                                  act=paddle.activation.Linear())

    self.log_probs = paddle.layer.mixed(
        input=paddle.layer.identity_projection(input=self.output),
        act=paddle.activation.Softmax())

    # 使用扭曲CTC来计算CTC任务的成本。
    if not self.is_infer:
        # 定义cost
        self.cost = paddle.layer.warp_ctc(
            input=self.output,
            label=self.label,
            size=self.num_classes + 1,
            norm_by_times=True,
            blank=self.num_classes)
        # 定义额外层
        self.eval = paddle.evaluator.ctc_error(input=self.output, label=self.label)
```
最后通过调用该类就可以获取到模型了，传入的参数是
`dict_size`是标签字典的大小，在上面有介绍是用来生成label的
`IMAGE_SHAPE`这个是图像的宽度和高度，格式是：(宽度,高度)
```python
model = Model(dict_size, IMAGE_SHAPE, is_infer=False)
```

# 生成训练器
-----
首先使用PaddlePaddle要先初始化PaddlePaddle，我们使用的是GPU，使用不了CPU，原因下面一部分会说到。
```python‘
# 初始化PaddlePaddle
paddle.init(use_gpu=True, trainer_count=1)
```
生成训练器在之前的例子中，我们知道要用到损失函数，训练参数和优化方法，这次我们多了一个额外层。
损失函数和额外层可以通过上一步的模型直接获取
```python
cost = model.cost
extra_layers = model.eval
```
这次的优化方法非常简单
```python
optimizer = paddle.optimizer.Momentum(momentum=0)
```
参数也可以通过上的损失函数生成
```python
params = paddle.parameters.create(model.cost)
```
最后结合这四个就可以生成一个训练器了
```python
trainer = paddle.trainer.SGD(cost=model.cost,
                             parameters=params,
                             update_equation=optimizer,
                             extra_layers=model.eval)
```

# 定义训练
-----------
经过上面获得的训练器，就可以开始训练了
```python
# 开始训练
trainer.train(reader=train_reader,
              feeding=feeding,
              event_handler=event_handler,
              num_passes=1000)
```
这个用到的`train_reader`就是在数据读取的时候获得的reader。
`feeding`是说明数据层之间的关系，定义如下：
```python
feeding = {'image': 0, 'label': 1}
```
训练事件`event_handler`，通过这个训练事件我们可以在训练的时候处理一下事情，如输出训练日志用于观察训练的效果，方便分析模型的性能。还可以保持模型，用于之后可预测或者再训练。定义如下：
```python
# 训练事件
def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 100 == 0:
            print("Pass %d, batch %d, Samples %d, Cost %f, Eval %s" %
                  (event.pass_id, event.batch_id, event.batch_id *
                   BATCH_SIZE, event.cost, event.metrics))

    if isinstance(event, paddle.event.EndPass):
        # 这里由于训练和测试数据共享相同的格式
        # 我们仍然使用reader.train_reader来读取测试数据
        test_reader = paddle.batch(
            my_reader.train_reader(test_file_list),
            batch_size=BATCH_SIZE)
        result = trainer.test(reader=test_reader, feeding=feeding)
        print("Test %d, Cost %f, Eval %s" % (event.pass_id, result.cost, result.metrics))
        # 检查保存model的路径是否存在，如果不存在就创建
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        with gzip.open(
                os.path.join(model_save_dir, "params_pass.tar.gz"), "w") as f:
            trainer.save_parameter_to_tar(f)
```
最后的`num_passes`就是训练轮数。

# 启动训练
----------
由官方文档可知，由于模型依赖的 warp CTC 只有CUDA的实现，本模型只支持 GPU 运行。所以读者要在自己的电脑安装paddlepaddle-gpu，如果读者的电脑是有GPU的话。
由于笔者的电脑没有GPU，所以不得不使用云服务器来训练我们的模型。笔者使用的是[百度深度学习GPU集群](https://cloud.baidu.com/product/bdl.html)，这有个非常好的地方就是购买来的服务器就已经安装了PaddlePaddle，无需我们再安装了，这省去了很多时间。不过笔者在使用的时候，出现了找不到`libwarpctc.so`这个库，所以要自己动手去安装该库，如果读者没有报该错，请忽略以下操作：
## 安装libwarpctc.so库
先从GitHub上获取源码
```
git clone https://github.com/baidu-research/warp-ctc.git
cd warp-ctc
```
创建build目录
```
mkdir build
cd build
```
默认是没有安装`cmake`的，所以要先安装`cmake`
```
apt install cmake
```
安装完成之后就可以cmake和编译了，这里的编译笔者使用6个线程，这个会快一点
```
cmake ../
make -j6
```
编译完成之后，就生成了一个`libwarpctc.so`，这个就是我们需要的库，执行以下命令，将其复制到相应的目录
```
cp libwarpctc.so /usr/lib/x86_64-linux-gnu/
```
最后测试一下是否正常了
```
./test_gpu
```
## 执行训练main方法
通过上面的操作，训练的程序就已经完成了，可以启动训练了
```python
if __name__ == "__main__":
    # 训练列表的的路径
    train_file_list_path = '../data/train_data/trainer.list'
    # 测试列表的路径
    test_file_list_path = '../data/test_data/test.list'
    # 标签字典的路径
    label_dict_path = '../data/label_dict.txt'
    # 保存模型的路径
    model_save_dir = '../models'
    train(train_file_list_path, test_file_list_path, label_dict_path, model_save_dir)
```
输出的日志大概如下：
```text
Pass 0, batch 0, Samples 0, Cost 16.149542, Eval {}
Pass 0, batch 100, Samples 1000, Cost 15.090727, Eval {}
Test 0, Cost 15.079704, Eval {}
Pass 1, batch 0, Samples 0, Cost 14.775064, Eval {}
Pass 1, batch 100, Samples 1000, Cost 15.448521, Eval {}
Test 1, Cost 14.826180, Eval {}
```

# 开始预测
-------
通过之前的训练，我们有了训练参数，可以使用这些参数进行预测了。
```python
def infer(img_path, model_path, image_shape, label_dict_path):
    # 获取标签字典
    char_dict = load_dict(label_dict_path)
    # 获取反转的标签字典
    reversed_char_dict = load_reverse_dict(label_dict_path)
    # 获取字典大小
    dict_size = len(char_dict)
    # 获取reader
    my_reader = Reader(char_dict=char_dict, image_shape=image_shape)
    # 初始化PaddlePaddle
    paddle.init(use_gpu=True, trainer_count=1)
    # 加载训练好的参数
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_path))
    # 获取网络模型
    model = Model(dict_size, image_shape, is_infer=True)
    # 获取预测器
    inferer = paddle.inference.Inference(output_layer=model.log_probs, parameters=parameters)
    # 加载数据
    test_batch = [[my_reader.load_image(img_path)]]
    # 开始预测
    return start_infer(inferer, test_batch, reversed_char_dict)
```
上面使用的反转的标签字典定义如下，通过标签字典的文件即可生成反转的标签字典
```python
def load_reverse_dict(dict_path):
    """
    从字典路径加载反转的标签字典
    :param dict_path: 标签字典的路径
    :type dict_path: str
    """
    return dict((idx, line.strip().split("\t")[0])
                for idx, line in enumerate(open(dict_path, "r").readlines()))
```
通过传入上面获取是的inferer和图像的一维向量，还有反转的标签字典就可以进行预测了。
```python
def start_infer(inferer, test_batch, reversed_char_dict):
    # 获取初步预测结果
    infer_results = inferer.infer(input=test_batch)
    num_steps = len(infer_results) // len(test_batch)
    probs_split = [
        infer_results[i * num_steps:(i + 1) * num_steps]
        for i in range(0, len(test_batch))]
    # 最佳路径解码
    result = ''
    for i, probs in enumerate(probs_split):
        result = ctc_greedy_decoder(
            probs_seq=probs, vocabulary=reversed_char_dict)
    return result
```
这个还使用到了最佳路径解码，使用的解码器如下：
```python
def ctc_greedy_decoder(probs_seq, vocabulary):
    """CTC贪婪（最佳路径）解码器。
    由最可能的令牌组成的路径被进一步后处理
    删除连续的重复和所有的空白。
    :param probs_seq: 每个词汇表上概率的二维列表字符。
                      每个元素都是浮点概率列表为一个字符。
    :type probs_seq: list
    :param vocabulary: 词汇表
    :type vocabulary: list
    :return: 解码结果字符串
    :rtype: baseline
    """
    # 尺寸验证
    for probs in probs_seq:
        if not len(probs) == len(vocabulary) + 1:
            raise ValueError("probs_seq dimension mismatchedd with vocabulary")
    # argmax以获得每个时间步长的最佳指标
    max_index_list = list(np.array(probs_seq).argmax(axis=1))
    # 删除连续的重复索引
    index_list = [index_group[0] for index_group in groupby(max_index_list)]
    # 删除空白索引
    blank_index = len(vocabulary)
    index_list = [index for index in index_list if index != blank_index]
    # 将索引列表转换为字符串
    return ''.join([vocabulary[index] for index in index_list])
```
最后在main方法中直接运行预测程序就可以了。
```python
if __name__ == "__main__":
    # 要预测的图像
    img_path = '../data/test_data/4uqh.png'
    # 模型的路径
    model_path = '../models/params_pass.tar.gz'
    # 图像的大小
    image_shape = (72, 27)
    # 标签的路径
    label_dict_path = '../data/label_dict.txt'
    # 获取预测结果
    result = infer(img_path, model_path, image_shape, label_dict_path)
    print '预测结果：%s' % result
```
预测输出
```text
预测结果：4uqh
```

# 项目代码
-------
GitHub地址:https://github.com/yeyupiaoling/LearnPaddle

# 参考资料
-------
1. http://paddlepaddle.org/
2. http://blog.csdn.net/qq_26819733/article/details/53608308
3. https://github.com/baidu-research/warp-ctc
