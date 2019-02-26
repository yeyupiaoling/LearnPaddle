# 目录
@[toc]
*本篇文章基于 PaddlePaddle 0.10.0、Python 2.7
# 前言
-------
在前两篇文章[验证码端到端的识别](http://blog.csdn.net/qq_33200967/article/details/79233565)和[车牌端到端的识别](http://blog.csdn.net/qq_33200967/article/details/79095335)这两篇文章中其实就使用到了场景文字识别了，在本篇中就针对场景文字识别这个问题好好说说。
场景文字识别到底有什么用呢，说得大一些在自动驾驶领域，公路上总会有很多的路牌和标识，这些路牌标识通常会有很多的文字说明，我们就要识别这些文字来了解它们的含义。还有老师在黑板上写的笔记，如果使用场景文字识别技术，我们直接拍个照，直接识别黑板中的文字内容，就可以省去很多抄笔记时间了。

# 数据集的介绍
--------
场景文字是怎样的呢，来看看这张图像
![这里写图片描述](http://img.blog.csdn.net/20180209223806972?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
这张图像中包含了大量的文字，我们要做的就是把这些文字识别出来。这张图像是[SynthText in the Wild Dataset](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)，这个数据集非常大，有41G。为了方便学习，我们在本项目中使用这个数据集，而是使用更小的[Task 2.3: Word Recognition (2013 edition)](http://rrc.cvc.uab.es/?ch=2&com=introduction)，这个数据集的训练数据和测试数据一共也就160M左右，非常适合我们做学习使用，该数据集的图像如下：
![这里写图片描述](http://img.blog.csdn.net/20180210201557391?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

# 数据的读取
------
官方给出的数据读取列表有两个，一个是训练数据的图像列表`gt.txt`，另一个是测试数据的图像列表`Challenge2_Test_Task3_GT.txt`。它们的格式如下：
```text
word_1.png, "Tiredness"
word_2.png, "kills"
word_3.png, "A"
word_4.png, "short"
word_5.png, "break"
word_6.png, "could"
```
前面的`word_1.png`是图像的路径，后面的`Tiredness`是图像包含的文字内容。
基于这个数据格式，我们要编写一个工具类来读取这些数据信息。
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
            line_split = line.strip().split(',', 1)
            filename = line_split[0].strip()
            path = os.path.join(dirname, filename)
            label = line_split[1][2:-1].strip()
            if label:
                path_list.append((path, label))

    return path_list
```
然后通过调用该方法就可以那到数据的信息了 ，通过这些数据就可以生成训练和测试用的reader了。
```python
# coding=utf-8
import os
import cv2
from paddle.v2.image import load_image

class DataGenerator(object):
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
        加载图像并将其转换为一维向量
        :param path: 图像数据的路径
        :type path: str
        '''
        image = load_image(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 将所有图像调整为固定形状
        if self.image_shape:
            image = cv2.resize(
                image, self.image_shape, interpolation=cv2.INTER_CUBIC)

        image = image.flatten() / 255.
        return image
```
从上面的代码你可能留意到这里使用的label是标签字典的value，所以我们要对在训练时出现的字符做一个标签字典，如下格式`字符	出现次数`：
```text
e	286
E	257
a	199
S	189
n	185
```
生成的标签字典的代码如下，使用到的数据就是上面通过路径和label拿到的list。
```text
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
生成了标签字典之后，就要拿这些标签字典来给`DataGenerator`生成训练所需要的`reader`，代码如下：
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
最后通过调用PaddlePaddle的API就可以生成`trainer`使用的`reader`。
```python
reader=paddle.batch(
            paddle.reader.shuffle(
                data_generator.train_reader(train_file_list),
                buf_size=conf.buf_size),
            batch_size=conf.batch_size)
```
获得的reader的可以`trainer.train`训练的时候传给训练器。
```python
# 开始训练
trainer.train(
    reader=reader,
    feeding=feeding,
    event_handler=event_handler,
    num_passes=conf.num_passes)
```
上面就是开始训练的代码，但是现在还不能直接开始训练，我们的训练器`trainer`还没有定义，接下来就介绍训练器的定义。

# 定义训练器
---------
通过调用PaddlePaddle的接口`paddle.trainer.SGD`就可以生成一个训练器`trainer`了
```python
trainer = paddle.trainer.SGD(cost=model.cost,
                             parameters=params,
                             update_equation=optimizer,
                             extra_layers=model.eval)
```
## 定义神经网络模型
在定义训练器的时候，需要用到参数`cost`和`extra_layers`都要用到神经网络模型来生成这两参数的值，所以还要先定义一个神经网络模型。
首先先要定义数据的大小和label，这定义数据的大小时，因为数据是个长方形，所以还有说明宽度和高度。
```python
# 图像输入为一个浮动向量
self.image = layer.data(
    name='image',
    type=paddle.data_type.dense_vector(self.image_vector_size),
    height=self.shape[1],
    width=self.shape[0])

# 将标签输入为ID列表
if not self.is_infer:
    self.label = layer.data(
        name='label',
        type=paddle.data_type.integer_value_sequence(self.num_classes))
```
然后通过卷积神经网络获取图像特征
```python
    def conv_groups(self, input, num, with_bn):
        '''
        用图像卷积组获得图像特征。
        :param input: 输入层
        :type input: LayerOutput
        :param num: 过滤器的数量。
        :type num: int
        :param with_bn: 是否使用BN层
        :type with_bn: bool
        '''
        assert num % 4 == 0

        filter_num_list = conf.filter_num_list
        is_input_image = True
        tmp = input

        for num_filter in filter_num_list:
            # 因为是灰度图所以num_channels参数是1
            if is_input_image:
                num_channels = 1
                is_input_image = False
            else:
                num_channels = None

            tmp = img_conv_group(
                input=tmp,
                num_channels=num_channels,
                conv_padding=conf.conv_padding,
                conv_num_filter=[num_filter] * (num / 4),
                conv_filter_size=conf.conv_filter_size,
                conv_act=Relu(),
                conv_with_batchnorm=with_bn,
                pool_size=conf.pool_size,
                pool_stride=conf.pool_stride, )

        return tmp
```
然后通过这些图像的特征张开成特征向量
```python
# 通过CNN获取图像特征
conv_features = self.conv_groups(self.image, conf.filter_num,
                                 conf.with_bn)

# 将CNN的输出展开成一系列特征向量。
sliced_feature = layer.block_expand(
    input=conv_features,
    num_channels=conf.num_channels,
    stride_x=conf.stride_x,
    stride_y=conf.stride_y,
    block_x=conf.block_x,
    block_y=conf.block_y)
```
然后将RNN的输出映射到字符分布
```python
# 使用RNN向前和向后捕获序列信息。
gru_forward = simple_gru(
    input=sliced_feature, size=conf.hidden_size, act=Relu())
gru_backward = simple_gru(
    input=sliced_feature,
    size=conf.hidden_size,
    act=Relu(),
    reverse=True)

# 将RNN的输出映射到字符分布。
self.output = layer.fc(input=[gru_forward, gru_backward],
                       size=self.num_classes + 1,
                       act=Linear())

self.log_probs = paddle.layer.mixed(
    input=paddle.layer.identity_projection(input=self.output),
    act=paddle.activation.Softmax())
```
最后就可以开始拿`cost`和`extra_layers`了，
```python
if not self.is_infer:
    self.cost = layer.warp_ctc(
        input=self.output,
        label=self.label,
        size=self.num_classes + 1,
        norm_by_times=conf.norm_by_times,
        blank=self.num_classes)

    self.eval = evaluator.ctc_error(input=self.output, label=self.label)
```
## 生成训练器
使用`cost`还可以生成训练参数
```python
# 创建训练参数
params = paddle.parameters.create(model.cost)
```
最后还缺一个优化方法
```python
# 创建训练参数
optimizer = paddle.optimizer.Momentum(momentum=conf.momentum)
```
这样四个参数`cost`，`parameters`，`update_equation`，`extra_layers`我们都拿到了。可以创建一个训练器了。

# 开始训练
--------
训练模型一共要4个参数，到目前为止，我们只拿到一个`reader`参数，还有另外`feeding`，`event_handler`，`num_passes`这三个参数。
定义数据层之间的关系
```python
# 说明数据层之间的关系
feeding = {'image': 0, 'label': 1}
```
定义训练事件，让它在训练训练的过程中输出一下日志信息，观察我们模型的收敛情况。
```python
# 训练事件
def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % conf.log_period == 0:
            print("Pass %d, batch %d, Samples %d, Cost %f, Eval %s" %
                  (event.pass_id, event.batch_id, event.batch_id *
                   conf.batch_size, event.cost, event.metrics))

    if isinstance(event, paddle.event.EndPass):
        # 这里由于训练和测试数据共享相同的格式
        # 我们仍然使用reader.train_reader来读取测试数据
        result = trainer.test(
            reader=paddle.batch(
                data_generator.train_reader(test_file_list),
                batch_size=conf.batch_size),
            feeding=feeding)
        print("Test %d, Cost %f, Eval %s" %
              (event.pass_id, result.cost, result.metrics))
        with gzip.open(
                os.path.join(model_save_dir, "params_pass.tar.gz"), "w") as f:
            trainer.save_parameter_to_tar(f)
```
说明训练的轮数
```python
num_passes=conf.num_passes
```
在训练之前还要初始化PaddlePaddle
```python
# 初始化PaddlePaddle
paddle.init(use_gpu=conf.use_gpu, trainer_count=conf.trainer_count) 
```
在训练的过程中会输入一下日志信息：
```text
Pass 0, batch 0, Samples 0, Cost 39.119792, Eval {}
Test 0, Cost 35.374924, Eval {}
Pass 1, batch 0, Samples 0, Cost 30.138696, Eval {}
Test 1, Cost 21.629668, Eval {}
Pass 2, batch 0, Samples 0, Cost 21.412227, Eval {}
Test 2, Cost 22.698648, Eval {}
Pass 3, batch 0, Samples 0, Cost 22.565864, Eval {}
Test 3, Cost 21.634227, Eval {}
```

# 开始预测
--------
通过之前的训练，我们有了训练参数，可以使用这些参数进行预测了。
```python
def infer(model_path, image_shape, label_dict_path,infer_file_list_path):

    infer_file_list = get_file_list(infer_file_list_path)
    # 获取标签字典
    char_dict = load_dict(label_dict_path)
    # 获取反转的标签字典
    reversed_char_dict = load_reverse_dict(label_dict_path)
    # 获取字典大小
    dict_size = len(char_dict)
    # 获取reader
    data_generator = DataGenerator(char_dict=char_dict, image_shape=image_shape)
    # 初始化PaddlePaddle
    paddle.init(use_gpu=True, trainer_count=2)
    # 加载训练好的参数
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_path))
    # 获取网络模型
    model = Model(dict_size, image_shape, is_infer=True)
    # 获取预测器
    inferer = paddle.inference.Inference(output_layer=model.log_probs, parameters=parameters)
    # 开始预测
    test_batch = []
    labels = []
    for i, (image, label) in enumerate(data_generator.infer_reader(infer_file_list)()):
        test_batch.append([image])
        labels.append(label)
    infer_batch(inferer, test_batch, labels, reversed_char_dict)
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
def infer_batch(inferer, test_batch, labels, reversed_char_dict):
    # 获取初步预测结果
    infer_results = inferer.infer(input=test_batch)
    num_steps = len(infer_results) // len(test_batch)
    probs_split = [
        infer_results[i * num_steps:(i + 1) * num_steps]
        for i in xrange(0, len(test_batch))
    ]
    results = []
    # 最佳路径解码
    for i, probs in enumerate(probs_split):
        output_transcription = ctc_greedy_decoder(
            probs_seq=probs, vocabulary=reversed_char_dict)
        results.append(output_transcription)
    # 打印预测结果
    for result, label in zip(results, labels):
        print("\n预测结果: %s\n实际文字: %s" %(result, label))
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
    infer_file_list_path = '../data/test_data/Challenge2_Test_Task3_GT.txt'
    # 模型的路径
    model_path = '../models/params_pass.tar.gz'
    # 图像的大小
    image_shape = (173, 46)
    # 标签的路径
    label_dict_path = '../data/label_dict.txt'
    # 开始预测
    infer(model_path, image_shape, label_dict_path, infer_file_list_path)
```
预测的结果：
```text
预测结果: FFt
实际文字: PROPER

预测结果: FD
实际文字: FOOD

预测结果: F:
实际文字: PRONTO

预测结果: 6vdt:tdnd
实际文字: professional

预测结果: La
实际文字: Java
```
从预测结果来看，模型效果并不是很理想，错误了非常高，这个数据量并不是很大，所以模型收敛的不是很好，也很容易出现过拟合现象。笔者加正则效果也不明显，读者可以自己在`config.py`这个文件中修改网络模型和训练器的配置，尝试是模型收敛得更好，也可以选择更大的数据来解决这个问题。




# 项目代码
--------
GitHub地址:https://github.com/yeyupiaoling/LearnPaddle

# 参考资料
---------
 1. http://paddlepaddle.org/
 2. http://www.robots.ox.ac.uk/~vgg/data/scenetext/
 3. http://rrc.cvc.uab.es/?ch=2&com=introduction
