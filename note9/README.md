# 目录
@[toc]
*本篇文章基于 PaddlePaddle 0.11.0、Python 2.7
# 前言
---------
**目标检测**的使用范围很广，比如我们使用相机拍照时，要正确检测人脸的位置，从而做进一步处理，比如美颜等等。在目标检测的深度学习领域上，从2014年到2016年，先后出现了R-CNN，Fast R-CNN, Faster R-CNN, ION, HyperNet, SDP-CRC, YOLO,G-CNN, SSD等神经网络模型，使得目标检测不管是在准确度上，还是速度上都有很大提高，几乎可以达到实时检测。

# VOC数据集
-------
## VOC数据集介绍
PASCAL VOC挑战赛是视觉对象的分类识别和检测的一个基准测试，提供了检测算法和学习性能的标准图像注释数据集和标准的评估系统。
PASCAL VOC图片集包括20个目录：

 - 人类； 动物(鸟、猫、牛、狗、马、羊)； 
 - 交通工具(飞机、自行车、船、公共汽车、小轿车、摩托车、火车)；
 - 室内(瓶子、椅子、餐桌、盆栽植物、沙发、电视)。
 
这些类别在`data/label_list`文件中都有列出来，但这个文件中多了一个类别，就是背景（background）
## 下载VOC数据集
可以通过以下命令下载数据集
```python
# 切换到项目的数据目录
cd data
# 下载2007年的训练数据
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
# 下载2007年的测试数据
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
# 下载2012年的训练数据
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
## 解压数据集
下载完成之后，要解压数据集到当前目录
```python
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtrainval_11-May-2012.tar
```
解压之后会得到一个目录，其中我们实质只用到`Annotations(标注文件)`和`JPEGImages(图像文件)`下的文件。
```text
VOCdevkit
    |____VOC2007
    |      |____Annotations(标注文件)
    |      |____JPEGImages(图像文件)
    |      |____ImageSets
    |      |____SegmentationClass
    |      |____SegmentationObject
    |
    |____VOC2012
           |____Annotations(标注文件)
           |____JPEGImages(图像文件)
           |____ImageSets
           |____SegmentationClass
           |____SegmentationObject
```
## 生成图像列表
我们要编写一个程序`data/prepare_voc_data.py`，把这些数据生成一个图像列表，就像之前的图像列表差不多，每一行对应的是图像的路径和标签。这次有点不同的是对应的不是`int`类型的label了，是一个`xml`的标注文件。其部分代码片段如下：
```python
def prepare_filelist(devkit_dir, years, output_dir):
    trainval_list = []
    test_list = []
    # 获取两个年份的数据
    for year in years:
        trainval, test = walk_dir(devkit_dir, year)
        trainval_list.extend(trainval)
        test_list.extend(test)
    # 打乱训练数据
    random.shuffle(trainval_list)
    # 保存训练图像列表
    with open(os.path.join(output_dir, 'trainval.txt'), 'w') as ftrainval:
        for item in trainval_list:
            ftrainval.write(item[0] + ' ' + item[1] + '\n')
    # 保存测试图像列表
    with open(os.path.join(output_dir, 'test.txt'), 'w') as ftest:
        for item in test_list:
            ftest.write(item[0] + ' ' + item[1] + '\n')

if __name__ == '__main__':
    # 数据存放的位置
    devkit_dir = 'VOCdevkit'
    # 数据的年份
    years = ['2007', '2012']
    prepare_filelist(devkit_dir, years, '.')
```
通过上面的程序，就可以生成一个图像列表，列表片段如下：
```text
VOCdevkit/VOC2007/JPEGImages/000001.jpg VOCdevkit/VOC2007/Annotations/000001.xml
VOCdevkit/VOC2007/JPEGImages/000002.jpg VOCdevkit/VOC2007/Annotations/000002.xml
VOCdevkit/VOC2007/JPEGImages/000003.jpg VOCdevkit/VOC2007/Annotations/000003.xml
VOCdevkit/VOC2007/JPEGImages/000004.jpg VOCdevkit/VOC2007/Annotations/000004.xml
```
数据集的操作就到这里了

# 数据预处理
--------
在之前的文章中可以知道，训练和测试的数据都是一个reader数据格式，所以我们要对我们的VOC数据集做一些处理。跟之前最大的不同是这次的标签不是简单的`int`或者是一个字符串，而是一个标注`XML`文件。而且训练的图像大小必须是统一大小的，但是实际的图像的大小是不固定的，如果改变了图像的大小，那么图像的标注信息就不正确了，所以对图像的大小修改同时，也要对标注信息做对应的变化。
获取标注信息的代码片段：
```python
# 保存列表的结构: label | xmin | ymin | xmax | ymax | difficult
if mode == 'train' or mode == 'test':
    # 保存每个标注框
    bbox_labels = []
    # 开始读取标注信息
    root = xml.etree.ElementTree.parse(label_path).getroot()
    # 查询每个标注的信息
    for object in root.findall('object'):
        # 每个标注框的信息
        bbox_sample = []
        # start from 1
        bbox_sample.append(
            float(
                settings.label_list.index(
                    object.find('name').text)))
        bbox = object.find('bndbox')
        difficult = float(object.find('difficult').text)
        # 获取标注信息，并计算比例保存
        bbox_sample.append(
            float(bbox.find('xmin').text) / img_width)
        bbox_sample.append(
            float(bbox.find('ymin').text) / img_height)
        bbox_sample.append(
            float(bbox.find('xmax').text) / img_width)
        bbox_sample.append(
            float(bbox.find('ymax').text) / img_height)
        bbox_sample.append(difficult)
        # 将整个框的信息保存
        bbox_labels.append(bbox_sample)
```
获取了标注信息并计算保存了标注信息，然后根据图像的原始大小和标注信息的比例，可以裁剪图像的标注信息对应的图像。
```python
def crop_image(img, bbox_labels, sample_bbox, image_width, image_height):
    '''
    裁剪图像
    :param img: 图像
    :param bbox_labels: 所有的标注信息
    :param sample_bbox: 对应一个的标注信息
    :param image_width: 图像原始的宽
    :param image_height: 图像原始的高
    :return:裁剪好的图像和其对应的标注信息
    '''
    sample_bbox = clip_bbox(sample_bbox)
    xmin = int(sample_bbox.xmin * image_width)
    xmax = int(sample_bbox.xmax * image_width)
    ymin = int(sample_bbox.ymin * image_height)
    ymax = int(sample_bbox.ymax * image_height)
    sample_img = img[ymin:ymax, xmin:xmax]
    sample_labels = transform_labels(bbox_labels, sample_bbox)
    return sample_img, sample_labels
```
然后使用这些图像就可以使用训练或者测试要使用的reader的了，代码片段如下：
```python
def reader():
    img = Image.fromarray(img)
    # 设置图像大小
    img = img.resize((settings.resize_w, settings.resize_h),
                     Image.ANTIALIAS)
    img = np.array(img)

    if mode == 'train':
        mirror = int(random.uniform(0, 2))
        if mirror == 1:
            img = img[:, ::-1, :]
            for i in xrange(len(sample_labels)):
                tmp = sample_labels[i][1]
                sample_labels[i][1] = 1 - sample_labels[i][3]
                sample_labels[i][3] = 1 - tmp

    if len(img.shape) == 3:
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 1, 0)

    img = img.astype('float32')
    img -= settings.img_mean
    img = img.flatten()

    if mode == 'train' or mode == 'test':
        if mode == 'train' and len(sample_labels) == 0: continue
        yield img.astype('float32'), sample_labels
    elif mode == 'infer':
        yield img.astype('float32')
    return reader
```
最后通过调用PaddlePaddle的借口就可以生成训练和测试使用的最终`reader`，代码如下：
```python
# 创建训练数据
train_reader = paddle.batch(
    data_provider.train(data_args, train_file_list),
    batch_size=cfg.TRAIN.BATCH_SIZE)
# 创建测试数据
dev_reader = paddle.batch(
    data_provider.test(data_args, dev_file_list),
    batch_size=cfg.TRAIN.BATCH_SIZE)
```

# SSD神经网络
-----
## SSD原理
SSD使用一个卷积神经网络实现“端到端”的检测：输入为原始图像，输出为检测结果，无需借助外部工具或流程进行特征提取、候选框生成等。论文中SSD使用VGG16作为基础网络进行图像特征提取。但SSD对原始VGG16网络做了一些改变：

 1. 将最后的fc6、fc7全连接层变为卷积层，卷积层参数通过对原始fc6、fc7参数采样得到。
 2. 将pool5层的参数由2x2-s2（kernel大小为2x2，stride size为2）更改为3x3-s1-p1（kernel大小为3x3，stride size为1，padding size为1）。
 3. 在conv4_3、conv7、conv8_2、conv9_2、conv10_2及pool11层后面接了priorbox层，priorbox层的主要目的是根据输入的特征图（feature map）生成一系列的矩形候选框。
下图为模型（输入图像尺寸：300x300）的总体结构：
![SSD 网络结构](http://www.paddlepaddle.org/docs/develop/models/ssd/images/ssd_network.png)
图1. SSD 网络结构

图中每个矩形盒子代表一个卷积层，最后两个矩形框分别表示汇总各卷积层输出结果和后处理阶段。在预测阶段，网络会输出一组候选矩形框，每个矩形包含：位置和类别得分。图中倒数第二个矩形框即表示网络的检测结果的汇总处理。由于候选矩形框数量较多且很多矩形框重叠严重，这时需要经过后处理来筛选出质量较高的少数矩形框，主要方法有非极大值抑制（Non-maximum Suppression）。

从SSD的网络结构可以看出，候选矩形框在多个特征图（feature map）上生成，不同的feature map具有的感受野不同，这样可以在不同尺度扫描图像，相对于其他检测方法可以生成更丰富的候选框，从而提高检测精度；另一方面SSD对VGG16的扩展部分以较小的代价实现对候选框的位置和类别得分的计算，整个过程只需要一个卷积神经网络完成，所以速度较快。

以上介绍摘自[PaddlePaddle官网的教程](http://www.paddlepaddle.org/docs/develop/models/ssd/README.cn.html#permalink-2-ssd-)
## SSD代码介绍
如上介绍所说，SSD使用VGG16作为基础网络进行图像特征提取
```python
# 卷积神经网络
def conv_group(stack_num, name_list, input, filter_size_list, num_channels,
               num_filters_list, stride_list, padding_list,
               common_bias_attr, common_param_attr, common_act):
    conv = input
    in_channels = num_channels
    for i in xrange(stack_num):
        conv = paddle.layer.img_conv(
            name=name_list[i],
            input=conv,
            filter_size=filter_size_list[i],
            num_channels=in_channels,
            num_filters=num_filters_list[i],
            stride=stride_list[i],
            padding=padding_list[i],
            bias_attr=common_bias_attr,
            param_attr=common_param_attr,
            act=common_act)
        in_channels = num_filters_list[i]
    return conv

# VGG神经网络
def vgg_block(idx_str, input, num_channels, num_filters, pool_size,
              pool_stride, pool_pad):
    layer_name = "conv%s_" % idx_str
    stack_num = 3
    name_list = [layer_name + str(i + 1) for i in xrange(3)]

    conv = conv_group(stack_num, name_list, input, [3] * stack_num,
                      num_channels, [num_filters] * stack_num,
                      [1] * stack_num, [1] * stack_num, default_bias_attr,
                      get_param_attr(1, default_l2regularization),
                      paddle.activation.Relu())

    pool = paddle.layer.img_pool(
        input=conv,
        pool_size=pool_size,
        num_channels=num_filters,
        pool_type=paddle.pooling.CudnnMax(),
        stride=pool_stride,
        padding=pool_pad)
    return conv, pool
```
将最后的fc6、fc7全连接层变为卷积层，卷积层参数通过对原始fc6、fc7参数采样得到：
```python
fc7 = conv_group(stack_num, ['fc6', 'fc7'], pool5, [3, 1], 512, [1024] *
                 stack_num, [1] * stack_num, [1, 0], default_bias_attr,
                 get_param_attr(1, default_l2regularization),
                 paddle.activation.Relu())
```
将pool5层的参数由2x2-s2（kernel大小为2x2，stride size为2）更改为3x3-s1-p1（kernel大小为3x3，stride size为1，padding size为1）：
```python
def mbox_block(layer_idx, input, num_channels, filter_size, loc_filters,
               conf_filters):
    mbox_loc_name = layer_idx + "_mbox_loc"
    mbox_loc = paddle.layer.img_conv(
        name=mbox_loc_name,
        input=input,
        filter_size=filter_size,
        num_channels=num_channels,
        num_filters=loc_filters,
        stride=1,
        padding=1,
        bias_attr=default_bias_attr,
        param_attr=get_param_attr(1, default_l2regularization),
        act=paddle.activation.Identity())

    mbox_conf_name = layer_idx + "_mbox_conf"
    mbox_conf = paddle.layer.img_conv(
        name=mbox_conf_name,
        input=input,
        filter_size=filter_size,
        num_channels=num_channels,
        num_filters=conf_filters,
        stride=1,
        padding=1,
        bias_attr=default_bias_attr,
        param_attr=get_param_attr(1, default_l2regularization),
        act=paddle.activation.Identity())
    return mbox_loc, mbox_conf
```
最后要获取到训练和预测使用到的损失函数和检查输出层
```python
if mode == 'train' or mode == 'eval':
    bbox = paddle.layer.data(
        name='bbox', type=paddle.data_type.dense_vector_sequence(6))
    loss = paddle.layer.multibox_loss(
        input_loc=loc_loss_input,
        input_conf=conf_loss_input,
        priorbox=mbox_priorbox,
        label=bbox,
        num_classes=cfg.CLASS_NUM,
        overlap_threshold=cfg.NET.MBLOSS.OVERLAP_THRESHOLD,
        neg_pos_ratio=cfg.NET.MBLOSS.NEG_POS_RATIO,
        neg_overlap=cfg.NET.MBLOSS.NEG_OVERLAP,
        background_id=cfg.BACKGROUND_ID,
        name="multibox_loss")
    paddle.evaluator.detection_map(
        input=detection_out,
        label=bbox,
        overlap_threshold=cfg.NET.DETMAP.OVERLAP_THRESHOLD,
        background_id=cfg.BACKGROUND_ID,
        evaluate_difficult=cfg.NET.DETMAP.EVAL_DIFFICULT,
        ap_type=cfg.NET.DETMAP.AP_TYPE,
        name="detection_evaluator")
    return loss, detection_out
elif mode == 'infer':
    return detection_out
```
关于SSD神经网络介绍就到这里，如果读者想跟详细了解SSD神经网络，可以阅读SSD的论文 [SSD: Single shot multibox detector](https://arxiv.org/abs/1512.02325)

# 训练模型
----------
## 训练流程图
训练的流程图：
```mermaid
flowchat 
st=>start: 开始 
e=>end: 结束 
get_trainer=>operation: 创建训练器
start_trainer=>operation: 开始训练

st->get_trainer->start_trainer->e
```
## 创建训练器
创建训练器，代码片段如下：
```python
# 创建优化方法
optimizer = paddle.optimizer.Momentum(
    momentum=cfg.TRAIN.MOMENTUM,
    learning_rate=cfg.TRAIN.LEARNING_RATE,
    regularization=paddle.optimizer.L2Regularization(
        rate=cfg.TRAIN.L2REGULARIZATION),
    learning_rate_decay_a=cfg.TRAIN.LEARNING_RATE_DECAY_A,
    learning_rate_decay_b=cfg.TRAIN.LEARNING_RATE_DECAY_B,
    learning_rate_schedule=cfg.TRAIN.LEARNING_RATE_SCHEDULE)

# 通过神经网络模型获取损失函数和额外层
cost, detect_out = vgg_ssd_net.net_conf('train')
# 通过损失函数创建训练参数
parameters = paddle.parameters.create(cost)
# 如果有训练好的模型，可以使用训练好的模型再训练
if not (init_model_path is None):
    assert os.path.isfile(init_model_path), 'Invalid model.'
    parameters.init_from_tar(gzip.open(init_model_path))
# 创建训练器
trainer = paddle.trainer.SGD(cost=cost,
                             parameters=parameters,
                             extra_layers=[detect_out],
                             update_equation=optimizer)
```
## 开始训练
有了训练器，我们才可以开始训练。如果单纯让它训练，没做一些数据保存处理，这种训练是没有意义的，所以我们要定义一个训练事件，让它在训练过程中保存我们需要的模型参数，同时输出一些日志信息，方便我们查看训练的效果，训练事件的代码片段：
```python
# 定义训练事件
def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 1 == 0:
            print "\nPass %d, Batch %d, TrainCost %f, Detection mAP=%f" % \
                    (event.pass_id,
                     event.batch_id,
                     event.cost,
                     event.metrics['detection_evaluator'])
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

    if isinstance(event, paddle.event.EndPass):
        with gzip.open('../models/params_pass.tar.gz', 'w') as f:
            trainer.save_parameter_to_tar(f)
        result = trainer.test(reader=dev_reader, feeding=feeding)
        print "\nTest with Pass %d, TestCost: %f, Detection mAP=%g" % \
                (event.pass_id,
                 result.cost,
                 result.metrics['detection_evaluator'])
```
最后就可以进行训练了，训练的代码为：
```python
# 开始训练
trainer.train(
    reader=train_reader,
    event_handler=event_handler,
    num_passes=cfg.TRAIN.NUM_PASS,
    feeding=feeding)
```
具体调用方法如下，`train_file_list`为训练数据；`dev_file_list`为测试数据；`data_args`为数据集的设置；`init_model_path`为初始化模型参数，在第三章[CIFAR彩色图像识别](http://blog.csdn.net/qq_33200967/article/details/79095224)我们就谈到SSD神经网络很容易发生浮点异常，所以我们要一个预训练的模型来提供初始化训练参数，笔者使用的是PaddlePaddle官方提供的[预训练的模型](http://paddlemodels.bj.bcebos.com/v2/vgg_model.tar.gz)：
```python
if __name__ == "__main__":
    # 初始化PaddlePaddle
    paddle.init(use_gpu=True, trainer_count=2)
    # 设置数据参数
    data_args = data_provider.Settings(
        data_dir='../data',
        label_file='../data/label_list',
        resize_h=cfg.IMG_HEIGHT,
        resize_w=cfg.IMG_WIDTH,
        mean_value=[104, 117, 124])
    # 开始训练
    train(
        train_file_list='../data/trainval.txt',
        dev_file_list='../data/test.txt',
        data_args=data_args,
        init_model_path='../models/vgg_model.tar.gz')
```
在训练过程中会输出以下训练日志：
```text
Pass 0, Batch 0, TrainCost 17.445816, Detection mAP=0.000000
...................................................................................................
Pass 0, Batch 100, TrainCost 8.544815, Detection mAP=2.871136
...................................................................................................
Pass 0, Batch 200, TrainCost 7.434404, Detection mAP=3.337185
...................................................................................................
Pass 0, Batch 300, TrainCost 7.404398, Detection mAP=7.070700
...................................................................................................
Pass 0, Batch 400, TrainCost 7.023655, Detection mAP=3.080483
```

# 评估模型
--------
我们训练好的模型之后，在使用模式进行预测，可以对模型进行评估。评估模型的方法跟训练是使用到的Test是一样的，只是我们专门把它提取处理，用于评估模型而已。
同样是要先创建训练器，代码片段如下：
```python
# 通过神经网络模型获取损失函数和额外层
cost, detect_out = vgg_ssd_net.net_conf(mode='eval')
# 检查模型模型路径是否正确
assert os.path.isfile(model_path), 'Invalid model.'
# 通过训练好的模型生成参数
parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_path))
# 创建优化方法
optimizer = paddle.optimizer.Momentum()
# 创建训练器
trainer = paddle.trainer.SGD(cost=cost,
                             parameters=parameters,
                             extra_layers=[detect_out],
                             update_equation=optimizer)
```
然后是去掉训练过程，只留下Test部分，所得的代码片段如下：
```python
# 定义数据层之间的关系
feeding = {'image': 0, 'bbox': 1}
# 生成要训练的数据
reader = paddle.batch(
    data_provider.test(data_args, eval_file_list), batch_size=batch_size)
# 获取测试结果
result = trainer.test(reader=reader, feeding=feeding)
# 打印模型的测试信息
print "TestCost: %f, Detection mAP=%g" % \
      (result.cost, result.metrics['detection_evaluator'])
```
具体调用方法如下，可以看到使用的的数据集还是我们在训练时候使用到的测试数据：
```python
if __name__ == "__main__":
    paddle.init(use_gpu=True, trainer_count=2)
    # 设置数据参数
    data_args = data_provider.Settings(
        data_dir='../data',
        label_file='../data/label_list',
        resize_h=cfg.IMG_HEIGHT,
        resize_w=cfg.IMG_WIDTH,
        mean_value=[104, 117, 124])
    # 开始评估
    eval(eval_file_list='../data/test.txt',
         batch_size=4,
         data_args=data_args,
         model_path='../models/params_pass.tar.gz')
```
评估模型输出的日志如下：
```text
TestCost: 7.185788, Detection mAP=1.07462
```

# 预测数据
-------
## 预测并保存预测结果
获得模型参数之后，就可以使用它来做目标检测了，比如我们要把下面这张图像做目标检测：
![这里写图片描述](http://img.blog.csdn.net/20180223112327884?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
预测的代码片段如下：
```python
# 通过网络模型获取输出层
detect_out = vgg_ssd_net.net_conf(mode='infer')
# 检查模型路径是否正确
assert os.path.isfile(model_path), 'Invalid model.'
# 加载训练好的参数
parameters = paddle.parameters.Parameters.from_tar(gzip.open(model_path))
# 或预测器
inferer = paddle.inference.Inference(
    output_layer=detect_out, parameters=parameters)
# 获取预测数据
reader = data_provider.infer(data_args, eval_file_list)
all_fname_list = [line.strip() for line in open(eval_file_list).readlines()]

# 获取预测原始结果
infer_res = inferer.infer(input=infer_data)
```
获得预测结果之后，我们可以将预测的结果保存的一个文件中，保存这些文件方便之后使用这些数据：
```python
# 获取图像的idx
img_idx = int(det_res[0])
# 获取图像的label
label = int(det_res[1])
# 获取预测的得分
conf_score = det_res[2]
# 获取目标的框
xmin = det_res[3] * img_w[img_idx]
ymin = det_res[4] * img_h[img_idx]
xmax = det_res[5] * img_w[img_idx]
ymax = det_res[6] * img_h[img_idx]
# 将预测结果写入到文件中
fout.write(fname_list[img_idx] + '\t' + str(label) + '\t' + str(
    conf_score) + '\t' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) +
           ' ' + str(ymax))
fout.write('\n')
```
具体调用方法，`eval_file_list`是要预测的数据的路径文件，`save_path`保存预测结果的路径，`resize_h`和`resize_w`指定图像的宽和高，`batch_size`只能设置为1，否则会数据丢失，`model_path`模型的路径，`threshold`是筛选最低得分。
```python
if __name__ == "__main__":
    paddle.init(use_gpu=True, trainer_count=2)
    # 设置数据参数
    data_args = data_provider.Settings(
        data_dir='../images',
        label_file='../data/label_list',
        resize_h=cfg.IMG_HEIGHT,
        resize_w=cfg.IMG_WIDTH,
        mean_value=[104, 117, 124])
    # 开始预测,batch_size只能设置为1，否则会数据丢失
    infer(
        eval_file_list='../images/infer.txt',
        save_path='../images/infer.res',
        data_args=data_args,
        batch_size=1,
        model_path='../models/params_pass.tar.gz',
        threshold=0.3)
```
预测的结果会保存在`images/infer.res`中，每一行对应的是一个目标框，格式为：`图像的路径	分类的标签	目标框的得分	xmin ymin xmax ymax`，每个图像可以有多个类别，所以会有多个框。
```text
infer/00001.jpg	7	0.7000513	287.25091552734375 265.18829345703125 599.12451171875 539.6732330322266
infer/00002.jpg	7	0.53912574	664.7453212738037 240.53946733474731 1305.063714981079 853.0169785022736
infer/00002.jpg	11	0.6429965	551.6539978981018 204.59033846855164 1339.9816703796387 843.807926774025
infer/00003.jpg	12	0.7647844	133.20248904824257 45.33928334712982 413.9954067468643 266.06680154800415
infer/00004.jpg	12	0.66517526	117.327481508255 251.13083073496819 550.8465766906738 665.4091544151306
```
## 显示画出的框
有了以上的预测文件，并不能很直观看到预测的结果，我们可以编写一个程序，让它在原图像上画上预测出来的框，这样就更直接看到结果了。核心代码如下：
```python
# 读取每张图像
for img_path in all_img_paht:
    im = cv2.imread('../images/' + img_path)
    # 为每张图像画上所有的框
    for label_1 in all_labels:
        label_img_path = label_1[0]
        # 判断是否是统一路径
        if img_path == label_img_path:
            xmin, ymin, xmax, ymax = label_1[3].split(' ')
            # 类型转换
            xmin = float(xmin)
            ymin = float(ymin)
            xmax = float(xmax)
            ymax = float(ymax)
            # 画框
            cv2.rectangle(im, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)
    # 保存画好的图像
    names = img_path.strip().split('/')
    name = names[len(names)-1]
    cv2.imwrite('../images/result/%s' % name, im)
```
最后通过在入口调用该方法就可以，代码如下：
```python
if __name__ == '__main__':
    # 预测的图像路径文件
    img_path_list = '../images/infer.txt'
    # 预测结果的文件路径
    result_data_path = '../images/infer.res'
    # 保存画好的图像路径
    save_path = '../images/result'
    show(img_path_list, result_data_path, save_path)
```
画好的图像如下：
![这里写图片描述](http://img.blog.csdn.net/201802231803276?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)











# 项目代码
-----------
GitHub地址:https://github.com/yeyupiaoling/LearnPaddle

# 参考资料
---------
 1. http://paddlepaddle.org/
 2. https://github.com/PaddlePaddle/models/tree/develop/ssd
 3. https://zhuanlan.zhihu.com/p/22045213
 4. https://arxiv.org/abs/1512.02325
