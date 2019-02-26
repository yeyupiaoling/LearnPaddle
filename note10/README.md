# 目录
@[toc]
*本篇文章基于 PaddlePaddle 0.11.0、Python 2.7
# 前言
------
在阅读这一篇文章之前，要先阅读上一篇文章[使用VOC数据集的实现目标检测](http://blog.csdn.net/qq_33200967/article/details/79126780)，因为大部分的程序都是使用上一篇文章所使用到的代码和数据集的格式。在这篇文章中介绍如何使用自定义的图像数据集来做目标检测。

# 数据集介绍
------
我们本次使用的到的数据集是自然场景下的车牌，不知读者是否还记得在[车牌端到端的识别](http://blog.csdn.net/qq_33200967/article/details/79095335)这篇文章中，我们使用到的车牌是如何裁剪的，我们是使用OpenCV经过多重的的图像处理才达到车牌定位的，而且定位的效果比较差。在这篇文章中我们尝试使用神经网络来定位车牌位置。
## 下载车牌
我们先从网络上下载车牌数据，来提供给我们进行训练，核心代码片段如下：
```python
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
```
## 重命名图像
下载好的图像会存放在`data/plate_number/images/`这个路径下，其中下载的一下数据可能不是车牌的图像，我们需要把它删除掉。然后为了让我们的数据集更符合VOC数据集，我们要对图像重命名，命名程序如下：
```python
# coding=utf-8
import os

def rename(images_dir):
    # 获取所有图像
    images = os.listdir(images_dir)
    i = 1
    for image in images:
        src_name = images_dir + image
        # 以六位数字命名，符合VOC数据集格式
        name = '%06d.jpg' % i
        dst_name = images_dir + name
        os.rename(src_name,dst_name)
        i += 1
    print '重命名完成'

if __name__ == '__main__':
    # 要重命名的文件所在的路径
    images_dir = '../data/plate_number/images/'
    rename(images_dir)
```

# 标注数据集
------
图像数据我们有了，也命名完成了，但是我们还缺少一个非常重要的标注信息，在VOC数据集中，每张图像的标注信息是存放在`XML`文件中的，并且命名跟图像是一样的（后缀名除外），所以我们要制作标注信息文件。当然，那么复杂的工作，肯定要一个程序来协助完成，我们使用的是LabelImg。下面就介绍使用LabelImg标注我们的图像。
## 安装LabelImg
在Ubuntu 16.04上安装LabelImg，操作非常简单，通过几行命名就可以完成安装了
```python
# 获取管理员权限
sudo su
# 安装依赖库
apt-get install pyqt4-dev-tools
pip install lxml
# 安装labelImg
pip install labelImg
# 退出管理员权限
exit
# 运行labelImg
labelImg
```
## 使用LabelImg
运行程序之后，显示的界面如下：
![这里写图片描述](http://img.blog.csdn.net/20180223231741554?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
然后我们点击`Open Dir`打开图像所在的文件夹`data/plate_number/images/`，程序显示如下：
![这里写图片描述](http://img.blog.csdn.net/20180223232046827?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
到这里我们不要急于标注图像，我们要先设置保存标注文件存放的位置，点击`Change Save Dir`选择保存标注文件存放的位置`data/plate_number/annotation/`，然后在点击`Create RectBox`标注车牌的位置，并打上标签`plate_number`。最后别忘了保存标注文件，点击`Save`，就会以图像的名称命名标注文件并保存。然后就可以点击`Next Image`，标注下一个图像了。
![这里写图片描述](http://img.blog.csdn.net/20180223232824801?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
标注的文件信息如下，符合VOC数据集格式要求：
```xml
<annotation>
	<folder>images</folder>
	<filename>000001.jpg</filename>
	<path>/home/yeyupiaoling/data/plate_number/images/000001.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>750</width>
		<height>562</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>plate_number</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>225</xmin>
			<ymin>298</ymin>
			<xmax>560</xmax>
			<ymax>405</ymax>
		</bndbox>
	</object>
</annotation>
```
## 生成图像列表
有了图像和图像的标注文件，我们还需要两个图像列表，训练图像列表`trainval.txt`和测试图像列表`test.txt`，应为我们这次的数据集的文件夹的结构跟之前的不一样，所以我们生成图像列表的程序也不一样了。

首先要读取所有的图像和标注文件，并将他们一一对应：
```python
for images in all_images:
    trainval = []
    test = []
    if data_num % 10 == 0:
        # 没10张图像取一个做测试集
        name = images.split('.')[0]
        annotation = os.path.join(annotation_path, name + '.xml')
        # 如果该图像的标注文件不存在，就不添加到图像列表中
        if not os.path.exists(annotation):
            continue
        test.append(os.path.join(images_path, images))
        test.append(annotation)
        # 添加到总的测试数据中
        test_list.append(test)
    else:
        # 其他的的图像做训练数据集
        name = images.split('.')[0]
        annotation = os.path.join(annotation_path, name + '.xml')
        # 如果该图像的标注文件不存在，就不添加到图像列表中
        if not os.path.exists(annotation):
            continue
        trainval.append(os.path.join(images_path, images))
        trainval.append(annotation)
        # 添加到总的训练数据中
        trainval_list.append(trainval)
    data_num += 1
```
然后把他们写入到图像列表的文件中，为了使得训练数据是随机性的，可以对训练的数据集打乱一下。
```python
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
```

# 训练模型
------
有了图像数据和标注文件，也有了图像列表，我们就可以开始训练模型了，在训练之前，我们还有修改一下配置文件`pascal_voc_conf.py`，把类别改成2，因为我们只有车牌和背景，所以只有两个类别。
```python
# 图像的分类种数
__C.CLASS_NUM = 2
```
## 预训练模型处理
如果直接训练是会出现浮点异常的，我们需要一个预训练的模型来初始化训练模型，我们这次使用的初始化模型同样是[官方预训练的模型](http://paddlemodels.bj.bcebos.com/v2/vgg_model.tar.gz)，但是不能直接使用，还有删除一些没用的文件，因为我们的类别数量更之前的不一样，官方预训练的模型的部分文件如下：
![这里写图片描述](http://img.blog.csdn.net/20180227200231549?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
**我们把文件名中包含mbox的文件都删掉**就可以用来做我们的初始化模型了。

## 开始训练
最后开始训练使用的是2个GPU，因为使用到的神经网络仅支持CUDA GPU环境，所以只能使用GPU来进行训练。`train_file_list`是训练图像列表文件路径，`dev_file_list`是测试图像列表文件路径，`data_args`是数据集的设置信息，`init_model_path`使用预训练的模型初始化训练参数的模型。
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
训练的过程中输入以下的日志信息：
```text

Pass 0, Batch 0, TrainCost 16.567970, Detection mAP=0.014627
......
Test with Pass 0, TestCost: 8.723172, Detection mAP=0.00609719

Pass 1, Batch 0, TrainCost 7.185760, Detection mAP=0.239866
......
Test with Pass 1, TestCost: 6.301503, Detection mAP=60.357

Pass 2, Batch 0, TrainCost 6.052617, Detection mAP=32.094097
......
Test with Pass 2, TestCost: 5.375503, Detection mAP=48.9882
```

# 评估模型
-------
我们同样可以评估我们训练好的模型，了解模型收敛的情况。`eval_file_list`是要用来评估模型的数据集，我们使用的是训练是使用的测试数据集，`batch_size`是batch的大小，`data_args`是数据集的设置信息，`model_path`要评估模型的路径。
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
评估输出的结果如下：
```text
TestCost: 1.813083, Detection mAP=90.5595
```

# 预测数据
-----
## 获取预测数据
首先我们先要找几张图像来作为预测的数据，我们在网上下载几张之前没有使用到的图像，把它们存放在`images/infer/`目录下，并在`images/infer.txt`文件中写入它们的路径，如下：
```text
infer/000001.jpg
infer/000002.jpg
infer/000003.jpg
infer/000004.jpg
infer/000005.jpg
infer/000006.jpg
```

## 获取预测结果
然后通过调用预测函数就可以获取到预测结果，并且把预测结果存放在`images/infer.res`。`eval_file_list`是要用来预测的数据集，就是上面获得的图像路径文件；`save_path`是保存预测结果的路径，预测的结果会存放在这个文件中；`batch_size`是batch的大小；`data_args`是数据集的设置信息；`model_path`要使用模型的路径；`threshold`筛选的最低得分。
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
预测的结果保存的文件格式是：`图像的路径 分类的标签 目标框的得分 xmin ymin xmax ymax`，具体如下：
```text
infer/000001.jpg        0       0.9999114       357.44736313819885 521.2164137363434 750.5996704101562 648.5584638118744
infer/000002.jpg        0       0.9970805       102.86840772628784 94.18213963508606 291.60091638565063 155.58562874794006
infer/000003.jpg        0       0.7187747       222.9731798171997 168.14028024673462 286.6227865219116 194.68939304351807
infer/000004.jpg        0       0.9988129       197.94835299253464 177.8149015903473 285.8962297439575 218.93768119812012
infer/000005.jpg        0       0.9149439       98.09065014123917 288.86341631412506 237.42297291755676 331.9027876853943
infer/000005.jpg        0       0.9114895       544.3056106567383 235.35346180200577 674.311637878418 283.9097347855568
infer/000006.jpg        0       0.92390853      265.203565120697 277.6864364147186 412.7485656738281 344.3739159107208
```

## 显示预测结果
预测的结果是一串数据，对于我们来说，并不是很直观，我们同样要编写一个程序，让它把每张图像的车牌框出来。程序的核心代码如下：
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
最后通过在入口调用该方法就可以，画好的框的图像都会保存到`images/result/`目录下，代码如下：
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
预测前的图像：
![这里写图片描述](http://img.blog.csdn.net/20180227202510123?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

预测后的图像：
![这里写图片描述](http://img.blog.csdn.net/20180227202518958?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)




# 项目代码
-------
GitHub地址:https://github.com/yeyupiaoling/LearnPaddle


# 参考资料
---------
 1. http://paddlepaddle.org/
 2. https://github.com/tzutalin/labelImg
