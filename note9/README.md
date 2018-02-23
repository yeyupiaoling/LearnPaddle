## data目录介绍
1. `VOCdevkit`是VOC数据集，包括2007和2012全部数据
2. `label_list`是所有类别的文件
3. `prepare_voc_data.py`是生成图像列表的程序
4. `test.txt`是测试图像列表
5. `trainval.txt`是训练的图像列表

## models目录介绍
1. 该目录是存放训练好的模型
2. 可以下载[官方提供的模型](http://paddlepaddle.bj.bcebos.com/model_zoo/detection/ssd_model/vgg_model.tar.gz)作为训练的初始化参数

## images目录介绍
1. `infer`是存放要预测的图像
2. `result`是存放预测后画好了框的图像
3. `infer.res`是预测结果数据的文件
4. `infer.txt`是要预测的图像路径

## 代码介绍
1. `data_provider.py`是预处理数据集的程序
2. `eval.py`是评估模型的程序
3. `image_util.py`是图像工具类的程序
4. `infer.py`是预测的程序
5. `pascal_voc_conf.py`是各种配置信息的程序
6. `show_infer_image.py`是在预测图像上画出预测的框程序
7. `train.py`是训练的程序
8. `vgg_ssd_net.py`是VGG神经网络和SSD神经网络

## 笔记
笔记文章地址:http://blog.csdn.net/qq_33200967/article/details/79126780