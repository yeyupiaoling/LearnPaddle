# 笔记4
## 目录介绍
1. `code`为存放代码的文件夹
2. `images`为存放裁剪前的验证码,裁剪后的验证码和下载的验证码
3. `model`为存放训练后的模型,因为没有上传,可能不存在
4. `data`为存放每个类别的图像数据列表和数据说明

## 代码介绍
1. `PaddleUtil.py`为使用PaddlePaddle训练和预测的代码
2. `CreateDataList.py`创建用于训练与测试的图像列表
3. `DownloadYanZhengMa.py`下载验证码程序
4. `MyReader.py`读取图像列表
5. `CorpYanZhengMa.py`把验证码裁剪成四个图片
6. `image.py`图像处理的,其就是paddle.v2.image的程序

## 笔记
程序存在bug,还没有完成