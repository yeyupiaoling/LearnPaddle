# coding:utf-8
import paddle.v2 as paddle

# 卷积神经网络LeNet-5,获取分类器
def convolutional_neural_network(datadim, type_size):
    image = paddle.layer.data(name="image",
                              type=paddle.data_type.dense_vector(datadim))

    # 第一个卷积--池化层
    conv_pool_1 = paddle.networks.simple_img_conv_pool(input=image,
                                                       filter_size=5,
                                                       num_filters=20,
                                                       num_channel=1,
                                                       pool_size=2,
                                                       pool_stride=2,
                                                       act=paddle.activation.Relu())
    # 第二个卷积--池化层
    conv_pool_2 = paddle.networks.simple_img_conv_pool(input=conv_pool_1,
                                                       filter_size=5,
                                                       num_filters=50,
                                                       num_channel=20,
                                                       pool_size=2,
                                                       pool_stride=2,
                                                       act=paddle.activation.Relu())
    # 以softmax为激活函数的全连接输出层
    out = paddle.layer.fc(input=conv_pool_2,
                          size=type_size,
                          act=paddle.activation.Softmax())
    return out
