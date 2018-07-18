# coding=utf-8
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
    一个卷积和BN层
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

    # tmp = paddle.layer.img_pool(
    #     input=tmp, pool_size=2, stride=1, pool_type=paddle.pooling.Avg())
    out = paddle.layer.fc(
        input=tmp, size=class_num, act=paddle.activation.Softmax())

    return out


if __name__ == '__main__':
    img_size = 3 * 32 * 32
    data_dim = 10
    out = mobile_net(img_size, data_dim, 1.0)
