# coding=utf-8
import paddle.v2 as paddle

# ***********************定义ResNet卷积神经网络模型***************************************
def resnet_cifar10(datadim,depth=32):
    # 获取输入数据大小
    ipt = paddle.layer.data(name="image",
                              type=paddle.data_type.dense_vector(datadim))


    def conv_bn_layer(input, ch_out, filter_size, stride, padding, active_type=paddle.activation.Relu(),
                      ch_in=None):
        tmp = paddle.layer.img_conv(input=input,
                                    filter_size=filter_size,
                                    num_channels=ch_in,
                                    num_filters=ch_out,
                                    stride=stride,
                                    padding=padding,
                                    act=paddle.activation.Linear(),
                                    bias_attr=False)
        return paddle.layer.batch_norm(input=tmp, act=active_type)

    def shortcut(ipt, n_in, n_out, stride):
        if n_in != n_out:
            return conv_bn_layer(ipt, n_out, 1, stride, 0, paddle.activation.Linear())
        else:
            return ipt

    def basicblock(ipt, ch_out, stride):
        ch_in = ch_out * 2
        tmp = conv_bn_layer(ipt, ch_out, 3, stride, 1)
        tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, paddle.activation.Linear())
        short = shortcut(ipt, ch_in, ch_out, stride)
        return paddle.layer.addto(input=[tmp, short],
                                  act=paddle.activation.Relu())

    def layer_warp(block_func, ipt, features, count, stride):
        tmp = block_func(ipt, features, stride)
        for i in range(1, count):
            tmp = block_func(tmp, features, 1)
        return tmp

    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    nStages = {16, 64, 128}
    conv1 = conv_bn_layer(ipt, ch_in=3, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 64, n, 2)
    pool = paddle.layer.img_pool(
        input=res3, pool_size=8, stride=1, pool_type=paddle.pooling.Avg())

    # 通过神经网络模型再使用Softmax获得分类器(全连接)
    out = paddle.layer.fc(input=pool,
                          size=10,
                          act=paddle.activation.Softmax())
    return out