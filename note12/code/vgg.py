# coding=utf-8
import paddle.fluid as fluid


def vgg16_bn_drop(input):
    # 定义卷积块
    def conv_block(input, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=input,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')
    # 定义一个VGG16的卷积组
    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])
    # 定义第一个drop层
    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    # 定义第一层全连接层
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    # 定义BN层
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    # 定义第二层全连接层
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    # 定义第二层全连接层
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    return fc2,conv1
