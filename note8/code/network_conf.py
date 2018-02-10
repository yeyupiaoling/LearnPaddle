# coding=utf-8
from paddle import v2 as paddle
from paddle.v2 import layer
from paddle.v2 import evaluator
from paddle.v2.activation import Relu, Linear
from paddle.v2.networks import img_conv_group, simple_gru
from config import ModelConfig as conf


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

    def __build_nn__(self):
        '''
        建立网络拓扑
        '''
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

        # 使用扭曲CTC来计算CTC任务的成本。
        if not self.is_infer:
            self.cost = layer.warp_ctc(
                input=self.output,
                label=self.label,
                size=self.num_classes + 1,
                norm_by_times=conf.norm_by_times,
                blank=self.num_classes)

            self.eval = evaluator.ctc_error(input=self.output, label=self.label)

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
