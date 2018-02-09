# coding=utf-8
import paddle.v2 as paddle


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
        self.image = paddle.layer.data(
            name='image',
            type=paddle.data_type.dense_vector(self.image_vector_size),
            # shape是(宽度,高度)
            height=self.shape[1],
            width=self.shape[0])

        # 将标签输入为ID列表
        if not self.is_infer:
            self.label = paddle.layer.data(
                name='label',
                type=paddle.data_type.integer_value_sequence(self.num_classes))

    def __build_nn__(self):
        '''
        建立网络拓扑
        '''
        # 通过CNN获取图像特征
        def conv_block(ipt, num_filter, groups, num_channels=None):
            return paddle.networks.img_conv_group(
                input=ipt,
                num_channels=num_channels,
                conv_padding=1,
                conv_num_filter=[num_filter] * groups,
                conv_filter_size=3,
                conv_act=paddle.activation.Relu(),
                conv_with_batchnorm=True,
                pool_size=2,
                pool_stride=2, )

        # 因为是灰度图所以最后一个参数是1
        conv1 = conv_block(self.image, 16, 2, 1)
        conv2 = conv_block(conv1, 32, 2)
        conv3 = conv_block(conv2, 64, 2)
        conv_features = conv_block(conv3, 128, 2)

        # 将CNN的输出展开成一系列特征向量。
        sliced_feature = paddle.layer.block_expand(
            input=conv_features,
            num_channels=128,
            stride_x=1,
            stride_y=1,
            block_x=1,
            block_y=11)

        # 使用RNN向前和向后捕获序列信息。
        gru_forward = paddle.networks.simple_gru(
            input=sliced_feature, size=128, act=paddle.activation.Relu())
        gru_backward = paddle.networks.simple_gru(
            input=sliced_feature,
            size=128,
            act=paddle.activation.Relu(),
            reverse=True)

        # 将RNN的输出映射到字符分布。
        self.output = paddle.layer.fc(input=[gru_forward, gru_backward],
                                      size=self.num_classes + 1,
                                      act=paddle.activation.Linear())

        self.log_probs = paddle.layer.mixed(
            input=paddle.layer.identity_projection(input=self.output),
            act=paddle.activation.Softmax())

        # 使用扭曲CTC来计算CTC任务的成本。
        if not self.is_infer:
            # 定义cost
            self.cost = paddle.layer.warp_ctc(
                input=self.output,
                label=self.label,
                size=self.num_classes + 1,
                norm_by_times=True,
                blank=self.num_classes)
            # 定义额外层
            self.eval = paddle.evaluator.ctc_error(input=self.output, label=self.label)