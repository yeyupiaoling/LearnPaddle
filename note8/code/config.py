# coding=utf-8
__all__ = ["TrainerConfig", "ModelConfig"]


class TrainerConfig(object):
    # 是否使用GPU
    use_gpu = True

    # 使用线程数量
    trainer_count = 2

    # batch的大小
    batch_size = 100

    # 训练的轮数
    num_passes = 1000

    # 创建参数的momentum.
    momentum = 0

    # 图像的大小
    image_shape = (173, 46)

    # 数据读取器的缓冲区大小。
    # 缓冲区大小样本的数量将在训练中混洗。
    buf_size = 1000

    # 该参数用于控制日志记录周期。
    # 每个log_period都会打印一个训练日志。
    log_period = 50


class ModelConfig(object):
    # 卷积组的过滤器数量。
    filter_num = 8

    # 在图像卷积组中使用批量标准化。
    with_bn = True

    # 块扩展层的通道数。
    num_channels = 128

    # 块扩展图层中的参数stride_x。
    stride_x = 1

    # 块扩展图层中的参数stride_y。
    stride_y = 1

    # 块扩展图层中的参数block_x。
    block_x = 1

    # 块扩展图层中的参数block_y。
    block_y = 11

    # 隐藏的大小
    hidden_size = num_channels

    # 是否使用BN层
    norm_by_times = True

    # 图像卷积组图层中的滤镜数目列表。
    filter_num_list = [16, 32, 64, 128]

    # 图像卷积组层中的参数conv_padding。
    conv_padding = 1

    # T图像卷积组层中的参数conv_filter_size。
    conv_filter_size = 3

    # 图像卷积组层中的参数pool_size。
    pool_size = 2

    # 图像卷积组层中的参数pool_stride。
    pool_stride = 2
