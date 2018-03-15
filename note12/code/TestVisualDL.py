# coding=utf-8
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.v2 as paddle
from paddle.fluid.initializer import NormalInitializer
from paddle.fluid.param_attr import ParamAttr
from visualdl import LogWriter
from vgg import vgg16_bn_drop

# 创建VisualDL，并指定当前该项目的VisualDL的路径
logdir = "../data/tmp"
logwriter = LogWriter(logdir, sync_cycle=10)

# 创建loss的趋势图
with logwriter.mode("train") as writer:
    loss_scalar = writer.scalar("loss")

# 创建acc的趋势图
with logwriter.mode("train") as writer:
    acc_scalar = writer.scalar("acc")

# 定义没多少次重新输出一遍
num_samples = 4
# 创建卷积层和输出图像的图形化展示
with logwriter.mode("train") as writer:
    conv_image = writer.image("conv_image", num_samples, 1)
    input_image = writer.image("input_image", num_samples, 1)

# 创建可视化的训练模型结构
with logwriter.mode("train") as writer:
    param1_histgram = writer.histogram("param1", 100)


def train(use_cuda, learning_rate, num_passes, BATCH_SIZE=128):
    # 定义图像的类别数量
    class_dim = 10
    # 定义图像的通道数和大小
    image_shape = [3, 32, 32]
    # 定义输入数据大小，指定图像的形状，数据类型是浮点型
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    # 定义标签，类型是整型
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # 获取神经网络
    net, conv1 = vgg16_bn_drop(image)
    # 获取全连接输出，获得分类器
    predict = fluid.layers.fc(
        input=net,
        size=class_dim,
        act='softmax',
        param_attr=ParamAttr(name="param1", initializer=NormalInitializer()))

    # 获取损失函数
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    # 定义平均损失函数
    avg_cost = fluid.layers.mean(x=cost)

    # 每个batch计算的时候能取到当前batch里面样本的个数，从而来求平均的准确率
    batch_size = fluid.layers.create_tensor(dtype='int64')
    print batch_size
    batch_acc = fluid.layers.accuracy(input=predict, label=label, total=batch_size)

    # 定义优化方法
    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(5 * 1e-5))

    opts = optimizer.minimize(avg_cost)

    # 是否使用GPU
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    # 创建调试器
    exe = fluid.Executor(place)
    # 初始化调试器
    exe.run(fluid.default_startup_program())

    # 获取训练数据
    train_reader = paddle.batch(
        paddle.dataset.cifar.train10(), batch_size=BATCH_SIZE)

    # 指定数据和label的对于关系
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    step = 0
    sample_num = 0
    start_up_program = framework.default_startup_program()
    param1_var = start_up_program.global_block().var("param1")

    accuracy = fluid.average.WeightedAverage()
    # 开始训练，使用循环的方式来指定训多少个Pass
    for pass_id in range(num_passes):
        # 从训练数据中按照一个个batch来读取数据
        accuracy.reset()
        for batch_id, data in enumerate(train_reader()):
            loss, conv1_out, param1, acc, weight = exe.run(fluid.default_main_program(),
                                                           feed=feeder.feed(data),
                                                           fetch_list=[avg_cost, conv1, param1_var, batch_acc,
                                                                       batch_size])
            accuracy.add(value=acc, weight=weight)
            pass_acc = accuracy.eval()

            # 重新启动图形化展示组件
            if sample_num == 0:
                input_image.start_sampling()
                conv_image.start_sampling()
            # 获取taken
            idx1 = input_image.is_sample_taken()
            idx2 = conv_image.is_sample_taken()
            # 保证它们的taken是一样的
            assert idx1 == idx2
            idx = idx1
            if idx != -1:
                # 加载输入图像的数据数据
                image_data = data[0][0]
                input_image_data = np.transpose(
                    image_data.reshape(image_shape), axes=[1, 2, 0])
                input_image.set_sample(idx, input_image_data.shape,
                                       input_image_data.flatten())
                # 加载卷积数据
                conv_image_data = conv1_out[0][0]
                conv_image.set_sample(idx, conv_image_data.shape,
                                      conv_image_data.flatten())

                # 完成输出一次
                sample_num += 1
                if sample_num % num_samples == 0:
                    input_image.finish_sampling()
                    conv_image.finish_sampling()
                    sample_num = 0

            # 加载趋势图的数据
            loss_scalar.add_record(step, loss)
            acc_scalar.add_record(step, acc)
            # 添加模型结构数据
            param1_histgram.add_record(step, param1.flatten())
            # 输出训练日志
            print("loss:" + str(loss) + " acc:" + str(acc) + " pass_acc:" + str(pass_acc))
            step += 1


if __name__ == '__main__':
    # 开始训练
    train(use_cuda=False, learning_rate=0.005, num_passes=300)
