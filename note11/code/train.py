# coding=utf-8
import os
import paddle.fluid as fluid
import paddle.v2 as paddle
from vgg import vgg16_bn_drop


def train(use_cuda, learning_rate, num_passes,BATCH_SIZE = 128, model_save_dir='../models'):
    # 定义图像的类别数量
    class_dim = 10
    # 定义图像的通道数和大小
    image_shape = [3, 32, 32]
    # 定义输入数据大小，指定图像的形状，数据类型是浮点型
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    # 定义标签，类型是整型
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # 获取神经网络的分类器
    predict = vgg16_bn_drop(image,class_dim)
    # 获取损失函数
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    # 定义平均损失函数
    avg_cost = fluid.layers.mean(x=cost)

    # 每个batch计算的时候能取到当前batch里面样本的个数，从而来求平均的准确率
    batch_size = fluid.layers.create_tensor(dtype='int64')
    batch_acc = fluid.layers.accuracy(input=predict, label=label, total=batch_size)

    # 测试程序
    inference_program = fluid.default_main_program().clone(for_test=True)

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
    # 获取测试数据
    test_reader = paddle.batch(
        paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE)

    # 指定数据和label的对于关系
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    accuracy = fluid.average.WeightedAverage()
    test_accuracy = fluid.average.WeightedAverage()
    # 开始训练，使用循环的方式来指定训多少个Pass
    for pass_id in range(num_passes):
        # 从训练数据中按照一个个batch来读取数据
        accuracy.reset()
        for batch_id, data in enumerate(train_reader()):
            loss, acc, weight = exe.run(fluid.default_main_program(),
                                feed=feeder.feed(data),
                                fetch_list=[avg_cost, batch_acc, batch_size])
            accuracy.add(value=acc, weight=weight)
            print("Pass {0}, batch {1}, loss {2}, acc {3}".format(
                pass_id, batch_id, loss[0], acc[0]))

        # 测试模型
        test_accuracy.reset()
        for data in test_reader():
            loss, acc, weight = exe.run(inference_program,
                                feed=feeder.feed(data),
                                fetch_list=[avg_cost, batch_acc, batch_size])
            test_accuracy.add(value=acc, weight=weight)

        # 输出相关日志
        pass_acc = accuracy.eval()
        test_pass_acc = test_accuracy.eval()
        print("End pass {0}, train_acc {1}, test_acc {2}".format(
            pass_id, pass_acc, test_pass_acc))

        # 每一个Pass就保存一次模型
        # 指定保存模型的路径
        model_path = os.path.join(model_save_dir, str(pass_id))
        # 如果保存路径不存在就创建
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        print 'save models to %s' % (model_path)
        # 保存预测的模型
        fluid.io.save_inference_model(model_path, ['image'], [predict], exe)


if __name__ == '__main__':
    # 开始训练
    train(use_cuda=False, learning_rate=0.005, num_passes=300)