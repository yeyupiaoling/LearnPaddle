# coding=utf-8
from paddle.utils.merge_model import merge_v2_model

# 导入神经网络
from mobilenet import mobile_net
from vgg import vgg_bn_drop

if __name__ == "__main__":
    # 图像的大小
    img_size = 3 * 32 * 32
    # 总分类数
    class_dim = 10
    net = mobile_net(img_size, class_dim)
    param_file = '../model/mobile_net.tar.gz'
    output_file = '../model/mobile_net.paddle'
    merge_v2_model(net, param_file, output_file)