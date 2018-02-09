# coding=utf-8
import os
from collections import defaultdict
import codecs


def get_file_list(image_file_list):
    '''
    生成用于训练和测试数据的文件列表。
    :param image_file_list: 图像文件和列表文件的路径
    :type image_file_list: str
    '''
    dirname = os.path.dirname(image_file_list)
    path_list = []
    with open(image_file_list) as f:
        for line in f:
            # 使用Tab键分离路径和label
            line_split = line.strip().split('\t')
            filename = line_split[0].strip()
            path = os.path.join(dirname, filename)
            label = line_split[1].strip()
            if label:
                path_list.append((path, label))

    return path_list


def build_label_dict(file_list, save_path):
    """
    从训练数据建立标签字典
    :param file_list: 包含标签的训练数据列表
    :type file_list: list
    :params save_path: 保存标签字典的路径
    :type save_path: str
    """
    values = defaultdict(int)
    for path, label in file_list:
        # 加上unicode(label, "utf-8")解决中文编码问题
        for c in unicode(label, "utf-8"):
            if c:
                values[c] += 1

    values['<unk>'] = 0
    # 解决写入文本文件的中文编码问题
    f = codecs.open(save_path,'w','utf-8')
    for v, count in sorted(values.iteritems(), key=lambda x: x[1], reverse=True):
        content = "%s\t%d\n" % (v, count)
        # print content
        f.write(content)


def load_dict(dict_path):
    """
    从字典路径加载标签字典
    :param dict_path: 标签字典的路径
    :type dict_path: str
    """
    return dict((line.strip().split("\t")[0], idx)
                for idx, line in enumerate(open(dict_path, "r").readlines()))


def load_reverse_dict(dict_path):
    """
    从字典路径加载反转的标签字典
    :param dict_path: 标签字典的路径
    :type dict_path: str
    """
    return dict((idx, line.strip().split("\t")[0])
                for idx, line in enumerate(open(dict_path, "r").readlines()))
