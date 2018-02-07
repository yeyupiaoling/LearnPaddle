# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import groupby
import numpy as np


def ctc_greedy_decoder(probs_seq, vocabulary):
    """CTC贪婪（最佳路径）解码器。
    由最可能的令牌组成的路径被进一步后处理
    删除连续的重复和所有的空白。
    :param probs_seq: 每个词汇表上概率的二维列表字符。
                      每个元素都是浮点概率列表为一个字符。
    :type probs_seq: list
    :param vocabulary: 词汇表
    :type vocabulary: list
    :return: 解码结果字符串
    :rtype: baseline
    """
    # 尺寸验证
    for probs in probs_seq:
        if not len(probs) == len(vocabulary) + 1:
            raise ValueError("probs_seq dimension mismatchedd with vocabulary")
    # argmax以获得每个时间步长的最佳指标
    max_index_list = list(np.array(probs_seq).argmax(axis=1))
    # 删除连续的重复索引
    index_list = [index_group[0] for index_group in groupby(max_index_list)]
    # 删除空白索引
    blank_index = len(vocabulary)
    index_list = [index for index in index_list if index != blank_index]
    # 将索引列表转换为字符串
    return ''.join([vocabulary[index] for index in index_list])
