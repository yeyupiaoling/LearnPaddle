# coding=utf-8
# 导入VisualDL的包
from visualdl import LogWriter

# 创建一个LogWriter，第一个参数是指定存放数据的路径，
# 第二个参数是指定多少次写操作执行一次内存到磁盘的数据持久化
logw = LogWriter("../data/random_log", sync_cycle=10000)

# 创建训练和测试的scalar图，
# mode是标注线条的名称，
# scalar标注的是指定这个组件的tag
with logw.mode('train') as logger:
    scalar0 = logger.scalar("scratch/scalar")

with logw.mode('test') as logger:
    scalar1 = logger.scalar("scratch/scalar")

# 读取数据
for step in range(1000):
    scalar0.add_record(step, step * 1. / 1000)
    scalar1.add_record(step, 1. - step * 1. / 1000)