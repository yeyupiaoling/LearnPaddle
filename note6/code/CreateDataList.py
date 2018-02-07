# coding=utf-8
import os

class CreateDataList:
    def __init__(self):
        pass

    def createDataList(self, data_path, isTrain):
        # 判断生成的列表是训练图像列表还是测试图像列表
        if isTrain:
            list_name = 'trainer.list'
        else:
            list_name = 'test.list'
        list_path = os.path.join(data_path, list_name)
        # 判断该列表是否存在，如果存在就删除，避免在生成图像列表时把该路径也写进去了
        if os.path.exists(list_path):
            os.remove(list_path)
        # 读取所有的图像路径，此时图像列表不存在，就不用担心写入非图像文件路径了
        imgs = os.listdir(data_path)
        for img in imgs:
            name = img.split('.')[0]
            with open(list_path, 'a') as f:
                # 写入图像路径和label，用Tab隔开
                f.write(img + '\t' + name + '\n')

if __name__ == '__main__':
    createDataList = CreateDataList()
    # 生成训练图像列表
    createDataList.createDataList('../data/train_data/', True)
    # 生成测试图像列表
    createDataList.createDataList('../data/test_data/', False)
