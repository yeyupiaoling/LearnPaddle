# coding=utf-8
import os
import json

class CreateDataList:
    def __init__(self):
        pass

    def createDataList(self, data_root_path):
        # # 把生产的数据列表都放在自己的总类别文件夹中
        data_list_path = ''
        # 所有类别的信息
        class_detail = []
        # 获取所有类别
        class_dirs = os.listdir(data_root_path)
        # 类别标签
        class_label = 0
        # 获取总类别的名称
        father_paths = data_root_path.split('/')
        while True:
            if father_paths[father_paths.__len__() - 1] == '':
                del father_paths[father_paths.__len__() - 1]
            else:
                break
        father_path = father_paths[father_paths.__len__() - 1]

        all_class_images = 0
        # 读取每个类别
        for class_dir in class_dirs:
            # 每个类别的信息
            class_detail_list = {}
            test_sum = 0
            trainer_sum = 0
            # 把生产的数据列表都放在自己的总类别文件夹中
            data_list_path = "../data/%s/" % father_path
            # 统计每个类别有多少张图片
            class_sum = 0
            # 获取类别路径
            path = data_root_path + "/" + class_dir
            # 获取所有图片
            img_paths = os.listdir(path)
            for img_path in img_paths:
                # 每张图片的路径
                name_path = path + '/' + img_path
                # 如果不存在这个文件夹,就创建
                isexist = os.path.exists(data_list_path)
                if not isexist:
                    os.makedirs(data_list_path)
                # 每10张图片取一个做测试数据
                if class_sum % 10 == 0:
                    test_sum += 1
                    with open(data_list_path + "test.list", 'a') as f:
                        f.write(name_path + "\t%d" % class_label + "\n")
                else:
                    trainer_sum += 1
                    with open(data_list_path + "trainer.list", 'a') as f:
                        f.write(name_path + "\t%d" % class_label + "\n")
                class_sum += 1
                all_class_images += 1
            class_label += 1
            # 说明的json文件的class_detail数据
            class_detail_list['class_name'] = class_dir
            class_detail_list['class_label'] = class_label
            class_detail_list['class_test_images'] = test_sum
            class_detail_list['class_trainer_images'] = trainer_sum
            class_detail.append(class_detail_list)
        # 获取类别数量
        all_class_sum = class_dirs.__len__()
        # 说明的json文件信息
        readjson = {}
        readjson['all_class_name'] = father_path
        readjson['all_class_sum'] = all_class_sum
        readjson['all_class_images'] = all_class_images
        readjson['class_detail'] = class_detail
        jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
        with open(data_list_path + "readme.json",'w') as f:
            f.write(jsons)


if __name__ == '__main__':
    createDataList = CreateDataList()
    createDataList.createDataList('../images/vegetables')
