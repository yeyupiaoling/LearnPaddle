# coding=utf-8
import cv2


def show(img_path_list, result_data_path, save_path):
    # 所有图像路径
    all_img_paht = []
    # 所有图像标注信息
    all_labels = []

    # 读取所有图像路径
    with open(img_path_list, 'r') as f:
        for img_path_temp in f:
            all_img_paht.append(img_path_temp.strip())

    # 读取标注信息
    with open(result_data_path, 'r') as f:
        for line in f:
            labels = []
            path, label, score, xy = line.strip().split('\t')
            labels.append(path)
            labels.append(label)
            labels.append(score)
            labels.append(xy)
            all_labels.append(labels)

    # 读取每张图像
    for img_path in all_img_paht:
        im = cv2.imread('../images/' + img_path)
        # 为每张图像画上所有的框
        for label_1 in all_labels:
            label_img_path = label_1[0]
            # 判断是否是统一路径
            if img_path == label_img_path:
                xmin, ymin, xmax, ymax = label_1[3].split(' ')
                # 类型转换
                xmin = float(xmin)
                ymin = float(ymin)
                xmax = float(xmax)
                ymax = float(ymax)
                # 画框
                cv2.rectangle(im, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)
        # 保存画好的图像
        names = img_path.strip().split('/')
        name = names[len(names)-1]
        cv2.imwrite('../images/result/%s' % name, im)


if __name__ == '__main__':
    # 预测的图像路径文件
    img_path_list = '../images/infer.txt'
    # 预测结果的文件路径
    result_data_path = '../images/infer.res'
    # 保存画好的图像路径
    save_path = '../images/result'
    show(img_path_list, result_data_path, save_path)
