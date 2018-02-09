# coding=utf-8
import uuid
import os
import cv2
import numpy as np

is_cut = False
class CutPlateNumber:
    def __init__(self):
        self.is_cut = False

    def preprocess(self,gray, iterations):
        # 高斯平滑
        gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
        # 中值滤波
        median = cv2.medianBlur(gaussian, 5)
        # Sobel算子，X方向求梯度
        sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize=3)
        # 二值化
        ret, binary = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)
        # 膨胀和腐蚀操作的核函数
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
        # 膨胀一次，让轮廓突出
        dilation = cv2.dilate(binary, element2, iterations=1)
        # 腐蚀一次，去掉细节
        erosion = cv2.erode(dilation, element1, iterations=1)
        # 再次膨胀，让轮廓明显一些
        dilation2 = cv2.dilate(erosion, element2, iterations=iterations)
        return dilation2


    def findPlateNumberRegion(self,img):
        region = []
        # 查找轮廓
        binary, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 筛选面积小的
        for i in range(len(contours)):
            cnt = contours[i]
            # 计算该轮廓的面积
            area = cv2.contourArea(cnt)
            # 面积小的都筛选掉
            if (area < 2000):
                continue

            # 找到最小的矩形，该矩形可能有方向
            rect = cv2.minAreaRect(cnt)

            # box是四个点的坐标
            box = cv2.cv2.boxPoints(rect)
            box = np.int0(box)

            # 计算高和宽
            height = abs(box[0][1] - box[2][1])
            width = abs(box[0][0] - box[2][0])
            # 车牌正常情况下长高比在2.7-5之间
            ratio = float(width) / float(height)
            if (ratio > 5 or ratio < 2):
                continue
            region.append(box)
        return region


    def detect(self,img, iterations, is_infer=False):
        # 转化成灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 形态学变换的预处理
        dilation = self.preprocess(gray, iterations)
        # 查找车牌区域
        region = self.findPlateNumberRegion(dilation)
        if len(region) > 0:
            # 如果使用6次迭代膨胀获取车牌成功，就裁剪保存
            box = region[0]
            ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
            xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
            ys_sorted_index = np.argsort(ys)
            xs_sorted_index = np.argsort(xs)

            x1 = box[xs_sorted_index[0], 0]
            x2 = box[xs_sorted_index[3], 0]

            y1 = box[ys_sorted_index[0], 1]
            y2 = box[ys_sorted_index[3], 1]

            img_plate = img[y1:y2, x1:x2]
            if is_infer:
                # 如果是用于预测的图像，就给定文件名
                cv2.imwrite('../images/infer.jpg', img_plate)
            else:
                # 如果是训练的图像，就裁剪到数据临时存放文件夹等待下一步处理
                cv2.imwrite('../data/data_temp/%s.jpg' % self.img_name, img_plate)
        else:
            if self.is_cut:
                pass
            else:
                self.is_cut = True
                # 如果使用6次迭代膨胀获取车牌不成功，就使用3次迭代膨胀
                self.detect(img, 3)


    def strat_crop(self,imagePath, is_infer=False,name=None):
        self.is_cut = False
        if not is_infer:
            self.img_name = name.split('.')[0]
        # 开始裁剪
        img = cv2.imread(imagePath)
        # 默认使用6次迭代膨胀
        self.detect(img=img, iterations=6, is_infer=is_infer)


if __name__ == '__main__':
    cutPlateNumber = CutPlateNumber()
    img_path = '../images/src_temp/'
    # 获取原图像的所有路径
    imgs = os.listdir(img_path)
    for img in imgs:
        cutPlateNumber.strat_crop(img_path + img, False,img)
