# -*- coding:utf-8 -*-
import re
import uuid
import requests
import os


class DownloadImages:
    def __init__(self,download_max,key_word):
        self.download_sum = 0
        self.download_max = download_max
        self.key_word = key_word
        self.save_path = '../images/download/' + key_word

    def start_download(self):
        self.download_sum = 0
        gsm = 80
        str_gsm = str(gsm)
        pn = 0
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        while self.download_sum < self.download_max:
            str_pn = str(self.download_sum)
            url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&' \
                  'word=' + self.key_word + '&pn=' + str_pn + '&gsm=' + str_gsm + '&ct=&ic=0&lm=-1&width=0&height=0'
            print url
            result = requests.get(url)
            self.downloadImages(result.text)
        print '下载完成'

    def downloadImages(self,html):
        img_urls = re.findall('"objURL":"(.*?)",', html, re.S)
        print '找到关键词:' + self.key_word + '的图片，现在开始下载图片...'
        for img_url in img_urls:
            print '正在下载第' + str(self.download_sum + 1) + '张图片，图片地址:' + str(img_url)
            try:
                pic = requests.get(img_url, timeout=50)
                pic_name = self.save_path + '/' + str(uuid.uuid1()) + '.jpg'
                with open(pic_name, 'wb') as f:
                    f.write(pic.content)
                self.download_sum += 1
                if self.download_sum >= self.download_max:
                    break
            except  Exception, e:
                print '【错误】当前图片无法下载，%s' % e
                continue


if __name__ == '__main__':
    key_word_max = input('请输入你要下载几个类别:')
    key_words = []
    for sum in range(key_word_max):
        key_words.append(raw_input('请输入第%s个关键字:' % str(sum+1)))
    max_sum = input('请输入每个类别下载的数量:')
    for key_word in key_words:
        downloadImages = DownloadImages(max_sum, key_word)
        downloadImages.start_download()
