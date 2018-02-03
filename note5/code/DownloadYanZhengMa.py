# -*- coding:utf-8 -*-
import re
import uuid
import requests
import os


class DownloadYanZhengMa:
    def __init__(self,save_path,download_max,url):
        self.download_sum = 0
        self.save_path = save_path
        self.download_max = download_max
        self.url = url

    def downloadImages(self):
        try:
            pic = requests.get(self.url, timeout=500)
            pic_name = self.save_path+'/' +str(uuid.uuid1()) + '.png'
            with open(pic_name, 'wb') as f:
                f.write(pic.content)
            self.download_sum += 1
            print '已下载完成'+str(self.download_sum)+'张验证码'
            if self.download_sum >= self.download_max:
                return
            else:
                return self.downloadImages()
        except  Exception, e:
            print '【错误】当前图片无法下载，%s' % e
            return self.downloadImages()


if __name__ == '__main__':
    # 下载验证码的保存路径
    save_path = '../images/download_yanzhengma'
    # 下载验证码的数量
    download_max = 500
    # 下载验证码的网址
    url = 'http://jwsys.ctbu.edu.cn/CheckCode.aspx?'
    downloadYanZhenMa = DownloadYanZhengMa(save_path=save_path, download_max=download_max,url=url)
    downloadYanZhenMa.downloadImages()
