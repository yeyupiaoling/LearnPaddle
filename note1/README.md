@[toc]
*  PaddlePaddle的新版本安装教程请参考：[《PaddlePaddle从入门到炼丹》一——新版本PaddlePaddle的安装](https://blog.csdn.net/qq_33200967/article/details/83052060)
# 环境
----------
系统：Ubuntu 16.0.4（64位）
处理器：Intel(R) Celeron(R) CPU
内存：8G
环境：Python 2.7

# Windows系统的安装
-------
PaddlePaddle目前还不支持Windows，如果读者直接在Windows上安装PaddlePaddlePaddle的话，就会提示没有找到该安装包。如果读者一定要在Windows上工作的话，笔者提供两个建议：一、在Windows系统上使用Docker容器，在Docker容器上安装带有PaddlePaddle的镜像；二、在Windows系统上安装虚拟机，再在虚拟机上安装Ubuntu。

本篇文章基于 PaddlePaddle 0.11.0、Python 2.7

## 在Windows上安装Docker容器
首先下载Docker容器的工具包DockerToolbox，笔者使用这个安装包不仅仅只有Docker，它还包含了VirtualBox虚拟机，使用者工具包我们就不用单独去安装VirtualBox虚拟机了，DockerToolbox的官网下载地址：
```text
https://docs.docker.com/toolbox/toolbox_install_windows/
```
下载之后，就可以直接安装了，双击安装包，开始安装
![这里写图片描述](http://img.blog.csdn.net/20180308095050976?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

选择安装路径，笔者使用默认的安装路径
![这里写图片描述](http://img.blog.csdn.net/20180308095156295?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

然后安装所依赖的软件，因为笔者之前在电脑上已经安装了git，所以在这里就不安装了，其他都要勾选
![这里写图片描述](http://img.blog.csdn.net/201803080957416?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

这一步不用修改什么，让程序为我们创建一个桌面快捷键
![这里写图片描述](http://img.blog.csdn.net/20180308095950407?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

最后就可以安装了，等待一小段时间即可
![这里写图片描述](http://img.blog.csdn.net/20180308100130645?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

到这里就安装完成了
![这里写图片描述](http://img.blog.csdn.net/20180308100203710?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

安装完成之后，如果直接启动Docker的话，有可能可能会卡在这里，因为还有下载一个`boot2docker.iso`镜像，网速比较慢的话就可能一直卡在这里。所以我们还要镜像下一步操作
```text
Running pre-create checks...
(default) No default Boot2Docker ISO found locally, downloading the latest release...
(default) Latest release for github.com/boot2docker/boot2docker is v17.12.1-ce
(default) Downloading C:\Users\15696\.docker\machine\cache\boot2docker.iso from https://github.com/boot2docker/boot2docker/releases/download/v17.12.1-ce/boot2docker.iso...
```

在下载DockerToolbox的时候，这个工具就已经带有`boot2docker.iso`镜像了。并且存在DockerToolbox安装的路径上，笔者的路径是：
```
C:\Program Files\Docker Toolbox\boot2docker.iso
```
我们把这个镜像复制到`用户目录\.docker\machine\cache\`，如笔者的目录如下：
```
C:\Users\15696\.docker\machine\cache\
```
复制完成之后，双击桌面快捷方式`Docker Quickstart Terminal`，启动Docker，命令窗口会输出以下信息：
```
Running pre-create checks...
Creating machine...
(default) Copying C:\Users\15696\.docker\machine\cache\boot2docker.iso to C:\Users\15696\.docker\machine\machines\default\boot2docker.iso...
(default) Creating VirtualBox VM...
(default) Creating SSH key...
(default) Starting the VM...
(default) Check network to re-create if needed...
(default) Windows might ask for the permission to create a network adapter. Sometimes, such confirmation window is minimized in the taskbar.
(default) Found a new host-only adapter: "VirtualBox Host-Only Ethernet Adapter #3"
(default) Windows might ask for the permission to configure a network adapter. Sometimes, such confirmation window is minimized in the taskbar.
(default) Windows might ask for the permission to configure a dhcp server. Sometimes, such confirmation window is minimized in the taskbar.
(default) Waiting for an IP...
```
最后看到Docker的logo就表示成功安装Docker容器了
```text


                        ##         .
                  ## ## ##        ==
               ## ## ## ## ##    ===
           /"""""""""""""""""\___/ ===
      ~~~ {~~ ~~~~ ~~~ ~~~~ ~~~ ~ /  ===- ~~~
           \______ o           __/
             \    \         __/
              \____\_______/

docker is configured to use the default machine with IP 192.168.99.100
For help getting started, check out the docs at https://docs.docker.com

Start interactive shell

15696@ɵ MINGW64 ~
$
```
到这就可以使用Docker来安装PaddlePaddle了，具体请看本文章中关于Docker使用PaddlePaddle部分


## 在Windows上安装Ubuntu
在Windows上在Ubuntu就要先安装虚拟机，虚拟机有很多，笔者使用的是开源的VirtualBox虚拟机，VirtualBox的官网：
```
https://www.virtualbox.org/
```
安装完成VirtualBox虚拟机之后，进入到VirtualBox虚拟机中点击`新建`，创建一个系统
![这里写图片描述](http://img.blog.csdn.net/20180308103308666?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

选择分配的内存，我这里只是分配了2G，如果正式使用PaddlePaddle训练模型，这远远不够，读者可以根据需求分配内存
![这里写图片描述](http://img.blog.csdn.net/20180308103340904?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

创建一个虚拟硬盘
![这里写图片描述](http://img.blog.csdn.net/2018030810335188?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

选择默认的VDI硬盘文件类型
![这里写图片描述](http://img.blog.csdn.net/20180308103359856?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

这里最好是选择动态分配硬盘，这样虚拟机会根据实际占用的空间大小使用电脑本身的磁盘大小，这样会减少电脑空间的占用率的。如果是固定大小，那么创建的虚拟机的虚拟硬盘一开始就是用户设置的大小了。
![这里写图片描述](http://img.blog.csdn.net/20180308103409379?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

这里就是选择虚拟硬盘大小的，最后分配20G以上，笔者分配30G，应该够用。
![这里写图片描述](http://img.blog.csdn.net/20180308103418449?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

然后选择刚才创建的Ubuntu系统，点击`设置`，这`系统`中取消勾选`软驱`，然后点击`存储`，选择Ubuntu镜像，笔者使用的是64位Ubuntu 16.04 桌面版的镜像
![这里写图片描述](http://img.blog.csdn.net/20180308103426628?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

最后就可以启动安装Ubuntu了。选择我们创建的Ubuntu系统，点击`启动`
进入到开始安装界面，为了方便使用，笔者选择中文版的
![这里写图片描述](http://img.blog.csdn.net/20180308105013596?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

为了安装之后不用在安装和更新应用，笔者勾选了`安装Ubuntu时下载更新`，这样在安装的时候就已经更新应用了
![这里写图片描述](http://img.blog.csdn.net/20180308105023998?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

然后是选安装的硬盘，因为我们使用的自己创建的整一个硬盘，所以我们可以直接选择`青春整个硬盘并安装Ubuntu`，这里就不用考虑分区和挂载问题了
![这里写图片描述](http://img.blog.csdn.net/20180308105035335?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

选择所在的位置，这没什么要求的，笔者随便选择一个城市
![这里写图片描述](http://img.blog.csdn.net/20180308105046712?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

然后是选择键盘的布局，通常的键盘布局都是`英语（美国）`
![这里写图片描述](http://img.blog.csdn.net/20180308105055899?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

创建Ubuntu的用户名称和密码
![这里写图片描述](http://img.blog.csdn.net/2018030810510962?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

最后就是安装了，这个安装过程可能有点久，耐心等待
![这里写图片描述](http://img.blog.csdn.net/20180308105131187?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

安装完成之后就可以在Windows系统上使用Ubuntu系统了，我们再使用Ubuntu来学习和使用PaddlePaddle做深度学习了。最好安装完成之后，把在`存储`中设置的Ubuntu镜像移除
![这里写图片描述](http://img.blog.csdn.net/20180308112919338?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzMyMDA5Njc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

在本篇文章之后部分都是在Ubuntu上操作，我们都可以使用Ubuntu这虚拟机来完成。
如果读者使用的是Windows 10，可以使用Windows系统自带的Linux子系统，安装教程可以看我之前的文章[Windows10安装Linux子系统](http://blog.csdn.net/qq_33200967/article/details/71950921)

# 使用pip安装
-----------
如果你还没有在pip命令的话，首先要安装pip，要确保安装的pip版本是大于9.0.0的，否则可能无法安装paddlepaddle。
安装pip命令如下：
```
sudo apt install python-pip
```
安装之后，还有看一下pip的的版本`pip --version`，如果版本低于9.0.0，那要先升级pip，先要下载一个升级文件，命令如下：
```
wget https://bootstrap.pypa.io/get-pip.py
```
下载完成之后，可以使用这个文件安装最新的pip了
```
python get-pip.py
```
安装pip就可以动手安装paddlepaddle了。如果权限不够，请在root下执行命。笔者使用了阿里的镜像源，这样下载速度会快很多。
```
pip install paddlepaddle==0.11.0 -i https://mirrors.aliyun.com/pypi/simple/
```
现在就测试看看paddlepaddle有没有，在python的命令终端中试着导入paddlepaddle包：
```python
 import paddle.v2 as paddle
```
如果没有报错的话就证明paddlepaddle安装成功了

# 使用Docker安装
----------
为什么要使用Docker安装paddlepaddle呢，Docker是完全使用沙箱机制的一个容器，在这个容器安装的环境是不会影响到本身系统的环境的。通俗来说，它就是一个虚拟机，但是它本身的性能开销很小。在使用Docker安装paddlepaddle前，首先要安装Docker，通过下面的命令就可以安装了：
```
sudo apt-get install docker.io
```
安装完成之后，可以使用`docker --version`查看Docker的版本，如果有显示，就证明安装成功了。可以使用`docker images`查看已经安装的镜像。
一切都没有问题之后，就可以用Docker安装paddlepaddle了，命令如下：
```
docker pull hub.baidubce.com/paddlepaddle/paddle
```
在这里不得不说的是，这个安装过程非常久，也许是笔者的带宽太小了。安装完成后，可以再使用`docker images`命令查看安装的镜像，应该可以 看到类似这样一个镜像，名字和TAG会相同，其他信息一般不同
```
hub.baidubce.com/paddlepaddle/paddle   latest                         2b1ae16d846e        27 hours ago        1.338 GB
```

# 从源码编译生成安装包
-------------
我们的硬件环境都有很大的不同，官方给出的pip安装包不一定是符合我们的需求，比如笔者的电脑是不支持AVX指令集的，在官方中没找到这个的安装包（也行现在已经有了），所以我们要根据自己的需求来打包一个自己的安装包

## 在本地编译生成安装包
**1. 安装依赖环境**
在一切开始之前，先要安装好依赖环境，下面表格是官方给出的依赖环境
|依赖|	版本|	说明|
|:-----:|:-----:|:-----:|
|GCC|	4.8.2	|推荐使用CentOS的devtools2|
|CMake|	\>=3.2	| |
|Python|	2.7.x|	依赖libpython2.7.so|
|pip|	\>=9.0	 ||
|numpy|	 	 ||
|SWIG|	\>=2.0|	 |
|Go|	\>=1.8|	可选|
1.1 **安装GCC**
一般现在的Ubuntu都是高于个版本了，可以使用`gcc --version`查看安装的版本。比如笔者的是4.8.4，如果你的是版本是低于4.8.2的就要更新一下了
```
sudo apt-get install gcc-4.9
```
1.2 **安装CMake**
先要从官网下CMake源码
```
wget https://cmake.org/files/v3.8/cmake-3.8.0.tar.gz
```
解压源码
```
tar -zxvf cmake-3.8.0.tar.gz
```
依次执行下面的代码
```python
# 进入解压后的目录
cd cmake-3.8.0
# 执行当前目录的bootstrap程序
./bootstrap
# make一下
make
# 开始安装
sudo make install
```
查看是否安装成功，`cmake --version`，如果正常显示版本，那已经安装成功了。
1.3 **安装pip**
关于安装pip9.0.0以上的版本，在上面的**使用pip安装**部分已经讲了，这里就不在熬述了
1.4 **安装numpy**
安装numpy很简单，一条命令就够了
```
sudo apt-get install python-numpy
```
顺便多说一点，matplotlib这个包也经常用到，顺便安装一下
```
sudo apt-get install python-matplotlib
```
1.5 **安装SWIG**
执行下面代码安装SWIG，安装成功之后，使用`swig -version`检查安装结果
```
sudo apt-get install -y git curl gfortran make build-essential automake swig libboost-all-dev
```
1.6 **安装Go**
官方说可选择，那看情况吧，如果像安装安装吧，笔者顺便安装了，就一条代码的事情，老规则`go version`
```
sudo apt-get install golang
```
到这里，依赖环境就已经安装好了，准备安装paddlepaddle

**2.首先要在GitHub上获取paddlepaddle源码**
```
git clone https://github.com/PaddlePaddle/Paddle.git
```
**3.然后输以下命令**
```python
# 进入刚下载的Paddle里面
cd Paddle
# 切换到0.11.0分支
git checkout release/0.11.0
# 创建一个build文件夹
mkdir build
# 进入build文件夹里
cd build
# 这就要选好你的需求了，比如笔者没有使用GPU，不支持AVX，为了节省空间，我把测试关闭了，这样会少很多空间。最后不要少了..
cmake .. -DWITH_GPU=OFF -DWITH_AVX=OFF -DWITH_TESTING=OFF
# 最后make，使用4个线程编译生成你想要的安装包，这个可能很久,一定要有耐心
make -j4
```
经过长久的make之后，终于生成了我们想要的安装包，它的路径在`Paddle/build/python/dist`下，比如笔者在该目录下有这个安装包`paddlepaddle-0.11.0-cp27-cp27mu-linux_x86_64.whl`，你的命名可能不是这个。之后就可以安装了，使用pip安装：
```python
# 请切入到该目录
cd build/python/dist/
# 每个人的安装包名字可能不一样。如果权限不够，请在root下执行命令
pip install paddlepaddle-0.11.0-cp27-cp27mu-linux_x86_64.whl
```
这个我们就已经安装了paddlepaddle，现在就测试看看paddlepaddle有没有安装成功了，在python的命令终端中试着导入paddlepaddle包：
```python
import paddle.v2 as paddle
```
如果没有报错的话就证明paddlepaddle安装成功了

## 在Docker编译生成安装包
使用Docker就轻松很多了，有多轻松，看一下便知，以下的命令都是在Ubuntu本地操作的，全程不用进入到docker镜像中的。
**1.首先要在GitHub上获取paddlepaddle源码**
```
git clone https://github.com/PaddlePaddle/Paddle.git
```
2.切入到项目的根目录下
```
cd Paddle
```
切换到0.11.0分支
```
git checkout release/0.11.0
```
**3.生成安装包**
执行以下代码，生成whl安装包，这个跟在本地操作差不多。
```python
# 启动并进入镜像
docker run -v $PWD:/paddle -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash
# 创建并进入build镜像
mkdir -p /paddle/build && cd /paddle/build
# 安装缺少的依赖环境
pip install protobuf==3.1.0
# 安装依赖环境
apt install patchelf
# 生成编译环境
cmake .. -DWITH_GPU=OFF -DWITH_AVX=OFF -DWITH_TESTING=OFF
# 开始编译
make -j4
```
然后使用`exit`命令退出镜像，再Ubuntu系统本地的`Paddle/build/python/dist`目录下同样会生成一个安装包，这对比在本地生成的安装包，是不是要简单很多，没错这就是Docker强大之处，所有的依赖环境都帮我们安装好了，现在只要安装这个安装包就行了：
```python
# 请切入到该目录
cd build/python/dist/
# 每个人的安装包名字可能不一样。如果权限不够，请在root下执行命令
pip install paddlepaddle-0.11.0-cp27-cp27mu-linux_x86_64.whl
```
同样我们要测试看看paddlepaddle有没有安装成功了，在python的命令终端中试着导入paddlepaddle包：
```python
 import paddle.v2 as paddle
```
如果没有报错的话就证明paddlepaddle安装成功了


# 编译Docker镜像
--------
如果你比较喜欢使用Docker来运行你的paddlepaddle代码，但是有没有你想要的镜像，这是就要自己来制作一个Docker镜像了，比如笔者的电脑是不支持AVX指令集的，还只有CPU，那么我就要一个不用AVX指令集和使用CPU训练的镜像。好吧，我们开始吧
1.我们要从GitHub下载源码：
```
git clone https://github.com/PaddlePaddle/Paddle.git
```
2.安装开发工具到 Docker image里
```python
# 切入到Paddle目录下
cd Paddle
# 切换到0.11.0分支
git checkout release/0.11.0
# 下载依赖环境并创建镜像，别少了最后的.
docker build -t paddle:dev .
```
有可能它不能够命名为`paddle:dev`，我们可以对他从重新命名，ID要是你镜像的ID
```
# docker tag <镜像对应的ID> <镜像名:TAG>
例如：docker tag 1e835127cf33 paddle:dev
```
3.编译
```
# 这个编译要很久的，请耐心等待
docker run --rm -e WITH_GPU=OFF -e WITH_AVX=OFF -v $PWD:/paddle paddle:dev
```
安装完成之后，使用`docker images`查看刚才安装的镜像

# 测试安装环境
----------
我们就使用官方给出的一个例子，来测试我们安装paddlepaddle真的安装成功了
1.创建一个记事本，命名为`housing.py`，并输入以下代码：
```python
import paddle.v2 as paddle

# Initialize PaddlePaddle.
paddle.init(use_gpu=False, trainer_count=1)

# Configure the neural network.
x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(13))
y_predict = paddle.layer.fc(input=x, size=1, act=paddle.activation.Linear())

# Infer using provided test data.
probs = paddle.infer(
    output_layer=y_predict,
    parameters=paddle.dataset.uci_housing.model(),
    input=[item for item in paddle.dataset.uci_housing.test()()])

for i in xrange(len(probs)):
    print 'Predicted price: ${:,.2f}'.format(probs[i][0] * 1000)
```
2.执行一下该代码
在本地执行代码请输入下面的命令
```
python housing.py
```
在Docker上执行代码的请输入下面的代码
```
docker run -v $PWD:/work -w /work -p 8899:8899 hub.baidubce.com/paddlepaddle/paddle python housing.py
```
`-v`命令是把本地目录挂载到docker镜像的目录上，`-w`设置该目录为工作目录，`-p`设置端口号，使用到的镜像是在**使用Docker安装**部分安装的镜像`hub.baidubce.com/paddlepaddle/paddle`

3.终端会输出下面类似的日志
```
I0116 08:40:12.004096     1 Util.cpp:166] commandline:  --use_gpu=False --trainer_count=1 
Cache file /root/.cache/paddle/dataset/fit_a_line.tar/fit_a_line.tar not found, downloading https://github.com/PaddlePaddle/book/raw/develop/01.fit_a_line/fit_a_line.tar
[==================================================]
Cache file /root/.cache/paddle/dataset/uci_housing/housing.data not found, downloading https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data
[==================================================]
Predicted price: $12,316.63
Predicted price: $13,830.34
Predicted price: $11,499.34
Predicted price: $17,395.05
Predicted price: $13,317.67
Predicted price: $16,834.08
Predicted price: $16,632.04
```
如果没有成功运行该代码，报错信息如下，说明安装的paddlepaddle版本过低，请安装高版本的paddlepaddle
```
I0116 13:53:48.957136 15297 Util.cpp:166] commandline:  --use_gpu=False --trainer_count=1
Traceback (most recent call last):
  File "housing.py", line 13, in <module>
    parameters=paddle.dataset.uci_housing.model(),
AttributeError: 'module' object has no attribute 'model'
```

# 最后提示
------
 - 有很多学习者会出现明明安装完成PaddlePaddle了，但是在PaddlePaddle的时候，在初始化PaddlePaddle这一行代码出错
```python
paddle.init(use_gpu=False, trainer_count=1)
```
这个多数是读者的电脑不支持AVX指令集，而在PaddlePaddle的时候，安装的是支持AVX指令集的版本，所以导致在初始化PaddlePaddle的时候报错。所以在安装或者编译PaddlePaddle安装包时，要根据读者电脑本身的情况，选择是否支持AVX指令集。查看电脑是否支持AVX指令集，可以在终端输入以下命令，输出Yes表示支持，输出No表示不支持
```text
if cat /proc/cpuinfo | grep -i avx; then echo Yes; else echo No; fi
```

 - 如果报以下的错误，这是在`/usr/lib`下没有找到对应的动态库，通常情况它们会放在`/usr/local/lib`目录下。可以使用命令`ldconfig`更新一下系统的动态库。
```
ImportError: libmkldnn.so.0: cannot open shared object file: No such file or directory
```

# 项目代码
------
GitHub地址:[https://github.com/yeyupiaoling/LearnPaddle](https://github.com/yeyupiaoling/LearnPaddle)



# 参考资料
------------------
1. http://paddlepaddle.org/
2. https://pip.pypa.io/en/stable/
3. http://www.runoob.com/
4. http://www.linuxidc.com/Linux/2016-12/138489.htm
5. https://www.jianshu.com/p/c6264cd5f5c7
