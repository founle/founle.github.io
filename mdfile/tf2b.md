#  Debian10基于Anaconda的TensorFlow-gpu 2.0安装（更新中） 

# 关于tensorflow
tensorflow是目前最火的深度学习框架，博主于自学深度学习中搭建基于debian10的tf2.0。此前搭建过基于，win10、debian9、macos10.14的tf1.12，其中win于debian为tensorflow-gpu版本，macos为tensorflow-cpu版本。借此逆天的tf2.0与debian10的重大发布，重装系统，重新干净配置。
## 变化
热乎的刚出炉，等下次填写

## debian10安装
本题略（有时间一起完善）
https://blog.csdn.net/IYALEI/article/details/95753420
## 检查GPU是否支持
sudo apt-get install nvidia-detect
 安装后运行
 nvidia-detect
查看gpu是否支持
## gcc8版本修改为gcc7

    sudo apt-get install gcc-7 g++-7
    sudo rm /usr/bin/gcc
    sudo ln -s /usr/bin/gcc-7 /usr/bin/gcc

## CUDA10.0下载
https://developer.nvidia.com/cuda-toolkit、
![官网下载界面](https://img-blog.csdnimg.cn/20190712213506893.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0lZQUxFSQ==,size_16,color_FFFFFF,t_70)
点击download
![选择下载界面](https://img-blog.csdnimg.cn/20190712213641848.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0lZQUxFSQ==,size_16,color_FFFFFF,t_70)
ubuntu基于debian 所以选择最新版本的ubuntu安装，点击下载run版本的，查找历史版本选择右下脚的LegacyRelesases从中选择即可。
![下载完成](https://img-blog.csdnimg.cn/20190712214124624.png)

强烈建议将下载好的文件从下载目录移动到用户的目录～/中方便在后续命令界面找到。
## CUDA安装



### X服务关闭
由于debian安装显卡需要关掉图形化界面到命令行操作，同时按<kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>F1</kbd>～<kbd>F6</kbd>都可。<kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>F7</kbd>为图形化界面。

    sudo service lightdm stop



### CUDA安装
切换后，登陆帐号密码
输入

    sudo sh cudaXXXX(CUDA名称，温馨提示，输入几个字母按下 tab 键就可自动补全).run
然后点击accept，一路yes飘过，注意提示输入回车地方。

添加环境变量,在用户的根目录下

    vim ./.bashrc 

按i，粘贴输入

    export PATH=/usr/local/cuda-10.0/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

生效

    source ./.bashrc 

安装完成后打开图形化界面

    sudo service lightdm start
    
至此CUDA安装结束。建议重启
输入 `nvcc -V` 查看是否安装成功。

## cudnn安装
官网地址：
https://developer.nvidia.com/rdp/cudnn-archive
注册，登陆，下载对应的指定版本，本文下载为cudnn7.5—cuda10.1版本
解压

    tar -xvzf cudnn-10.1-linux-x64-v7.5.0.56.tgz 

移动到指定位置

    cd cuda
    sudo cp include/cudnn.h /usr/local/cuda/include
    sudo cp lib64/libcudnn.* /usr/local/cuda/lib64

## python安装
此处计划安装python最新的3.7.4
下载（建议在终端里$下操作别用#，下载在～/里）
更新、安装所需包

    sudo apt update
    sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget

下载源码安装

    curl -O https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tar.xz
    tar -xf Python-3.7.4.tar.xz

进去目录开始配置

    cd Python-3.7.4
    ./configure --enable-optimizations

编译并安装

    make && sudo make install

查看版本

    python3.7 --version
    
若出来版本信息则成功。
## Anaconda安装

    sh Anaconda3-2019.03-Linux-x86_64.sh
    
  输入conda -v 查看是否成功与有环境，没有就按照debug-3手动添加。

## conda虚拟环境搭建
查看官方安装文件，需要python版本为3.6
https://tensorflow.google.cn/install/gpu
创建虚拟环境——tf2b

    conda create -n tf2b python=3.6
运行`conda info -e`查看是否安装
激活虚拟环境：

    source activate tf2b
    
进入后显示虚拟环境光标前方带虚拟环境名称（tf2b）如下图：

若关闭环境退出为：

     source deactivate tf2b
## tensorflow-gpu安装
不要退出环境 ，继续使用pip3 下载tensorflow2.0 beta版

    pip3 install tensorflow-gpu==2.0.0-beta1

 
## 验证
打开虚拟环境输入`ipython`

    import tensorflow as tf

若过这一步则安装成功

    tf.test.is_gpu_available()
这部会显示显卡相关的参数

安装结束。
## pycharm结合配置
等待网络中。。。
# *debug
### 1、手动禁用自带NVIDIA第三方驱动nouveau
debian9可以由cuda安装文件卸载，但无奈手气不好，debian10不起效，推荐先使用cuda，若不行使用一下方式。

先检查nouveau是否加载

     lsmod|grep nouveau

若无任何显示则无需进行。
若有进行下列操作：（手动禁用，强烈推荐使用cuda办法，不支持再手动）
创建黑名单禁用nouveau

     vim /etc/modprobe.d/blacklist.conf

在文件里输入

    blacklist nouveau

更新boot

    sudo update-initramfs -u

重启查看是否加载nouveau
  
    lsmod|grep nouveau

若无任何显示则被干掉。



### 2、`ldconfig: /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudnn.so.7 is not a symbolic link`

安装cudnn后，sudo apt update的时候会出现

解决方式：
查看目录是否存在

    ls /usr/local/cuda-10.1//targets/x86_64-linux/lib/

![查看libcudnn.so.7](https://img-blog.csdnimg.cn/20190713144948895.png)
7与7.5都存在，执行重建快捷方式

    sudo ln -sf /usr/local/cuda-10.1//targets/x86_64-linux/lib/libcudnn.so.7.5.0 /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudnn.so.7

问题解决。

### 3、手动添加anaconda环境变量

     sudo vim /etc/profile

按i，粘贴输入

     export PATH=$PATH:/home/userName/anaconda3/bin:$PATH

生效

    source /etc/profile
  
### 4、pip升级
pip install --upgrade pip

致谢——本文有参考以下博文及教程

> （超完整）Linux（debian9）服务器配置tensorflow环境：nvidia驱动、CUDA、cudnn、anaconda
> https://blog.csdn.net/Star_code/article/details/76616958
> tensorflow中国网官方教程
> https://tensorflow.google.cn/install/gpu
> Anaconda安装+conda环境+tensorflow(GPU)+SSD
> https://blog.csdn.net/liuyan20062010/article/details/78872729

