# AscendPyTorch


# 项目简介
本项目开发了PyTorch Adapter插件，用于昇腾适配PyTorch框架，为使用PyTorch框架的开发者提供昇腾AI处理器的超强算力。

# 编译/执行约束

gcc版本: 7.3.0（只在编译场景要求）

cmake版本：3.12.0以上版本（只在编译场景要求）

python版本：3.7.x


# 系统依赖库

## CentOS & EulerOS

yum install -y cmake zlib-devel libffi-devel openssl-devel libjpeg-turbo-devel gcc-c++ sqlite-devel dos2unix openblas

## Ubuntu

apt-get install -y gcc g++ make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev m4 cmake dos2unix libopenblas-dev


# python依赖库


依赖库安装:

```python3
pip3 install -r requirements.txt
```


# 使用方式 --生成全量代码并编译

## 获取PyTorch源代码和third_party代码

git clone -b v1.5.0 --depth=1 https://github.com/pytorch/pytorch.git

cd pytorch 

获取PyTorch被动依赖代码(获取时间较长，请耐心等待)。

git submodule sync

git submodule update --init --recursive 

## 生成适配昇腾AI处理器的PyTorch代码。

进入到build目录，执行

```sh
bash build.sh
```

会在当前根目录下pytorch目录中生成npu适配全量代码，编译之后的二进制在dist目录下


# 安装

**x86_64:**

torch-1.5.0+ascend-cp37-cp37m-linux_x86_64.whl (实际可能附带小版本号例如torch-1.5.0.post2+ascend-cp37-cp37m-linux_x86_64.whl)

```python
pip3 uninstall torch
pip3 install --upgrade torch-1.5.0+ascend-cp37-cp37m-linux_x86_64.whl
```


**arm:**

torch-1.5.0+ascend-cp37-cp37m-linux_aarch64.whl (实际可能附带小版本号例如torch-1.5.0.post2+ascend-cp37-cp37m-linux_aarch64.whl)

```python
pip3 uninstall torch
pip3 install --upgrade torch-1.5.0+ascend-cp37-cp37m-linux_aarch64.whl
```


# 运行

## 运行环境变量

在当前仓库根目录中执行设置环境变量脚本

```
source src/env.sh
```


## 自定义环境变量

以下环境变量为NPU场景下使用的功能类或可以提升性能的环境变量：

```
export TASK_QUEUE_ENABLE=1 # 使用异步任务下发，异步调用acl接口，建议默认开启，开启设置为1
export PTCOPY_ENABLE=1 # 使用PTCopy算子模式，加速转连续及copy等过程，建议默认开启，开启设置为1
```

可选的环境变量可能会对运行的模型产生影响:

```
export DYNAMIC_COMPILE_ENABLE=1  # 动态shape特性功能，针对shape变化场景，可选，开启设置为1
export COMBINED_ENABLE=1 # 非连续两个算子组合类场景优化，可选，开启设置为1
export TRI_COMBINED_ENABLE=1 # 非连续三个算子组合类场景优化，可选，开启设置为1
export ACL_DUMP_DATA=1 # 算子数据dump功能，调试时使用，可选，开启设置为1
export DYNAMIC_OP="ADD#MUL" # 算子实现，ADD和MUL算子在不同场景下有不同的性能表现。可选
```


## 执行单元测试脚本

验证运行, 输出结果OK

```python
python3 test/test_npu/test_div.py
```

# 路标

以下将展示AscendPyTorch近期的计划，我们会根据用户的反馈诉求，持续调整计划的优先级。

总体而言，我们会努力在以下几个方面不断改进。

    1、pytorch AICPU多核能力补齐
    
    2、pytorch单算子模式const节点输入
    
    3、pytorch场景支持算子非连续输入
    
    4、pytorch单算子模式支持动态shape
    
    5、NLP网络格式转化方案完善
    
    6、pytorch的基线格式调整
    
    7、reshapeType总体方案梳理优化
    
    8、AICPU针对pytorch方案梳理优化

热忱希望各位在用户社区加入讨论，并贡献您的建议。



# FAQ

## 运行遇到找不到te问题

开发态:

cd /urs/local/ascend-toolkit/latest/fwkacllib/lib64

用户态:

cd /urs/local/nnae/latest/fwkacllib/lib64

pip3 install --upgrade topi-0.4.0-py3-none-any.whl

pip3 install --upgrade te-0.4.0-py3-none-any.whl



## 编译cmake报错版本过低

cmake官网下载linux版本安装（当前3.18.0）

1. 使用yum命令安装： yum install -y cmake==3.18.0

2. 下载cmake sh脚本安装：（参考cmake官网）

   X86_64环境推荐脚本安装: cmake-3.18.2-Linux-x86_64.sh

   

## GCC版本问题切换问题

目前存在测试环境从GCC4.8.5 切换到 GCC7.3.0。这个过程容易出现错误导致pytorch编译不过，以下是需要软连接的库

gcc, g++,c++(--version 必须是7.3.0)

libstdc++->libstdc++.so.6.0.24(7.3.0)



## 找不到libblas.so问题

环境缺少openblas库，需要安装openblas库

Centos，EulerOS环境

```sh
yum -y install openblas
```

Ubuntu环境

```sh
apt install libopenblas-dev
```

