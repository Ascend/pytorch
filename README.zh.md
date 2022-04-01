# AscendPyTorch


# 项目简介
本项目开发了PyTorch Adapter插件，用于昇腾适配PyTorch框架，为使用PyTorch框架的开发者提供昇腾AI处理器的超强算力。

# 编译/执行约束

gcc版本: 7.3.0（只在编译场景要求）

cmake版本：3.12.0以上版本（只在编译场景要求）

python版本：3.7.5、3.8.x

# 系统依赖库

## CentOS & EulerOS

yum install -y cmake zlib-devel libffi-devel openssl-devel libjpeg-turbo-devel gcc-c++ sqlite-devel dos2unix openblas

## Ubuntu

apt-get install -y gcc g++ make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev m4 cmake dos2unix libopenblas-dev


# Ascend配套软件

| AscendPyTorch版本 | CANN版本 | 支持PyTorch版本 |
| :------------ | :----------- | :----------- |
| 2.0.2 | CANN 5.0.2 | 1.5.0.post2 |
| 2.0.3 | CANN 5.0.3 | 1.5.0.post3 |
| 2.0.4 | CANN 5.0.4 | 1.5.0.post4 |
| 3.0.rc1 | CANN 5.0.4 | 1.5.0.post5，1.8.1.rc1 |

# 使用方式 --生成全量代码并编译

## 获取适配昇腾AI处理器的PyTorch源代码patch

获取适配昇腾AI处理器的PyTorch源代码（即当前仓库代码），并切换到所需的分支。

   ```
   git clone https://gitee.com/ascend/pytorch.git
   # 当前master分支为pytorch 1.8.1版本，需要1.5.0版本请使用git checkout 命令切换到v1.5.0对应版本分支
   cd pytorch
   git checkout -b v1.5.0-3.0.rc1 remotes/origin/v1.5.0-3.0.rc1
   ```

## 获取原生PyTorch源代码和third_party代码

在当前仓库根目录pytorch/下获取原生PyTorch的源代码

```sh
git clone -b v1.5.0 --depth=1 https://github.com/pytorch/pytorch.git
```

进入到pytorch/pytorch/目录下, 获取PyTorch被动依赖代码(获取时间较长，请耐心等待)。

```sh
cd pytorch
git submodule sync
git submodule update --init --recursive
```

完成且没有报错之后就生成了PyTorch及其依赖的三方代码

## 生成适配昇腾AI处理器的PyTorch全量代码。

进入到pytorch/scripts目录，根据选择的版本执行，执行脚本（注意：下载原生Pytorch源代码和下面版本要对应，否则可能出错）

```sh
cd ../scripts/
bash gen.sh
```

会在pytorch/pytorch/目录中生成npu适配全量代码


## python依赖库

进入到pytorch/pytorch/目录，依赖库安装:

```python3
cd ../pytorch
pip3 install -r requirements.txt
```


## 编译torch的二进制包

在pytorch/pytorch/目录，执行

```sh
# python3.7版本
bash build.sh
或者
bash build.sh --python=3.7（推荐）

# python3.8版本
bash build.sh --python=3.8
```

生成的二进制包在pytorch/pytorch/dist/目录下

# 安装

### （以1.5.0版本为例，1.8.1版本同理）

**x86_64:**

torch-1.5.0+ascend-cp37-cp37m-linux_x86_64.whl (实际可能附带小版本号例如torch-1.5.0.post2+ascend-cp37-cp37m-linux_x86_64.whl)

```shell
cd dist
pip3 uninstall torch
pip3 install --upgrade torch-1.5.0+ascend-cp37-cp37m-linux_x86_64.whl
```


**arm:**

torch-1.5.0+ascend-cp37-cp37m-linux_aarch64.whl (实际可能附带小版本号例如torch-1.5.0.post2+ascend-cp37-cp37m-linux_aarch64.whl)

```shell
cd dist
pip3 uninstall torch
pip3 install --upgrade torch-1.5.0+ascend-cp37-cp37m-linux_aarch64.whl
```


# 运行

## 运行环境变量

在pytorch/pytorch/中执行设置环境变量脚本

```
cd ../
source env.sh
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

```shell
// 根据前述版本，选择对应的测试脚本，以下为1.5.0版本
cd ../
python3 pytorch1.5.0/test/test_npu/test_div.py
```

# 文档

有关安装指南、模型迁移和训练/推理教程和API列表等更多详细信息，请参考[用户文档](docs/zh)。

# 建议与交流

热忱希望各位在用户社区加入讨论，并贡献您的建议，我们会尽快给您回复。

# 分支维护策略

Ascend PyTorch的版本分支有以下几种维护阶段：

| **状态**       | **持续时间**    | **说明**                                          |
|-------------|---------------|--------------------------------------------------|
| Planning    | 1 - 3 months  | 特性规划。                     |
| Development | 3 months      | 特性开发。                  |
| Maintained  | 6 - 12 months | 允许所有问题修复的合入，并发布版本。 |
| Unmaintained| 0 - 3 months  | 允许所有问题修复的合入，无专人维护，不再发布版本。                                                 |
| End Of Life (EOL) |  N/A |  不再接受修改合入该分支。    |

# 现有分支维护状态

| **分支名** | **当前状态**  | **上线时间**          | **后续状态**                           | **EOL 日期**|
|------------|--------------|----------------------|----------------------------------------|------------|
| **v2.0.2**   | Maintained   | 2021-07-29           | Unmaintained <br> 2022-07-29 estimated |            |
| **v2.0.3**   | Maintained   | 2021-10-15           | Unmaintained <br> 2022-10-15 estimated |            |
| **v2.0.4**   | Maintained   | 2022-01-15           | Unmaintained <br> 2023-01-15 estimated |            |
| **v3.0.rc1**   | Maintained   | 2022-04-10           | Unmaintained <br> 2023-04-10 estimated |            |


# FAQ

## 编译过程执行bash build.sh报错no module named yaml/typing_extensions.

pytorch编译依赖 yaml库和typing_extensions库，需要手动安装。

pip3 install pyyaml
pip3 install typing_extensions

安装成功后，注意需要执行make clean在执行bash build.sh进行编译，否则可能因缓存出现未知编译错误。

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

# 版本说明

版本说明请参阅[ReleseNote](docs/zh/RELEASENOTE)