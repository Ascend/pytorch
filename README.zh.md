# AscendPyTorch


# 项目简介
本项目开发了PyTorch Adapter插件，用于昇腾适配PyTorch框架，为使用PyTorch框架的开发者提供昇腾AI处理器的超强算力。

# 编译/执行约束

gcc版本: 7.3.0（只在编译场景要求）

cmake版本：3.12.0以上版本（只在编译场景要求）

python版本：3.7.5、3.8.x、3.9.x（PyTorch1.5不支持python3.9.x）

# 系统依赖库

## CentOS & EulerOS

yum install -y cmake zlib-devel libffi-devel openssl-devel libjpeg-turbo-devel gcc-c++ sqlite-devel dos2unix openblas

## Ubuntu

apt-get install -y gcc g++ make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev m4 cmake dos2unix libopenblas-dev


# Ascend配套软件

| AscendPyTorch版本 | CANN版本 | 支持PyTorch版本 | Gitee分支名称 |
| :------------ | :----------- | :----------- | ------------- |
| 2.0.2 | CANN 5.0.2 | 1.5.0.post2 | 2.0.2.tr5 |
| 2.0.3 | CANN 5.0.3 | 1.5.0.post3 | 2.0.3.tr5 |
| 2.0.4 | CANN 5.0.4 | 1.5.0.post4 | 2.0.4.tr5 |
| 3.0.rc1 | CANN 5.1.RC1 | 1.5.0.post5 | v1.5.0-3.0.rc1 |
| 3.0.rc1 | CANN 5.1.RC1 | 1.8.1.rc1 | v1.8.1-3.0.rc1 |

# 安装方式

## 编译安装PyTorch和昇腾插件

首先安装官方torch包，然后编译安装插件
```sh
pip3 install torch==1.8.1
```

编译生成pytorch插件的二进制安装包。

```
git clone https://gitee.com/ascend/pytorch.git & cd pytorch    # 下载代码，进入插件根目录
# 指定python版本编包方式：
bash ci/build.sh --python=3.7
# 或
bash ci/build.sh --python=3.8
# 或
bash ci/build.sh --python=3.9
```

然后安装pytorch/dist下生成的插件torch_npu包

```
pip3 install --upgrade dist/torch_npu-1.8.1rc1-cp37-cp37m-linux_{arch}.whl
```


# 运行

## 运行环境变量

在当前仓库根目录中执行设置环境变量脚本

```
cd ../
source env.sh
```


## 自定义环境变量

以下环境变量为NPU场景下使用的功能类或可以提升性能的环境变量：

```
export TASK_QUEUE_ENABLE=1 # 使用异步任务下发，异步调用acl接口，建议默认开启，开启设置为1
```

可选的环境变量可能会对运行的模型产生影响:

```
export DYNAMIC_COMPILE_ENABLE=1  # 动态shape特性功能，针对shape变化场景，可选，开启设置为1
export COMBINED_ENABLE=1 # 非连续两个算子组合类场景优化，可选，开启设置为1
export ACL_DUMP_DATA=1 # 算子数据dump功能，调试时使用，可选，开启设置为1
export DYNAMIC_OP="ADD#MUL" # 算子实现，ADD和MUL算子在不同场景下有不同的性能表现。可选
```


## 执行单元测试脚本

验证运行, 输出结果OK

```shell
cd test/test_network_ops/
python3 test_div.py
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

cd /urs/local/Ascend/ascend-toolkit/latest/{arch}-linux/lib64

用户态:

cd /urs/local/Ascend/nnae/latest/{arch}-linux/lib64

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

## 容器中未挂载device问题

在容器中运行脚本出现NPU相关ERROR。由于启动容器实例时，未挂载device参数，导致无法正常启动实例。

![](figures/FAQ.png)

请用户参考以下命令，重启容器。

```sh
docker run -it --ipc=host \
--device=/dev/davinciX \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver/{lib64,include,tools}
-v /usr/local/dcmi
-v /usr/local/bin/npu-smi
```



# 版本说明

版本说明请参阅[ReleseNote](docs/zh/RELEASENOTE)