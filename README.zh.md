# AscendPyTorch


<h1 id="简介md">简介</h1>
本项目开发了PyTorch Adapter插件，用于昇腾适配PyTorch框架，为使用PyTorch框架的开发者提供昇腾AI处理器的超强算力。用户在准备相关环境进行基于PyTorch框架模型的开发、运行时，可以选择在服务器中手动编译安装PyTorch框架相关模块。

<h3 id="前提条件md">前提条件</h3>

- 需完成CANN开发或运行环境的安装，具体操作请参考《CANN 软件安装指南》。
- Python支持版本：3.7.5、3.8。

# 系统依赖库

## CentOS & EulerOS

yum install -y patch cmake==3.12.0 zlib-devel libffi-devel openssl-devel libjpeg-turbo-devel gcc-c++ sqlite-devel dos2unix openblas git gcc==7.3.0 dos2unix

## Ubuntu

apt-get install -y patch gcc==7.3.0 g++ make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev m4 cmake==3.12.0 dos2unix libopenblas-dev git dos2unix


# Ascend配套软件

| AscendPyTorch版本 | CANN版本 | 支持PyTorch版本 | Gitee分支名称 |
| :------------ | :----------- | :----------- | ------------ |
| 2.0.2 | CANN 5.0.2 | 1.5.0.post2 | 2.0.2.tr5 |
| 2.0.3 | CANN 5.0.3 | 1.5.0.post3 | 2.0.3.tr5 |
| 2.0.4 | CANN 5.0.4 | 1.5.0.post4 | 2.0.4.tr5 |
| 3.0.rc1 | CANN 5.1.RC1 | 1.5.0.post5 | v1.5.0-3.0.rc1 |
| 3.0.rc1 | CANN 5.1.RC1 | 1.8.1.rc1 | v1.8.1-3.0.rc1 |
| 3.0.rc2 | CANN 5.1.RC2 | 1.5.0.post6 | v1.5.0-3.0.rc2 |
| 3.0.rc2 | CANN 5.1.RC2 | 1.8.1.rc2 | v1.8.1-3.0.rc2 |

# 安装方式

## 安装PyTorch依赖环境


获取适配昇腾AI处理器的PyTorch源代码（即当前仓库代码）。

   ```
   git clone -b v1.5.0 https://gitee.com/ascend/pytorch.git
   ```

## 获取原生PyTorch源代码和third_party代码

在当前仓库根目录pytorch/下获取原生PyTorch1.5.0的源代码。请关注PyTorch原生社区的安全板块与Issue板块是否有安全相关问题修复，并根据社区修复及时更新原生PyTorch代码。

```sh
cd pytorch
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

## 安装pytorch

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




可选的环境变量可能会对运行的模型产生影响:

```

export COMBINED_ENABLE=1 # 非连续转连续二级推导优化，可选，开启设置为1。当模型中有大量AsStrided高耗时算子被调用时，可以尝试开启此优化以获得潜在的device执行效率的提升
export ACL_DUMP_DATA=1 # 算子数据dump功能，调试时使用，可选，开启设置为1
export TASK_QUEUE_ENABLE=1 # 使用异步任务下发，异步调用acl接口，建议默认开启，开启设置为1

```
当系统为openEuler及其继承操作系统时，如UOS，需设置此命令，取消CPU绑核。

    ```
    # unset GOMP_CPU_AFFINITY
    ```
**表 1**  环境变量说明

<a name="zh-cn_topic_0000001152616261_table42017516135"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001152616261_row16198951191317"><th class="cellrowborder" valign="top" width="55.48%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001152616261_p51981251161315"><a name="zh-cn_topic_0000001152616261_p51981251161315"></a><a name="zh-cn_topic_0000001152616261_p51981251161315"></a>配置项</p>
</th>
<th class="cellrowborder" valign="top" width="44.519999999999996%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001152616261_p9198135114133"><a name="zh-cn_topic_0000001152616261_p9198135114133"></a><a name="zh-cn_topic_0000001152616261_p9198135114133"></a>说明</p>
</td>
</tr>
<tr id="row78312162301"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p1832171673019"><a name="p1832171673019"></a><a name="p1832171673019"></a>COMBINED_ENABLE</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p583261643014"><a name="p583261643014"></a><a name="p583261643014"></a>（可选）非连续转连续二级推导优化，可选，开启设置为1。当模型中有大量AsStrided高耗时算子被调用时，可以尝试开启此优化以获得潜在的device执行效率的提升。</p>
</td>
</tr>
<tr id="row183041355123411"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p730435533415"><a name="p730435533415"></a><a name="p730435533415"></a>ACL_DUMP_DATA</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p16304105533412"><a name="p16304105533412"></a><a name="p16304105533412"></a>（可选）算子数据dump功能，调试时使用，开启设置为1。</p>
</td>
</tr>
<tr id="row19173161510309"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001152616261_p16711563237"><a name="zh-cn_topic_0000001152616261_p16711563237"></a><a name="zh-cn_topic_0000001152616261_p16711563237"></a>TASK_QUEUE_ENABLE</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001152616261_p0711356152317"><a name="zh-cn_topic_0000001152616261_p0711356152317"></a><a name="zh-cn_topic_0000001152616261_p0711356152317"></a>（可选）使用异步任务下发，异步调用acl接口，建议默认开启，开启设置为1。 </p>
</td>
</tbody>
</table>


## 执行单元测试脚本

验证运行, 输出结果OK


```shell
// 根据前述版本，选择对应的测试脚本，以下为1.5.0版本
cd ../
python3 pytorch1.5.0/test/test_npu/test_network_ops/test_div.py
```
<h3 id="安装混合精度模块md">安装混合精度模块（可选）</h3>
请用户根据以下功能需要选择使用，若需要安装Apex模块请参考相关[README文档](https://gitee.com/ascend/apex/tree/v1.5.0/)进行编译安装Apex模块。

- APEX
  - O1配置模式：Conv，Matmul等使用float16精度计算，其他如softmax、BN使用float32精度。
  - O2配置模式：除BN使用float32精度外，其他部分使用float16精度。
  - 静态loss scale：静态设置参数确保混合精度训练收敛。
  - 动态loss scale：动态计算loss scale的值并判断是否溢出。

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

## 在PIP设置为华为源时，安装requirments.txt中的typing依赖后，会导致python环境错误。

在PIP设置为华为源时，需打开requirments.txt文件，删除typing依赖，再执行命令。

```
pip3 install -r requirments.txt
```



## import torch 报找不到libhccl.so错误

未配置环境变量，需通过env.sh脚本配置

```
source pytorch/pytorch1.5.0/src/env.sh
```



## 编译过程执行bash build.sh报错no module named yaml/typing_extensions.

pytorch编译依赖 yaml库和typing_extensions库，需要手动安装。

```
pip3 install pyyaml
pip3 install typing_extensions
```

安装成功后，注意需要执行make clean在执行bash build.sh进行编译，否则可能因缓存出现未知编译错误。



## 运行遇到找不到te问题

开发态:

```
cd /urs/local/Ascend/ascend-toolkit/latest/{arch}-linux/lib64 #{arch}为架构名称。

pip3 install --upgrade topi-0.4.0-py3-none-any.whl

pip3 install --upgrade te-0.4.0-py3-none-any.whl
```

用户态:

```
cd /urs/local/Ascend/nnae/latest/{arch}-linux/lib64 #{arch}为架构名称

pip3 install --upgrade topi-0.4.0-py3-none-any.whl

pip3 install --upgrade te-0.4.0-py3-none-any.whl
```



## 命令行安装cmake依赖时提示找不到包、编译cmake报错版本过低，可使用安装脚本或源码编译安装。

方法一：下载安装脚本安装cmake。（参考cmake官网）

 X86_64环境脚本安装：cmake-3.12.0-Linux-x86_64.sh
 aarch64环境脚本安装：cmake-3.12.0-Linux-aarch64.sh

方法二：使用源码编译安装。

1. 获取Cmake软件包。

   ```
   wget https://cmake.org/files/v3.12/cmake-3.12.0.tar.gz --no-check-certificate
   ```

2. 解压并进入软件包目录。

   ```
   tar -xf cmake-3.12.0.tar.gz
   cd cmake-3.12.0/
   ```

3. 执行配置、编译和安装命令。

   ```
   ./configure --prefix=/usr/local/cmake
   make && make install
   ```

4. 设置软连接。

   ```
   ln -s /usr/local/cmake/bin/cmake /usr/bin/cmake
   ```

5. 执行如下命令验证是否安装成功。

   ```
   cmake --version
   ```

   如显示“cmake version 3.12.0”则表示安装成功。



## 命令行安装gcc依赖时提示找不到包、编译时gcc报错问题

部分源下载gcc时会提示无法找到包，需要使用源码编译安装。

以下步骤请在root用户下执行。

1. 下载gcc-7.3.0.tar.gz，下载地址为[https://mirrors.tuna.tsinghua.edu.cn/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz](https://gitee.com/link?target=https%3A%2F%2Fmirrors.tuna.tsinghua.edu.cn%2Fgnu%2Fgcc%2Fgcc-7.3.0%2Fgcc-7.3.0.tar.gz)。

2. 安装gcc时候会占用大量临时空间，所以先执行下面的命令清空/tmp目录：

   ```
   sudo rm -rf /tmp/*
   ```

3. 安装依赖（以CentOS和Ubuntu系统为例）。

   - CentOS执行如下命令安装。

     ```
     yum install bzip2    
     ```

   - Ubuntu执行如下命令安装。

     ```
     apt-get install bzip2    
     ```

4. 编译安装gcc。

   1. 进入gcc-7.3.0.tar.gz源码包所在目录，解压源码包，命令为：

      ```
      tar -zxvf gcc-7.3.0.tar.gz
      ```

   2. 进入解压后的文件夹，执行如下命令下载gcc依赖包：

      ```
      cd gcc-7.3.0
      ./contrib/download_prerequisites
      ```

      如果执行上述命令报错，需要执行如下命令在“gcc-7.3.0/“文件夹下下载依赖包：

      ```
      wget http://gcc.gnu.org/pub/gcc/infrastructure/gmp-6.1.0.tar.bz2
      wget http://gcc.gnu.org/pub/gcc/infrastructure/mpfr-3.1.4.tar.bz2
      wget http://gcc.gnu.org/pub/gcc/infrastructure/mpc-1.0.3.tar.gz
      wget http://gcc.gnu.org/pub/gcc/infrastructure/isl-0.16.1.tar.bz2
      ```

      下载好上述依赖包后，重新执行以下命令：

      ```
      ./contrib/download_prerequisites
      ```

      如果命令校验失败，需要确认上述依赖包在文件夹中的唯一性，无重复下载，若存在重复的依赖包，需删除。

   3. 执行配置、编译和安装命令：

      ```
      ./configure --enable-languages=c,c++ --disable-multilib --with-system-zlib --prefix=/usr/local/linux_gcc7.3.0
      make -j15    # 通过grep -w processor /proc/cpuinfo|wc -l查看cpu数，示例为15，用户可自行设置相应参数。
      make install    
      ```

      > ![img](figures/icon-notice.gif) **须知：** 其中“--prefix“参数用于指定linux_gcc7.3.0安装路径，用户可自行配置，但注意不要配置为“/usr/local“及“/usr“，因为会与系统使用软件源默认安装的gcc相冲突，导致系统原始gcc编译环境被破坏。示例指定为“/usr/local/linux_gcc7.3.0“。

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



## ARM环境pip安装torchvision失败

可采用源码安装（需先安装昇腾pytorch并通过env.sh配置环境变量）

```
git clone -b v0.6.0 https://github.com/pytorch/vision.git 
cd vision
python setup.py install
```

验证torchvision是否安装成功

```
python -c "import torchvision"
```

若不报错，则说明安装成功



# 版本说明

版本说明请参阅[ReleseNote](docs/zh/RELEASENOTE)