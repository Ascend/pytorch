# PyTorch安装指南

- [简介](#简介)

- [系统依赖库](#系统依赖库)

- [Ascend配套软件](#Ascend配套软件)

- [安装方式](#安装方式)

- [运行](#运行)
- [安装混合精度模块（可选）](#安装混合精度模块（可选）)
- [FAQ](#FAQ)
- [版本说明](#版本说明)

# 简介<a name="简介"></a>

本项目开发了PyTorch Adapter插件，用于昇腾适配PyTorch框架，为使用PyTorch框架的开发者提供昇腾AI处理器的超强算力。用户在准备相关环境进行基于PyTorch框架模型的开发、运行时，可以选择在服务器中手动编译相关模块。

<h2 id="md">前提条件</h2>

- 需完成CANN开发或运行环境的安装，具体操作请参考《CANN 软件安装指南》。
- 需安装python版本为3.7.5、3.8、3.9。

# 系统依赖库<a name="系统依赖库"></a>

## CentOS & EulerOS

yum install -y cmake==3.12.0 zlib-devel libffi-devel openssl-devel libjpeg-turbo-devel gcc-c++ sqlite-devel dos2unix openblas git gcc==7.3.0

## Ubuntu

apt-get install -y gcc==7.3.0 g++ make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev m4 cmake==3.12.0 dos2unix libopenblas-dev git 

# Ascend配套软件<a name="Ascend配套软件"></a>

| AscendPyTorch版本 | CANN版本     | 支持PyTorch版本 | Gitee分支名称  |
| :---------------- | :----------- | :-------------- | -------------- |
| 2.0.2             | CANN 5.0.2   | 1.5.0.post2     | 2.0.2.tr5      |
| 2.0.3             | CANN 5.0.3   | 1.5.0.post3     | 2.0.3.tr5      |
| 2.0.4             | CANN 5.0.4   | 1.5.0.post4     | 2.0.4.tr5      |
| 3.0.rc1           | CANN 5.1.RC1 | 1.5.0.post5     | v1.5.0-3.0.rc1 |
| 3.0.rc1           | CANN 5.1.RC1 | 1.8.1.rc1       | v1.8.1-3.0.rc1 |

# 安装方式<a name="安装方式"></a>

## 安装Pytorch依赖环境

如果使用非root用户安装，用户需要在本步骤的每句命令后加**--user**，示例如：pip3 install pyyaml --user。

```sh
pip3 install pyyaml
pip3 install wheel
```

## 编译安装PyTorch和昇腾插件

首先安装官方torch包，然后编译安装插件。（ARM架构CPU安装，请参见第一条FAQ）

```sh
#x86_64
pip3 install torch==1.8.1+cpu #安装cpu版本PyTorch 下载地址：（https://download.pytorch.org/whl/torch）

#aarch64
#社区未提供arm架构cpu安装包，请参见FAQ第一条，使用源码编译安装pytorch
```

编译生成pytorch插件的二进制安装包。

```
git clone -b v1.8.1-3.0.rc2 https://gitee.com/ascend/pytorch.git & cd pytorch    # 下载master分支代码，进入插件根目录
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


# 运行<a name="运行"></a>

## 运行环境变量

在当前仓库根目录中执行设置环境变量脚本

```
source env.sh
```


## 自定义环境变量

可选的环境变量可能会对运行的模型产生影响:

```
export COMBINED_ENABLE=1 # 非连续两个算子组合类场景优化，可选，开启设置为1
export ACL_DUMP_DATA=1 # 算子数据dump功能，调试时使用，可选，开启设置为1
```

**表 1**  环境变量说明
<a name="zh-cn_topic_0000001152616261_table42017516135"></a>

<table><thead align="left"><tr id="zh-cn_topic_0000001152616261_row16198951191317"><th class="cellrowborder" valign="top" width="55.48%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001152616261_p51981251161315"><a name="zh-cn_topic_0000001152616261_p51981251161315"></a><a name="zh-cn_topic_0000001152616261_p51981251161315"></a>配置项</p>
</th>
<th class="cellrowborder" valign="top" width="44.519999999999996%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001152616261_p9198135114133"><a name="zh-cn_topic_0000001152616261_p9198135114133"></a><a name="zh-cn_topic_0000001152616261_p9198135114133"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001152616261_row6882121917329"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001152616261_p688241953218"><a name="zh-cn_topic_0000001152616261_p688241953218"></a><a name="zh-cn_topic_0000001152616261_p688241953218"></a>LD_LIBRARY_PATH</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001152616261_p1888291915322"><a name="zh-cn_topic_0000001152616261_p1888291915322"></a><a name="zh-cn_topic_0000001152616261_p1888291915322"></a>动态库的查找路径，参考上述举例配置。</p>
<p id="p1292181892120"><a name="p1292181892120"></a><a name="p1292181892120"></a>若训练所在系统环境需要升级gcc（例如CentOS、Debian和BClinux系统），则<span class="parmname" id="parmname795020446318"><a name="parmname795020446318"></a><a name="parmname795020446318"></a>“LD_LIBRARY_PATH”</span>配置项处动态库查找路径需要添加<span class="filepath" id="zh-cn_topic_0256062644_filepath115819811512"><a name="zh-cn_topic_0256062644_filepath115819811512"></a><a name="zh-cn_topic_0256062644_filepath115819811512"></a>“${install_path}/lib64”</span>，其中<span class="filepath" id="zh-cn_topic_0256062644_filepath195951574421"><a name="zh-cn_topic_0256062644_filepath195951574421"></a><a name="zh-cn_topic_0256062644_filepath195951574421"></a>“{install_path}”</span>为gcc升级安装路径。请参见<a href="#安装7-3-0版本gccmd#zh-cn_topic_0000001135347812_zh-cn_topic_0000001173199577_zh-cn_topic_0000001172534867_zh-cn_topic_0276688294_li9745165315131">5</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001152616261_row16194175523010"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001152616261_p16195185523019"><a name="zh-cn_topic_0000001152616261_p16195185523019"></a><a name="zh-cn_topic_0000001152616261_p16195185523019"></a>PYTHONPATH</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001152616261_p19637083322"><a name="zh-cn_topic_0000001152616261_p19637083322"></a><a name="zh-cn_topic_0000001152616261_p19637083322"></a>Python搜索路径，参考上述举例配置。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001152616261_row2954102119329"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001152616261_p195452113218"><a name="zh-cn_topic_0000001152616261_p195452113218"></a><a name="zh-cn_topic_0000001152616261_p195452113218"></a>PATH</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001152616261_p964914893211"><a name="zh-cn_topic_0000001152616261_p964914893211"></a><a name="zh-cn_topic_0000001152616261_p964914893211"></a>可执行程序的查找路径，参考上述举例配置。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001152616261_row58592816294"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001152616261_p1886016892913"><a name="zh-cn_topic_0000001152616261_p1886016892913"></a><a name="zh-cn_topic_0000001152616261_p1886016892913"></a>ASCEND_OPP_PATH</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001152616261_p28608892915"><a name="zh-cn_topic_0000001152616261_p28608892915"></a><a name="zh-cn_topic_0000001152616261_p28608892915"></a>算子根目录，参考上述举例配置。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001152616261_row144592037903"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001152616261_p104601373014"><a name="zh-cn_topic_0000001152616261_p104601373014"></a><a name="zh-cn_topic_0000001152616261_p104601373014"></a>OPTION_EXEC_EXTERN_PLUGIN_PATH</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001152616261_p1046013716017"><a name="zh-cn_topic_0000001152616261_p1046013716017"></a><a name="zh-cn_topic_0000001152616261_p1046013716017"></a>算子信息库路径。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001152616261_row16184379493"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001152616261_p131851873492"><a name="zh-cn_topic_0000001152616261_p131851873492"></a><a name="zh-cn_topic_0000001152616261_p131851873492"></a>ASCEND_AICPU_PATH</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001152616261_p181851575497"><a name="zh-cn_topic_0000001152616261_p181851575497"></a><a name="zh-cn_topic_0000001152616261_p181851575497"></a>aicpu算子包路径。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001152616261_row1680820246202"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001152616261_p4809112415207"><a name="zh-cn_topic_0000001152616261_p4809112415207"></a><a name="zh-cn_topic_0000001152616261_p4809112415207"></a>HCCL_WHITELIST_DISABLE</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001152616261_p952814428206"><a name="zh-cn_topic_0000001152616261_p952814428206"></a><a name="zh-cn_topic_0000001152616261_p952814428206"></a>配置在使用HCCL时是否开启通信白名单。</p>
<a name="ul928845132310"></a><a name="ul928845132310"></a><ul id="ul928845132310"><li>0：开启白名单，无需校验HCCL通信白名单。</li><li>1：关闭白名单，需校验HCCL通信白名单。</li></ul>
<p id="zh-cn_topic_0000001152616261_p5809162416201"><a name="zh-cn_topic_0000001152616261_p5809162416201"></a><a name="zh-cn_topic_0000001152616261_p5809162416201"></a>缺省值为0，默认开启白名单。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001152616261_row0671137162115"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001152616261_p4671203792114"><a name="zh-cn_topic_0000001152616261_p4671203792114"></a><a name="zh-cn_topic_0000001152616261_p4671203792114"></a>HCCL_IF_IP</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001152616261_p1822165982114"><a name="zh-cn_topic_0000001152616261_p1822165982114"></a><a name="zh-cn_topic_0000001152616261_p1822165982114"></a>配置HCCL的初始化通信网卡IP。</p>
<a name="ul2676102292415"></a><a name="ul2676102292415"></a><ul id="ul2676102292415"><li>ip格式为点分十进制。</li><li>暂只支持host网卡。</li></ul>
<p id="zh-cn_topic_0000001152616261_p1167163719217"><a name="zh-cn_topic_0000001152616261_p1167163719217"></a><a name="zh-cn_topic_0000001152616261_p1167163719217"></a>缺省时，按照以下优先级选定host通信网卡名：docker/local以外网卡（网卡名字字典序升序排列）&gt;docker 网卡 &gt; local网卡</p>
</td>
</tr>
<tr id="row743212132309"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p17433111312307"><a name="p17433111312307"></a><a name="p17433111312307"></a>ASCEND_SLOG_PRINT_TO_STDOUT</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p6433151393018"><a name="p6433151393018"></a><a name="p6433151393018"></a>（可选）设置是否开启日志打屏。</p>
<a name="ul760201917473"></a><a name="ul760201917473"></a><ul id="ul760201917473"><li>0：表示采用日志的默认输出方式。</li><li>1：表示日志打屏显示。</li><li>其他值为非法值。</li></ul>
</td>
</tr>
<tr id="row19237171814300"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p14238161893019"><a name="p14238161893019"></a><a name="p14238161893019"></a>ASCEND_GLOBAL_LOG_LEVEL</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p223841810303"><a name="p223841810303"></a><a name="p223841810303"></a>设置应用类日志的全局日志级别。</p>
<a name="ul175714586453"></a><a name="ul175714586453"></a><ul id="ul175714586453"><li>0：对应DEBUG级别。</li><li>1：对应INFO级别。</li><li>2：对应WARNING级别。</li><li>3：对应ERROR级别。</li><li>4：对应NULL级别，不输出日志。</li><li>其他值为非法值。</li></ul>
</td>
</tr>
<tr id="row1348192313303"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p1734815235305"><a name="p1734815235305"></a><a name="p1734815235305"></a>ASCEND_GLOBAL_EVENT_ENABLE</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p12348202373018"><a name="p12348202373018"></a><a name="p12348202373018"></a>设置应用类日志是否开启Event日志。</p>
<a name="ul416352114610"></a><a name="ul416352114610"></a><ul id="ul416352114610"><li>0：不开启Event日志。</li><li>1：开启Event日志。</li><li>其他值为非法值。</li></ul>
</td>
</tr>
<tr id="row78312162301"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p1832171673019"><a name="p1832171673019"></a><a name="p1832171673019"></a>COMBINED_ENABLE</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p583261643014"><a name="p583261643014"></a><a name="p583261643014"></a>（可选）非连续两个算子组合类场景优化，开启设置为1。</p>
</td>
</tr>
<tr id="row183041355123411"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p730435533415"><a name="p730435533415"></a><a name="p730435533415"></a>ACL_DUMP_DATA</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p16304105533412"><a name="p16304105533412"></a><a name="p16304105533412"></a>（可选）算子数据dump功能，调试时使用，开启设置为1。</p>
</td>
</tr>
<tr id="row19173161510309"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001152616261_p16711563237"><a name="zh-cn_topic_0000001152616261_p16711563237"></a><a name="zh-cn_topic_0000001152616261_p16711563237"></a>unset GOMP_CPU_AFFINITY</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001152616261_p0711356152317"><a name="zh-cn_topic_0000001152616261_p0711356152317"></a><a name="zh-cn_topic_0000001152616261_p0711356152317"></a>（可选）当系统为openEuler及其继承操作系统时，如UOS，需设置此命令，取消CPU绑核。</p>
</td>
</tr>
</tbody>
</table>




## 执行单元测试脚本

验证运行, 输出结果OK

```shell
cd test/test_network_ops/
python3 test_div.py
```

# 安装混合精度模块（可选）<a name="安装混合精度模块（可选）"></a>

AscendPyTorch1.8.1集成了AMP模块，也可用于混合精度训练等应用场景，与Apex模块的区别如下，请用户根据功能需要选择使用，若需安装Apex模块请参考相关[README文档](https://gitee.com/ascend/apex)进行编译安装Apex模块。

- AMP

  - 动态loss scale：动态计算loss scale的值并判断是否溢出。
  - 不支持tensor融合功能。

- APEX

  - O1配置模式：Conv，Matmul等使用float16精度计算，其他如softmax、BN使用float32精度。
  - O2配置模式：除BN使用float32精度外，其他部分使用float16精度。
  - 静态loss scale：静态设置参数确保混合精度训练收敛。
  - 动态loss scale：动态计算loss scale的值并判断是否溢出。


# FAQ<a name="FAQ"></a>

## CPU架构为ARM架构时，由于社区未提供ARM架构CPU版本的torch包，无法使用PIP3命令安装PyTorch1.8.1，需要使用源码编译安装。

下载PyTorch v1.8.1源码包。

```
git clone -b v1.8.1 https://github.com/pytorch/pytorch.git --depth=1 pytorch_v1.8.1
```

进入源码包获取被动依赖代码。

```
cd pytorch_v1.8.1
git submodule sync
git submodule update --init --recursive 
```

执行编译安装。

```
python3 setup.py install
```

## 在PIP设置为华为源时，安装requirments.txt中的typing依赖后，会导致python环境错误。

在PIP设置为华为源时，需打开requirments.txt文件，删除typing依赖，再执行命令。

```
pip3 install -r requirments.txt
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
cd /urs/local/Ascend/ascend-toolkit/latest/{arch}-linux/lib64
```

用户态:

```
cd /urs/local/Ascend/nnae/latest/{arch}-linux/lib64

pip3 install --upgrade topi-0.4.0-py3-none-any.whl

pip3 install --upgrade te-0.4.0-py3-none-any.whl
```

## 命令行安装cmake依赖时提示找不到包、编译cmake报错版本过低，可使用安装脚本或源码编译安装。

下载安装脚本安装cmake。（参考cmake官网）

​		X86_64环境推荐脚本安装：cmake-3.12.0-Linux-x86_64.sh


部分源下载cmake时会提示无法找到包，需要使用源码编译安装。

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

1. 下载gcc-7.3.0.tar.gz，下载地址为[https://mirrors.tuna.tsinghua.edu.cn/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz](https://mirrors.tuna.tsinghua.edu.cn/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)。

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

      如果上述命令校验失败，需要确保依赖包为一次性下载成功，无重复下载现象。

   3. <a name="zh-cn_topic_0000001135347812_zh-cn_topic_0000001173199577_zh-cn_topic_0000001172534867_zh-cn_topic_0276688294_li1649343041310"></a>执行配置、编译和安装命令：

      ```
      ./configure --enable-languages=c,c++ --disable-multilib --with-system-zlib --prefix=/usr/local/linux_gcc7.3.0
      make -j15    # 通过grep -w processor /proc/cpuinfo|wc -l查看cpu数，示例为15，用户可自行设置相应参数。
      make install    
      ```

      >![](D:\projects\pzrpytorch\pytorch\docs\zh\PyTorch安装指南\public_sys-resources\icon-notice.gif) **须知：** 
      >其中“--prefix“参数用于指定linux\_gcc7.3.0安装路径，用户可自行配置，但注意不要配置为“/usr/local“及“/usr“，因为会与系统使用软件源默认安装的gcc相冲突，导致系统原始gcc编译环境被破坏。示例指定为“/usr/local/linux\_gcc7.3.0“。


5. 配置环境变量。

   当用户执行训练时，需要用到gcc升级后的编译环境，因此要在训练脚本中配置环境变量，通过如下命令配置。

   ```
   export LD_LIBRARY_PATH=${install_path}/lib64:${LD_LIBRARY_PATH}
   ```

   其中$\{install\_path\}为[3.](#zh-cn_topic_0000001135347812_zh-cn_topic_0000001173199577_zh-cn_topic_0000001172534867_zh-cn_topic_0276688294_li1649343041310)中配置的gcc7.3.0安装路径，本示例为“/usr/local/gcc7.3.0/“。

   >![](D:\projects\pzrpytorch\pytorch\docs\zh\PyTorch安装指南\public_sys-resources\icon-note.gif) **说明：** 
   >本步骤为用户在需要用到gcc升级后的编译环境时才配置环境变量。

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

![](D:\projects\pzrpytorch\pytorch\figures\FAQ.png)

请用户参考以下命令，重启容器。

```sh
docker run -it --ipc=host \
--device=/dev/davinciX \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver \
-v /usr/local/dcmi \
-v /usr/local/bin/npu-smi \
${镜像名称}:{tag} \
/bin/bash
```

参数说明：

/dev/davinciX：NPU设配，X是芯片物理ID号例如davinci0。

/dev/davinci_manager：管理设备。

/dev/devmm_svm：管理设备。

/dev/hisi_hdc：管理设备。

/usr/local/Ascend/driver：驱动目录。

/usr/local/dcmi：DCMI目录。

/usr/local/bin/npu-smi：npu-smi工具。

${镜像名称}:{tag}：镜像名称与版本号。

## 安装-torch--whl-提示-torch-1-5-0xxxx-与-torchvision-所依赖的版本不匹配

安装“torch-\*.whl”时，提示"ERROR：torchvision 0.6.0 has requirement torch==1.5.0, but you'll have torch 1.5.0a0+1977093 which is incompatible"。
![](D:\projects\pzrpytorch\pytorch\figures\zh-cn_image_0000001190081735.png)

安装torch时，会自动触发torchvision进行依赖版本检查，环境中安装的torchvision版本为0.6.0，检查时发现我们安装的torch-\*.whl的版本号与要求的1.5.0不一致，所以提示报错，但实际安装成功 。

对实际结果无影响，无需处理。



# 版本说明<a name='版本说明'></a>

版本说明请参阅[ReleseNote](docs/zh/RELEASENOTE)
