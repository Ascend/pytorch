# PyTorch Ascend Adapter插件

## 简介

本项目开发了名为**torch_npu**的**PyTorch Ascend Adapter**插件，使昇腾NPU可以适配PyTorch框架，为使用PyTorch框架的开发者提供昇腾AI处理器的超强算力。

昇腾为基于华为昇腾处理器和软件的行业应用及服务提供全栈AI计算基础设施。您可以通过访问[昇腾社区](https://www.hiascend.com/zh/)，了解关于昇腾的更多信息。

## 安装

### 使用二进制文件进行安装

我们为用户提供可以快速安装**torch_npu**的whl安装包。在安装**torch_npu**之前，您需要先安装**CANN**软件。[昇腾辅助软件](#昇腾辅助软件)中有更多关于CANN的版本信息。请参考[CANN安装指南](https://www.hiascend.com/zh/software/cann/community)获取**CANN**安装包。

1. **安装PyTorch**

通过 pip 安装 PyTorch。

**aarch64:**

```Python
pip3 install torch==2.0.1
```

**x86:**

```Python
pip3 install torch==2.0.1+cpu  --index-url https://download.pytorch.org/whl/cpu
```

若使用pip命令安装失败，请使用下载链接或进入[PyTorch官方网站](https://pytorch.org/)进行查询下载对应版本。

| 架构    | Python版本 | 下载链接                                                     |
| ------- | ---------- | ------------------------------------------------------------ |
| x86     | Python3.8  | [下载链接](https://download.pytorch.org/whl/cpu/torch-2.0.1%2Bcpu-cp38-cp38-linux_x86_64.whl#sha256=8046f49deae5a3d219b9f6059a1f478ae321f232e660249355a8bf6dcaa810c1) |
| x86     | Python3.9  | [下载链接](https://download.pytorch.org/whl/cpu/torch-2.0.1%2Bcpu-cp39-cp39-linux_x86_64.whl#sha256=73482a223d577407c45685fde9d2a74ba42f0d8d9f6e1e95c08071dc55c47d7b) |
| x86     | Python3.10 | [下载链接](https://download.pytorch.org/whl/cpu/torch-2.0.1%2Bcpu-cp310-cp310-linux_x86_64.whl#sha256=fec257249ba014c68629a1994b0c6e7356e20e1afc77a87b9941a40e5095285d) |
| aarch64 | Python3.8  | [下载链接](https://download.pytorch.org/whl/torch-2.0.1-cp38-cp38-manylinux2014_aarch64.whl) |
| aarch64 | Python3.9  | [下载链接](https://download.pytorch.org/whl/torch-2.0.1-cp39-cp39-manylinux2014_aarch64.whl) |
| aarch64 | Python3.10 | [下载链接](https://download.pytorch.org/whl/torch-2.0.1-cp310-cp310-manylinux2014_aarch64.whl) |

2. **安装torch_npu依赖**

运行一下命令安装依赖。

```Python
pip3 install pyyaml
pip3 install setuptools
```

3. **安装torch_npu**

以下使用AArch64和Python 3.8 为例。

>![](figures/icon-note.gif) **NOTE:****说明：**
>很快将支持通过pip从PyPI安装**torch_npu**。

```Python
wget https://gitee.com/ascend/pytorch/releases/download/v5.0.rc3-pytorch2.0.1/torch_npu-2.0.1-cp38-cp38-linux_aarch64.whl

pip3 install torch_npu-2.0.1-cp38-cp38-linux_aarch64.whl
```

### 使用源代码进行安装

某些特殊场景下，用户可能需要自行编译**torch_npu**。可以根据[昇腾辅助软件表](#昇腾辅助软件)和[PyTorch与Python版本配套表](#PyTorch与Python版本配套表)选择合适的分支。推荐使用Docker镜像编译**torch_npu**，可以通过以下步骤获取：

1. **克隆torch_npu代码仓**

   ```
   git clone https://gitee.com/ascend/pytorch.git -b v2.0.1-5.0.rc3 --depth 1
   ```

2. **构建镜像**

   ```
   cd pytorch/ci/docker/{arch} # {arch} for X86 or ARM
   docker build -t manylinux-builder:v1 .
   ```

3. **进入Docker容器**

   ```
   docker run -it -v /{code_path}/pytorch:/home/pytorch manylinux-builder:v1 bash
   # {code_path} is the torch_npu source code path
   ```

4. **编译torch_npu**

   以**Python 3.8** 为例。

   ```
   cd /home/pytorch
   bash ci/build.sh --python=3.8
   ```

## 入门

### 前提

运行以下命令初始化**CANN**环境变量。

```Shell
# Default path, change it if needed.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 快速验证

 可以通过以下样例快速体验**昇腾NPU**。

```Python
import torch
import torch_npu

x = torch.randn(2, 2).npu()
y = torch.randn(2, 2).npu()
z = x.mm(y)

print(z)
```

## PyTorch与Python版本配套表

| PyTorch版本   | Python版本                                                   |
| ------------- | :----------------------------------------------------------- |
| PyTorch1.11.0 | Python3.7.x(>=3.7.5), Python3.8.x, Python3.9.x, Python3.10.x |
| PyTorch2.0.1  | Python3.8.x, Python3.9.x, Python3.10.x                       |
| PyTorch2.1.0  | Python3.8.x, Python3.9.x, Python3.10.x                       |

## 昇腾辅助软件

<table><thead align="left"><tr id="row721911327225"><th class="cellrowborder" valign="top"  id="mcps1.2.6.1.1"><p id="p0878358152310"><a name="p0878358152310"></a><a name="p0878358152310"></a>CANN版本</p>
</th>
<th class="cellrowborder" valign="top"  id="mcps1.2.6.1.2"><p id="p1833820270243"><a name="p1833820270243"></a><a name="p1833820270243"></a>支持的PyTorch版本</p>
</th>
<th class="cellrowborder" valign="top"  id="mcps1.2.6.1.3"><p id="p7878175817236"><a name="p7878175817236"></a><a name="p7878175817236"></a>支持的Adapter版本</p>
</th>
<th class="cellrowborder" valign="top"  id="mcps1.2.6.1.4"><p id="p58781058202311"><a name="p58781058202311"></a><a name="p58781058202311"></a>Github分支</p>
</th>
<th class="cellrowborder" valign="top"  id="mcps1.2.6.1.5"><p id="p1887865812234"><a name="p1887865812234"></a><a name="p1887865812234"></a>AscendHub镜像版本/名称(<a href="https://ascendhub.huawei.com/#/detail/pytorch-modelzoo" target="_blank" rel="noopener noreferrer">链接</a>)</p>
</th>
</tr>
</thead>
<tbody><tr id="row1121913217224"><td class="cellrowborder" rowspan="3" valign="top"  headers="mcps1.2.6.1.1 "><p id="p387812581238"><a name="p387812581238"></a><a name="p387812581238"></a>CANN 7.0.RC1</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.2 "><p id="p633892712241"><a name="p633892712241"></a><a name="p633892712241"></a>2.1.0</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.3 "><p id="p13878165862310"><a name="p13878165862310"></a><a name="p13878165862310"></a>2.1.0.rc1</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.4 "><p id="p5878658162312"><a name="p5878658162312"></a><a name="p5878658162312"></a>v2.1.0-5.0.rc3</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.5 "><p id="p1887895818234"><a name="p1887895818234"></a><a name="p1887895818234"></a>-</p>
</td>
</tr>
<tr id="row122191232122216"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1333872762420"><a name="p1333872762420"></a><a name="p1333872762420"></a>2.0.1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p08788581239"><a name="p08788581239"></a><a name="p08788581239"></a>2.0.1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p88793589233"><a name="p88793589233"></a><a name="p88793589233"></a>v2.0.1-5.0.rc3</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p18791958192318"><a name="p18791958192318"></a><a name="p18791958192318"></a>-</p>
</td>
</tr>
<tr id="row1220113211225"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p16339162718249"><a name="p16339162718249"></a><a name="p16339162718249"></a>1.11.0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p587919583231"><a name="p587919583231"></a><a name="p587919583231"></a>1.11.0.post4</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1879185852316"><a name="p1879185852316"></a><a name="p1879185852316"></a>v1.11.0-5.0.rc3</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p987915814231"><a name="p987915814231"></a><a name="p987915814231"></a>-</p>
</td>
</tr>
<tr id="row92207321228"><td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.1 "><p id="p5879195862316"><a name="p5879195862316"></a><a name="p5879195862316"></a>CANN 6.3.RC3.1</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.2 "><p id="p13391127102419"><a name="p13391127102419"></a><a name="p13391127102419"></a>1.11.0</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.3 "><p id="p987935812311"><a name="p987935812311"></a><a name="p987935812311"></a>1.11.0.post3</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.4 "><p id="p178797587233"><a name="p178797587233"></a><a name="p178797587233"></a>v1.11.0-5.0.rc2.2</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.5 "><p id="p16879458102316"><a name="p16879458102316"></a><a name="p16879458102316"></a>-</p>
</td>
</tr>
<tr id="row17220143220221"><td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.1 "><p id="p287917582235"><a name="p287917582235"></a><a name="p287917582235"></a>CANN 6.3.RC3</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.2 "><p id="p933902762410"><a name="p933902762410"></a><a name="p933902762410"></a>1.11.0</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.3 "><p id="p10879135819230"><a name="p10879135819230"></a><a name="p10879135819230"></a>1.11.0.post2</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.4 "><p id="p17879125816231"><a name="p17879125816231"></a><a name="p17879125816231"></a>v1.11.0-5.0.rc2.1</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.5 "><p id="p10879155882310"><a name="p10879155882310"></a><a name="p10879155882310"></a>-</p>
</td>
</tr>
<tr id="row422116324224"><td class="cellrowborder" rowspan="3" valign="top"  headers="mcps1.2.6.1.1 "><p id="p13879165832316"><a name="p13879165832316"></a><a name="p13879165832316"></a>CANN 6.3.RC2</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.2 "><p id="p1191901662516"><a name="p1191901662516"></a><a name="p1191901662516"></a>2.0.1</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.3 "><p id="p198791258142317"><a name="p198791258142317"></a><a name="p198791258142317"></a>2.0.1.rc1</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.4 "><p id="p1587955816237"><a name="p1587955816237"></a><a name="p1587955816237"></a>v2.0.1-5.0.rc2</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.5 "><p id="p0879558192313"><a name="p0879558192313"></a><a name="p0879558192313"></a>-</p>
</td>
</tr>
<tr id="row1522123282216"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p19191169254"><a name="p19191169254"></a><a name="p19191169254"></a>1.11.0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p4879185815238"><a name="p4879185815238"></a><a name="p4879185815238"></a>1.8.1.post2</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p887935810232"><a name="p887935810232"></a><a name="p887935810232"></a>v1.8.1-5.0.rc2</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p14879358172313"><a name="p14879358172313"></a><a name="p14879358172313"></a>23.0.RC1-1.8.1</p>
</td>
</tr>
<tr id="row19716641182317"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p691981615257"><a name="p691981615257"></a><a name="p691981615257"></a>1.8.1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p208791758142314"><a name="p208791758142314"></a><a name="p208791758142314"></a>1.11.0.post1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p1987935852319"><a name="p1987935852319"></a><a name="p1987935852319"></a>v1.11.0-5.0.rc2</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1087918585237"><a name="p1087918585237"></a><a name="p1087918585237"></a>23.0.RC1-1.11.0</p>
</td>
</tr>
<tr id="row822143252217"><td class="cellrowborder" rowspan="2" valign="top"  headers="mcps1.2.6.1.1 "><p id="p108803589235"><a name="p108803589235"></a><a name="p108803589235"></a>CANN 6.3.RC1</p>
<p id="p129081558152312"><a name="p129081558152312"></a><a name="p129081558152312"></a></p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.2 "><p id="p12346162018257"><a name="p12346162018257"></a><a name="p12346162018257"></a>1.11.0</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.3 "><p id="p6880205814236"><a name="p6880205814236"></a><a name="p6880205814236"></a>1.8.1.post1</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.4 "><p id="p6880258162313"><a name="p6880258162313"></a><a name="p6880258162313"></a>v1.8.1-5.0.rc1</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.5 "><p id="p08807589236"><a name="p08807589236"></a><a name="p08807589236"></a>-</p>
</td>
</tr>
<tr id="row13745135213239"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p183463202251"><a name="p183463202251"></a><a name="p183463202251"></a>1.8.1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1888019580230"><a name="p1888019580230"></a><a name="p1888019580230"></a>1.11.0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p15880105813236"><a name="p15880105813236"></a><a name="p15880105813236"></a>v1.11.0-5.0.rc1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1588012586230"><a name="p1588012586230"></a><a name="p1588012586230"></a>-</p>
</td>
</tr>
<tr id="row1667135119236"><td class="cellrowborder" rowspan="3" valign="top"  headers="mcps1.2.6.1.1 "><p id="p188801958102315"><a name="p188801958102315"></a><a name="p188801958102315"></a>CANN 6.0.1</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.2 "><p id="p43395279244"><a name="p43395279244"></a><a name="p43395279244"></a>1.5.0</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.3 "><p id="p118801658182310"><a name="p118801658182310"></a><a name="p118801658182310"></a>1.5.0.post8</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.4 "><p id="p4880125819233"><a name="p4880125819233"></a><a name="p4880125819233"></a>v1.5.0-3.0.0</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.5 "><p id="p8880858122319"><a name="p8880858122319"></a><a name="p8880858122319"></a>22.0.0</p>
</td>
</tr>
<tr id="row123884481236"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p0339162717244"><a name="p0339162717244"></a><a name="p0339162717244"></a>1.8.1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p15880115811236"><a name="p15880115811236"></a><a name="p15880115811236"></a>1.8.1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p68808584233"><a name="p68808584233"></a><a name="p68808584233"></a>v1.8.1-3.0.0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1688035812238"><a name="p1688035812238"></a><a name="p1688035812238"></a>22.0.0-1.8.1</p>
</td>
</tr>
<tr id="row133914572237"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p10339102719241"><a name="p10339102719241"></a><a name="p10339102719241"></a>1.11.0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p9880205892318"><a name="p9880205892318"></a><a name="p9880205892318"></a>1.11.0.rc2（beta)</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p888005822314"><a name="p888005822314"></a><a name="p888005822314"></a>v1.11.0-3.0.0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p6880145812238"><a name="p6880145812238"></a><a name="p6880145812238"></a>-</p>
</td>
</tr>
<tr id="row127654514238"><td class="cellrowborder" rowspan="3" valign="top"  headers="mcps1.2.6.1.1 "><p id="p1288085862314"><a name="p1288085862314"></a><a name="p1288085862314"></a>CANN 6.0.RC1</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.2 "><p id="p20327103216255"><a name="p20327103216255"></a><a name="p20327103216255"></a>1.5.0</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.3 "><p id="p7880145811235"><a name="p7880145811235"></a><a name="p7880145811235"></a>1.5.0.post7</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.4 "><p id="p1788095815239"><a name="p1788095815239"></a><a name="p1788095815239"></a>v1.5.0-3.0.rc3</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.5 "><p id="p6880115810237"><a name="p6880115810237"></a><a name="p6880115810237"></a>22.0.RC3</p>
</td>
</tr>
<tr id="row129461142152313"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p14327173292510"><a name="p14327173292510"></a><a name="p14327173292510"></a>1.8.1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p1588035812311"><a name="p1588035812311"></a><a name="p1588035812311"></a>1.8.1.rc3</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p13880165816233"><a name="p13880165816233"></a><a name="p13880165816233"></a>v1.8.1-3.0.rc3</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p108806581238"><a name="p108806581238"></a><a name="p108806581238"></a>22.0.RC3-1.8.1</p>
</td>
</tr>
<tr id="row36411150122319"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p1132743211257"><a name="p1132743211257"></a><a name="p1132743211257"></a>1.11.0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p168800589238"><a name="p168800589238"></a><a name="p168800589238"></a>1.11.0.rc1（beta)</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p28816580233"><a name="p28816580233"></a><a name="p28816580233"></a>v1.11.0-3.0.rc3</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p198811588236"><a name="p198811588236"></a><a name="p198811588236"></a>-</p>
</td>
</tr>
<tr id="row12371144713235"><td class="cellrowborder" rowspan="2" valign="top"  headers="mcps1.2.6.1.1 "><p id="p11881558122317"><a name="p11881558122317"></a><a name="p11881558122317"></a>CANN 5.1.RC2</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.2 "><p id="p629873412510"><a name="p629873412510"></a><a name="p629873412510"></a>1.5.0</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.3 "><p id="p3881105818236"><a name="p3881105818236"></a><a name="p3881105818236"></a>1.5.0.post6</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.4 "><p id="p1088125852319"><a name="p1088125852319"></a><a name="p1088125852319"></a>v1.5.0-3.0.rc2</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.5 "><p id="p20881258112316"><a name="p20881258112316"></a><a name="p20881258112316"></a>22.0.RC2</p>
</td>
</tr>
<tr id="row14956184342315"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p929893462515"><a name="p929893462515"></a><a name="p929893462515"></a>1.8.1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p688120581238"><a name="p688120581238"></a><a name="p688120581238"></a>1.8.1.rc2</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p198819587236"><a name="p198819587236"></a><a name="p198819587236"></a>v1.8.1-3.0.rc2</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p1388175832319"><a name="p1388175832319"></a><a name="p1388175832319"></a>22.0.RC2-1.8.1</p>
</td>
</tr>
<tr id="row1522133210223"><td class="cellrowborder" rowspan="2" valign="top"  headers="mcps1.2.6.1.1 "><p id="p7881205813233"><a name="p7881205813233"></a><a name="p7881205813233"></a>CANN 5.1.RC1</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.2 "><p id="p1610094112517"><a name="p1610094112517"></a><a name="p1610094112517"></a>1.5.0</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.3 "><p id="p12881165818235"><a name="p12881165818235"></a><a name="p12881165818235"></a>1.5.0.post5</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.4 "><p id="p14881105892315"><a name="p14881105892315"></a><a name="p14881105892315"></a>v1.5.0-3.0.rc1</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.5 "><p id="p5881115812234"><a name="p5881115812234"></a><a name="p5881115812234"></a>22.0.RC1</p>
</td>
</tr>
<tr id="row182218327224"><td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1 "><p id="p19100441202520"><a name="p19100441202520"></a><a name="p19100441202520"></a>1.8.1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2 "><p id="p688165810235"><a name="p688165810235"></a><a name="p688165810235"></a>1.8.1.rc1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3 "><p id="p6881195818236"><a name="p6881195818236"></a><a name="p6881195818236"></a>v1.8.1-3.0.rc1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4 "><p id="p58817582239"><a name="p58817582239"></a><a name="p58817582239"></a>-</p>
</td>
</tr>
<tr id="row137028392234"><td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.1 "><p id="p208815585236"><a name="p208815585236"></a><a name="p208815585236"></a>CANN 5.0.4</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.2 "><p id="p1289474162511"><a name="p1289474162511"></a><a name="p1289474162511"></a>1.5.0</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.3 "><p id="p68812058102318"><a name="p68812058102318"></a><a name="p68812058102318"></a>1.5.0.post4</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.4 "><p id="p17881658122311"><a name="p17881658122311"></a><a name="p17881658122311"></a>2.0.4.tr5</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.5 "><p id="p1588119588233"><a name="p1588119588233"></a><a name="p1588119588233"></a>21.0.4</p>
</td>
</tr>
<tr id="row182227322224"><td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.1 "><p id="p1188113584232"><a name="p1188113584232"></a><a name="p1188113584232"></a>CANN 5.0.3</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.2 "><p id="p1389417414253"><a name="p1389417414253"></a><a name="p1389417414253"></a>1.8.1</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.3 "><p id="p14881358112314"><a name="p14881358112314"></a><a name="p14881358112314"></a>1.5.0.post3</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.4 "><p id="p9881205862316"><a name="p9881205862316"></a><a name="p9881205862316"></a>2.0.3.tr5</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.5 "><p id="p68811858122317"><a name="p68811858122317"></a><a name="p68811858122317"></a>21.0.3</p>
</td>
</tr>
<tr id="row17222203202219"><td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.1 "><p id="p68814580234"><a name="p68814580234"></a><a name="p68814580234"></a>CANN 5.0.2</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.2 "><p id="p610954522514"><a name="p610954522514"></a><a name="p610954522514"></a>1.5.0</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.3 "><p id="p1882758112318"><a name="p1882758112318"></a><a name="p1882758112318"></a>1.5.0.post2</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.4 "><p id="p988215589234"><a name="p988215589234"></a><a name="p988215589234"></a>2.0.2.tr5</p>
</td>
<td class="cellrowborder" valign="top"  headers="mcps1.2.6.1.5 "><p id="p6882115811231"><a name="p6882115811231"></a><a name="p6882115811231"></a>21.0.2</p>
</td>
</tr>
</tbody>
</table>
## 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交[gitee Issues](https://gitee.com/Ascend/pytorch/issues)，我们会尽快回复。感谢您的支持。

## 分支维护策略

AscendPyTorch版本分支的维护阶段如下：

| **状态**            | **时间** | **说明**                                         |
| ------------------- | -------- | ------------------------------------------------ |
| 计划                | 1—3 个月 | 计划特性                                         |
| 开发                | 3个月    | 开发特性                                         |
| 维护                | 6—12个月 | 合入所有已解决的问题并发布版本                   |
| 无维护              | 0—3 个月 | 合入所有已解决的问题，无专职维护人员，无版本发布 |
| 生命周期终止（EOL） | N/A      | 分支不再接受任何修改                             |

## 现有分支的维护状态

| **分支**     | **状态** | **发布日期** | **后续状态**               | **EOL日期** |
| ------------ | -------- | ------------ | -------------------------- | ----------- |
| **v2.0.2**   | EOL      | 2021/7/29    | N/A                        |             |
| **v2.0.3**   | EOL      | 2021/10/15   | N/A                        |             |
| **v2.0.4**   | EOL      | 2022/1/15    | N/A                        |             |
| **v3.0.rc1** | EOL      | 2022/4/10    | N/A                        |             |
| **v3.0.rc2** | EOL      | 2022/7/15    | N/A                        |             |
| **v3.0.rc3** | 维护     | 2022/10/20   | 预计2023/10/20起无维护     |             |
| **v3.0.0**   | 维护     | 2023/1/18    | 预计2024/1/18起无维护      |             |
| **v5.0.rc1** | 维护     | 2023/4/19    | 预计2024/4/19起无维护      |             |
| **v5.0.rc2** | 维护     | 2023/7/19    | 预计2024/7/19起无维护      |             |
| **v5.0.rc3** | 维护     | 2023/10/15   | 预计2024/10/15起无维护     |             |

## 参考文档

有关安装指南、模型迁移和训练/推理教程和API列表等更多详细信息，请参考[昇腾社区PyTorch Ascend Adapter](https://www.hiascend.com/software/ai-frameworks/commercial)。

| 文档名称                   | 文档链接                                                     |
| -------------------------- | ------------------------------------------------------------ |
| AscendPyTorch 安装指南           | [参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/envdeployment/instg/instg_0083.html) |
| AscendPyTorch 网络模型迁移和训练 | [参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/modeldevpt/ptmigr/AImpug_0002.html) |
| AscendPyTorch 在线推理           | [参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/modeldevpt/ptonlineinfer/PyTorch_Infer_000001.html) |
| AscendPyTorch 算子适配           | [参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/tbeaicpudevg/atlasopdev_10_0086.html) |
| AscendPyTorch API清单（PyTorch原生接口与自定义接口）            | [参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/modeldevpt/ptmigr/ptaoplist_001.html) |

## 许可证

PyTorch Ascend Adapter插件使用BSD许可证。详见[LICENSE](LICENSE)文件。
