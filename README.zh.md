# Ascend Extension for PyTorch插件


## 简介

本项目开发了名为**torch_npu**的**Ascend Extension for PyTorch**插件，使昇腾NPU可以适配PyTorch框架，为使用PyTorch框架的开发者提供昇腾AI处理器的超强算力。

昇腾为基于华为昇腾处理器和软件的行业应用及服务提供全栈AI计算基础设施。您可以通过访问[昇腾社区](https://www.hiascend.com/zh/)，了解关于昇腾的更多信息。

## 版本说明

### PyTorch与Python版本配套表

| PyTorch版本     | Python版本                                                     |
|---------------|:-------------------------------------------------------------|
| PyTorch1.11.0 | Python3.7.x(>=3.7.5), Python3.8.x, Python3.9.x, Python3.10.x |
| PyTorch2.1.0  | Python3.8.x, Python3.9.x, Python3.10.x, Python 3.11.x        |
| PyTorch2.2.0  | Python3.8.x, Python3.9.x, Python3.10.x                       |
| PyTorch2.3.1  | Python3.8.x, Python3.9.x, Python3.10.x, Python 3.11.x        |
| PyTorch2.4.0  | Python3.8.x, Python3.9.x, Python3.10.x, Python 3.11.x        |
| PyTorch2.5.1  | Python3.9.x, Python3.10.x, Python 3.11.x                     |
| PyTorch2.6.0  | Python3.9.x, Python3.10.x, Python 3.11.x                     |
| PyTorch2.7.1  | Python3.9.x, Python3.10.x, Python 3.11.x                     |
| PyTorch2.8.0  | Python3.9.x, Python3.10.x, Python 3.11.x                     |
| PyTorch2.9.0  | Python3.10.x, Python 3.11.x                                  |



### 昇腾辅助软件

**Ascend Extension for PyTorch**的分支名称采用`{PyTorch版本}-{昇腾版本}`命名规则，前者为**Ascend Extension for PyTorch**匹配的PyTorch版本，后者为**Ascend Extension for PyTorch**版本号，详细匹配如下：

| CANN版本                | 支持的PyTorch版本 | 支持的Extension版本   | GitCode分支         | 
|-----------------------|--------------|------------------|-------------------|
| CANN 8.5.0            | 2.9.0        | 2.9.0            | v2.9.0-7.3.0      |
|                       | 2.8.0        | 2.8.0.post2      | v2.8.0-7.3.0      |
|                       | 2.7.1        | 2.7.1.post2      | v2.7.1-7.3.0      |
|                       | 2.6.0        | 2.6.0.post5      | v2.6.0-7.3.0      |
| CANN 8.3.RC1          | 2.8.0        | 2.8.0            | v2.8.0-7.2.0      |
|                       | 2.7.1        | 2.7.1            | v2.7.1-7.2.0      |
|                       | 2.6.0        | 2.6.0.post3      | v2.6.0-7.2.0      |
|                       | 2.1.0        | 2.1.0.post17     | v2.1.0-7.2.0      |
| CANN 8.2.RC1          | 2.6.0        | 2.6.0            | v2.6.0-7.1.0      |
|                       | 2.5.1        | 2.5.1.post1      | v2.5.1-7.1.0      |
|                       | 2.1.0        | 2.1.0.post13     | v2.1.0-7.1.0      |
| CANN 8.1.RC1          | 2.5.1        | 2.5.1            | v2.5.1-7.0.0      |
|                       | 2.4.0        | 2.4.0.post4      | v2.4.0-7.0.0      |
|                       | 2.3.1        | 2.3.1.post6      | v2.3.1-7.0.0      |
|                       | 2.1.0        | 2.1.0.post12     | v2.1.0-7.0.0      |
| CANN 8.0.0            | 2.4.0        | 2.4.0.post2      | v2.4.0-6.0.0      |
|                       | 2.3.1        | 2.3.1.post4      | v2.3.1-6.0.0      |
|                       | 2.1.0        | 2.1.0.post10     | v2.1.0-6.0.0      |
| CANN 8.0.RC3          | 2.4.0        | 2.4.0            | v2.4.0-6.0.rc3    |
|                       | 2.3.1        | 2.3.1.post2      | v2.3.1-6.0.rc3    |
|                       | 2.1.0        | 2.1.0.post8      | v2.1.0-6.0.rc3    | 
| CANN 8.0.RC2          | 2.3.1        | 2.3.1            | v2.3.1-6.0.rc2    | 
|                       | 2.2.0        | 2.2.0.post2      | v2.2.0-6.0.rc2    |
|                       | 2.1.0        | 2.1.0.post6      | v2.1.0-6.0.rc2    |
|                       | 1.11.0       | 1.11.0.post14    | v1.11.0-6.0.rc2   |
| CANN 8.0.RC1          | 2.2.0        | 2.2.0            | v2.2.0-6.0.rc1    |
|                       | 2.1.0        | 2.1.0.post4      | v2.1.0-6.0.rc1    | 
|                       | 1.11.0       | 1.11.0.post11    | v1.11.0-6.0.rc1   | 
| CANN 7.0.0            | 2.1.0        | 2.1.0            | v2.1.0-5.0.0      |
|                       | 2.0.1        | 2.0.1.post1      | v2.0.1-5.0.0      | 
|                       | 1.11.0       | 1.11.0.post8     | v1.11.0-5.0.0     | 
| CANN 7.0.RC1          | 2.1.0        | 2.1.0.rc1        | v2.1.0-5.0.rc3    | 
|                       | 2.0.1        | 2.0.1            | v2.0.1-5.0.rc3    | 
|                       | 1.11.0       | 1.11.0.post4     | v1.11.0-5.0.rc3   | 
| CANN 6.3.RC3.1        | 1.11.0       | 1.11.0.post3     | v1.11.0-5.0.rc2.2 | 
| CANN 6.3.RC3          | 1.11.0       | 1.11.0.post2     | v1.11.0-5.0.rc2.1 | 
| CANN 6.3.RC2          | 2.0.1        | 2.0.1.rc1        | v2.0.1-5.0.rc2    | 
|                       | 1.11.0       | 1.11.0.post1     | v1.11.0-5.0.rc2   |
|                       | 1.8.1        | 1.8.1.post2      | v1.8.1-5.0.rc2    |
| CANN 6.3.RC1          | 1.11.0       | 1.11.0           | v1.11.0-5.0.rc1   | 
|                       | 1.8.1        | 1.8.1.post1      | v1.8.1-5.0.rc1    | 
| CANN 6.0.1            | 1.5.0        | 1.5.0.post8      | v1.5.0-3.0.0      |
|                       | 1.8.1        | 1.8.1            | v1.8.1-3.0.0      |
|                       | 1.11.0       | 1.11.0.rc2（beta) | v1.11.0-3.0.0     | 
| CANN 6.0.RC1          | 1.5.0        | 1.5.0.post7      | v1.5.0-3.0.rc3    |
|                       | 1.8.1        | 1.8.1.rc3        | v1.8.1-3.0.rc3    |
|                       | 1.11.0       | 1.11.0.rc1（beta) | v1.11.0-3.0.rc3   | 
| CANN 5.1.RC2          | 1.5.0        | 1.5.0.post6      | v1.5.0-3.0.rc2    |
|                       | 1.8.1        | 1.8.1.rc2        | v1.8.1-3.0.rc2    |
| CANN 5.1.RC1          | 1.5.0        | 1.5.0.post5      | v1.5.0-3.0.rc1    |
|                       | 1.8.1        | 1.8.1.rc1        | v1.8.1-3.0.rc1    | 
| CANN 5.0.4            | 1.5.0        | 1.5.0.post4      | 2.0.4.tr5         |
| CANN 5.0.3            | 1.8.1        | 1.5.0.post3      | 2.0.3.tr5         |
| CANN 5.0.2            | 1.5.0        | 1.5.0.post2      | 2.0.2.tr5         |


## 快速入门

基于CNN模型识别手写数字的脚本，对在GPU上训练的该脚本代码进行修改，使其可以迁移到昇腾NPU上进行训练。具体操作请参见[Ascend Extension for PyTorch 快速入门](https://www.hiascend.com/document/detail/zh/Pytorch/710/fastexperience/fastexperience_0001.html)。

## 环境部署

### 使用二进制文件进行安装

我们为用户提供可以快速安装**torch_npu**的whl安装包。在安装**torch_npu**之前，您需要先安装**CANN**软件。[昇腾辅助软件](#昇腾辅助软件)中有更多关于CANN的版本信息。请参考[CANN安装指南](https://www.hiascend.com/cann)获取**CANN**安装包。

1. **安装PyTorch**

   通过 pip 安装 PyTorch。

   **aarch64:**

   ```bash
   pip3 install torch==2.9.0
   ```

   **x86:**

   ```bash
   pip3 install torch==2.9.0+cpu  --index-url https://download.pytorch.org/whl/cpu
   ```

   若使用pip命令安装失败，请使用下载链接或进入[PyTorch官方网站](https://pytorch.org/)进行查询下载对应版本。

   | 架构      | Python版本   | 下载链接                                                                                                                                                                          |
   |---------|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   | x86     | Python3.10 | [下载链接](https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=bd2a257e670ede9fc01c6d76dccdc473040913b8e9328169bf177dbdc38e2484)  |
   | x86     | Python3.11 | [下载链接](https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=add3e93ecc1eeaa6853f6a973ce60ffb3cb14ed2e80f5055e139b09385dce0a7)  |
   | aarch64 | Python3.10 | [下载链接](https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl#sha256=b224792ea567b52c7f1ce1d789567f6920e06fd3b339fa1e1b05948845f783ad) |
   | aarch64 | Python3.11 | [下载链接](https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=add3e93ecc1eeaa6853f6a973ce60ffb3cb14ed2e80f5055e139b09385dce0a7)  |

2. **安装torch_npu依赖**

   运行以下命令安装依赖。

   ```bash
   pip3 install pyyaml
   pip3 install setuptools
   ```

3. **安装torch_npu**

   ```bash
   pip3 install torch-npu==2.9.0rc1
   ```
   如需要保存安装日志，可在pip3 install命令后面加上参数 `--log <PATH>`，并对您指定的目录`<PATH>`做好权限管控。

### 使用源代码进行安装

某些特殊场景下，用户可能需要自行编译**torch_npu**。可以根据[昇腾辅助软件表](#昇腾辅助软件)和[PyTorch与Python版本配套表](#pytorch与python版本配套表)选择合适的分支。

推荐使用Docker镜像编译**torch_npu**，可以通过以下步骤获取（建议只挂载工作路径，并避开系统路径，以降低安全风险）, 生成的.whl文件路径为./dist/。

>**须知：**<br>
>如果不使用镜像，编译时请注意gcc版本需要为13.3

1. **克隆torch_npu代码仓**

   ```
   git clone https://gitcode.com/ascend/pytorch.git -b v2.9.0 --depth 1
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

   以**Python 3.10** 为例。

   ```
   cd /home/pytorch
   bash ci/build.sh --python=3.10
   ```

 **提示**
 
   如果想使用新的C++ ABI编译，请首先运行如下命令，此时推荐和社区torch包相同的编译环境：glibc 2.28, gcc 13.3。

   ```
   export _GLIBCXX_USE_CXX11_ABI=1
   ```

   同时，我们支持使用如下变量配置-fabi-version，要求和社区torch包ABI版本一致

   ```
   export _ABI_VERSION=18
   ```

### 安装后验证

#### 前提

运行以下命令初始化**CANN**环境变量。

```Shell
# Default path, change it if needed.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

#### 快速验证

 可以通过以下样例快速体验**昇腾NPU**。

```diff
import torch
- import torch_npu # torch_npu2.5.1及以后版本可以不用手动导包

x = torch.randn(2, 2).npu()
y = torch.randn(2, 2).npu()
z = x.mm(y)

print(z)
```

## 卸载
Pytorch框架训练环境的卸载可以参考[昇腾官方文档](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes/ptes_00032.html)。

torch_npu的卸载只需执行命令：

  ```bash
  pip3 uninstall torch_npu
  ```

如需要保存卸载日志，可在pip3 uninstall命令后面加上参数 `--log <PATH>`，并对您指定的目录`<PATH>`做好权限管控。



## 硬件配套

昇腾训练设备包含以下型号，都可作为PyTorch模型的训练环境。
| 产品系列               | 产品型号                         |
|-----------------------|----------------------------------|
| Atlas 训练系列产品     | Atlas 800 训练服务器（型号：9000） |
|                       | Atlas 800 训练服务器（型号：9010） |
|                       | Atlas 900 PoD（型号：9000）       |
|                       | Atlas 300T 训练卡（型号：9000）    |
|                       | Atlas 300T Pro 训练卡（型号：9000）|
| Atlas A2 训练系列产品  | Atlas 800T A2 训练服务器          |
|                       | Atlas 900 A2 PoD 集群基础单元     |
|                       | Atlas 200T A2 Box16 异构子框      |
| Atlas A3 训练系列产品  | Atlas 800T A3 训练服务器          |
|                       | Atlas 900 A3 SuperPoD 超节点     |

昇腾推理设备包含以下型号，都可作为大模型的推理环境。
| 产品系列               | 产品型号                         |
|-----------------------|----------------------------------|
| Atlas 800I A2推理产品  | Atlas 800I A2 推理服务器          |

## 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交[GitCode Issues](https://gitcode.com/Ascend/pytorch/issues)，我们会尽快回复。感谢您的支持。

## 分支维护策略

Ascend Extension for PyTorch版本分支的维护阶段如下：


| **状态**            | **时间** | **说明**                                         |
| ------------------- | -------- | ------------------------------------------------ |
| 计划                | 1—3 个月 | 计划特性                                         |
| 开发                | 6—12 个月   | 开发新特性并修复问题，定期发布新版本。针对不同的PyTorch版本采取不同的策略，常规分支的开发周期分别为6个月，长期支持分支的开发周期为12个月 |
| 维护                |  1年/3.5年 | 常规分支维护1年,长期支持分支维护3.5年。对重大BUG进行修复，不合入新特性，并视BUG的影响发布补丁版本 |
| 生命周期终止（EOL） | N/A      | 分支不再接受任何修改                             |

## PyTorch版本维护策略

| **PyTorch版本** | **维护策略** | **当前状态** | **发布时间**   | **后续状态**             | **EOL日期** |
|---------------|----------|----------|------------|----------------------|-----------|
| 2.9.0         | 常规分支     | 开发       | 2025/10/15| 预计2026/03/15起进入维护状态        | -          | 
| 2.8.0         | 常规分支     | 开发       | 2025/10/15| 预计2026/03/15起进入维护状态        | -          | 
| 2.7.1         | 长期分支     | 开发       |  2025/10/15| 预计2026/10/15起进入维护状态       | -          | 
| 2.6.0         | 常规分支     | 开发       | 2025/07/25 | 预计2026/01/25起进入维护状态       | -          | 
| 2.5.1         | 常规分支     | 维护       | 2024/11/08 | 预计2026/08/08起进入无维护状态     | -          | 
| 2.4.0         | 常规分支     | 维护       | 2024/10/15 | 预计2026/06/15起进入无维护状态     | -          | 
| 2.3.1         | 常规分支     | 维护       | 2024/06/06 | 预计2026/06/07起进入无维护状态     |            |
| 2.2.0         | 常规分支     | EOL        | 2024/04/01 |                                  | 2025/10/14 |
| 2.1.0         | 长期支持     | 维护       | 2023/10/15 | 预计2026/12/30起进入无维护状态     |            |
| 2.0.1         | 常规分支     | EOL        | 2023/7/19  |                                  | 2024/3/14  |
| 1.11.0        | 长期支持     | EOL        | 2023/4/19  |                                  | 2025/10/25 |
| 1.8.1         | 长期支持     | EOL        | 2022/4/10  |                                  | 2023/4/10 |
| 1.5.0         | 长期支持     | EOL        | 2021/7/29  |                                  | 2022/7/29 |

## 安全声明

[Ascend Extension for PyTorch插件 安全声明](./SECURITYNOTE.md)

## 参考文档

有关安装指南、模型迁移和训练/推理教程和API列表等更多详细信息，请参考[昇腾社区Ascend Extension for PyTorch](https://www.hiascend.com/software/ai-frameworks?framework=pytorch)。

| 文档名称                   | 文档链接                                                     |
| -------------------------- | ------------------------------------------------------------ |
| 软件安装指南           | [参考链接](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html) |
| 网络模型迁移和训练 | [参考链接](https://www.hiascend.com/document/detail/zh/Pytorch/720/ptmoddevg/trainingmigrguide/PT_LMTMOG_0002.html) |
| 算子适配           | [参考链接](https://www.hiascend.com/document/detail/zh/Pytorch/720/ptmoddevg/Frameworkfeatures/featuresguide_00021.html) |
| PyTorch原生接口清单          | [参考链接](https://www.hiascend.com/document/detail/zh/Pytorch/720/apiref/PyTorchNativeapi/ptaoplist_000003.html) |
| Ascend Extension for PyTorch自定义API参考          | [参考链接](https://www.hiascend.com/document/detail/zh/Pytorch/720/apiref/torchnpuCustomsapi/context/%E6%A6%82%E8%BF%B0.md) |


## 许可证

Ascend Extension for PyTorch插件使用BSD许可证。详见[LICENSE](LICENSE)文件。
