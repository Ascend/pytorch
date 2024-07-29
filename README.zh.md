# Ascend Extension for PyTorch插件

## 简介

本项目开发了名为**torch_npu**的**Ascend Extension for PyTorch**插件，使昇腾NPU可以适配PyTorch框架，为使用PyTorch框架的开发者提供昇腾AI处理器的超强算力。

昇腾为基于华为昇腾处理器和软件的行业应用及服务提供全栈AI计算基础设施。您可以通过访问[昇腾社区](https://www.hiascend.com/zh/)，了解关于昇腾的更多信息。

## 安装

### 使用二进制文件进行安装

我们为用户提供可以快速安装**torch_npu**的whl安装包。在安装**torch_npu**之前，您需要先安装**CANN**软件。[昇腾辅助软件](#昇腾辅助软件)中有更多关于CANN的版本信息。请参考[CANN安装指南](https://www.hiascend.com/zh/software/cann/community)获取**CANN**安装包。

1. **安装PyTorch**

通过 pip 安装 PyTorch。

**aarch64:**

```Python
pip3 install torch==2.4.0
```

**x86:**

```Python
pip3 install torch==2.4.0+cpu  --index-url https://download.pytorch.org/whl/cpu
```

若使用pip命令安装失败，请使用下载链接或进入[PyTorch官方网站](https://pytorch.org/)进行查询下载对应版本。

| 架构      | Python版本   | 下载链接                                                                                                                                                                                          |
|---------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| x86     | Python3.8  | [下载链接](https://download.pytorch.org/whl/cpu/torch-2.4.0%2Bcpu-cp38-cp38-linux_x86_64.whl#sha256=08753c3d776ae49dc9ddbae02e26720a513a4dc7997e41d95392bca71623a0cd)                             |
| x86     | Python3.9  | [下载链接](https://download.pytorch.org/whl/cpu/torch-2.4.0%2Bcpu-cp39-cp39-linux_x86_64.whl#sha256=040abaee8affa1bb0f3ca14ca693ba81d0d90d88df5b8a839af96933a7fa2d29)                             |
| x86     | Python3.10 | [下载链接](https://download.pytorch.org/whl/cpu/torch-2.4.0%2Bcpu-cp310-cp310-linux_x86_64.whl#sha256=0e59377b27823dda6d26528febb7ca06fc5b77816eaa58b4420cc8785e33d4ce)                           |
| aarch64 | Python3.8  | [下载链接](https://download.pytorch.org/whl/cpu/torch-2.4.0-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl#sha256=9eba83f8a8f98542f917e39000c903f154655acf6375c073cfcd4306a154eb80)   |
| aarch64 | Python3.9  | [下载链接](https://download.pytorch.org/whl/cpu/torch-2.4.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl#sha256=2786a47c8d8dec176fc679d2aab9a6f549c25452510b49650ab134135266ba33)   |
| aarch64 | Python3.10 | [下载链接](https://download.pytorch.org/whl/cpu/torch-2.4.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl#sha256=7c159e89d4ecf08403f9d1373d554422240b9b1146a0a19129069dc357a72b2b) |

2. **安装torch_npu依赖**

运行以下命令安装依赖。

```Python
pip3 install pyyaml
pip3 install setuptools
```

3. **安装torch_npu**

```
pip3 install torch-npu==2.4.0rc1
```
如需要保存安装日志，可在pip3 install命令后面加上参数 `--log <PATH>`，并对您指定的目录`<PATH>`做好权限管控。

### 使用源代码进行安装

某些特殊场景下，用户可能需要自行编译**torch_npu**。可以根据[昇腾辅助软件表](#昇腾辅助软件)和[PyTorch与Python版本配套表](#PyTorch与Python版本配套表)选择合适的分支。推荐使用Docker镜像编译**torch_npu**，可以通过以下步骤获取(建议只挂载工作路径，并避开系统路径，以降低安全风险), 生成的.whl文件路径为./dist/：

1. **克隆torch_npu代码仓**

   ```
   git clone https://gitee.com/ascend/pytorch.git -b v2.4.0 --depth 1
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


## 卸载
Pytorch框架训练环境的卸载可以参考[昇腾官方文档](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes/ptes_00032.html)。

torch_npu的卸载只需执行命令：

  ```
  pip3 uninstall torch_npu
  ```

如需要保存卸载日志，可在pip3 uninstall命令后面加上参数 `--log <PATH>`，并对您指定的目录`<PATH>`做好权限管控。

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

| PyTorch版本     | Python版本                                                     |
|---------------|:-------------------------------------------------------------|
| PyTorch1.11.0 | Python3.7.x(>=3.7.5), Python3.8.x, Python3.9.x, Python3.10.x |
| PyTorch2.1.0  | Python3.8.x, Python3.9.x, Python3.10.x                       |
| PyTorch2.2.0  | Python3.8.x, Python3.9.x, Python3.10.x                       |
| PyTorch2.3.1  | Python3.8.x, Python3.9.x, Python3.10.x                       |
| PyTorch2.4.0  | Python3.8.x, Python3.9.x, Python3.10.x                       |

## 昇腾辅助软件

**PyTorch Extension**版本号采用`{PyTorch版本}-{昇腾版本}`命名规则，前者为**PyTorch Extension**匹配的PyTorch版本，后者用于匹配CANN版本，详细匹配如下：

| CANN版本                | 支持的PyTorch版本 | 支持的Extension版本   | Gitee分支           | AscendHub镜像版本/名称([链接](https://ascendhub.huawei.com/#/detail/pytorch-modelzoo)) |
|-----------------------|--------------|------------------|-------------------|--------------------------------------------------------------------------------|
| CANN 8.0.RC3.alpha001 | 2.4.0        | 2.4.0rc1         | v2.4.0            | -                                                                              |
| CANN 8.0.RC2          | 2.3.1        | 2.3.1            | v2.3.1-6.0.rc2    | -                                                                              |
|                       | 2.2.0        | 2.2.0.post2      | v2.2.0-6.0.rc2    | -                                                                              |
|                       | 2.1.0        | 2.1.0.post6      | v2.1.0-6.0.rc2    | -                                                                              |
|                       | 1.11.0       | 1.11.0.post14    | v1.11.0-6.0.rc2   | -                                                                              |
| CANN 8.0.RC1          | 2.2.0        | 2.2.0            | v2.2.0-6.0.rc1    | -                                                                              |
|                       | 2.1.0        | 2.1.0.post3      | v2.1.0-6.0.rc1    | -                                                                              |
|                       | 1.11.0       | 1.11.0.post11    | v1.11.0-6.0.rc1   | -                                                                              |
| CANN 7.0.0            | 2.1.0        | 2.1.0            | v2.1.0-5.0.0      | -                                                                              |
|                       | 2.0.1        | 2.0.1.post1      | v2.0.1-5.0.0      | -                                                                              |
|                       | 1.11.0       | 1.11.0.post8     | v1.11.0-5.0.0     | -                                                                              |
| CANN 7.0.RC1          | 2.1.0        | 2.1.0.rc1        | v2.1.0-5.0.rc3    | -                                                                              |
|                       | 2.0.1        | 2.0.1            | v2.0.1-5.0.rc3    | -                                                                              |
|                       | 1.11.0       | 1.11.0.post4     | v1.11.0-5.0.rc3   | -                                                                              |
| CANN 6.3.RC3.1        | 1.11.0       | 1.11.0.post3     | v1.11.0-5.0.rc2.2 | -                                                                              |
| CANN 6.3.RC3          | 1.11.0       | 1.11.0.post2     | v1.11.0-5.0.rc2.1 | -                                                                              |
| CANN 6.3.RC2          | 2.0.1        | 2.0.1.rc1        | v2.0.1-5.0.rc2    | -                                                                              |
|                       | 1.11.0       | 1.11.0.post1     | v1.11.0-5.0.rc2   | 23.0.RC1-1.11.0                                                                |
|                       | 1.8.1        | 1.8.1.post2      | v1.8.1-5.0.rc2    | 23.0.RC1-1.8.1                                                                 |
| CANN 6.3.RC1          | 1.11.0       | 1.11.0           | v1.11.0-5.0.rc1   | -                                                                              |
|                       | 1.8.1        | 1.8.1.post1      | v1.8.1-5.0.rc1    | -                                                                              |
| CANN 6.0.1            | 1.5.0        | 1.5.0.post8      | v1.5.0-3.0.0      | 22.0.0                                                                         |
|                       | 1.8.1        | 1.8.1            | v1.8.1-3.0.0      | 22.0.0-1.8.1                                                                   |
|                       | 1.11.0       | 1.11.0.rc2（beta) | v1.11.0-3.0.0     | -                                                                              |
| CANN 6.0.RC1          | 1.5.0        | 1.5.0.post7      | v1.5.0-3.0.rc3    | 22.0.RC3                                                                       |
|                       | 1.8.1        | 1.8.1.rc3        | v1.8.1-3.0.rc3    | 22.0.RC3-1.8.1                                                                 |
|                       | 1.11.0       | 1.11.0.rc1（beta) | v1.11.0-3.0.rc3   | -                                                                              |
| CANN 5.1.RC2          | 1.5.0        | 1.5.0.post6      | v1.5.0-3.0.rc2    | 22.0.RC2                                                                       |
|                       | 1.8.1        | 1.8.1.rc2        | v1.8.1-3.0.rc2    | 22.0.RC2-1.8.1                                                                 |
| CANN 5.1.RC1          | 1.5.0        | 1.5.0.post5      | v1.5.0-3.0.rc1    | 22.0.RC1                                                                       |
|                       | 1.8.1        | 1.8.1.rc1        | v1.8.1-3.0.rc1    | -                                                                              |
| CANN 5.0.4            | 1.5.0        | 1.5.0.post4      | 2.0.4.tr5         | 21.0.4                                                                         |
| CANN 5.0.3            | 1.8.1        | 1.5.0.post3      | 2.0.3.tr5         | 21.0.3                                                                         |
| CANN 5.0.2            | 1.5.0        | 1.5.0.post2      | 2.0.2.tr5         | 21.0.2                                                                         |

## 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交[gitee Issues](https://gitee.com/Ascend/pytorch/issues)，我们会尽快回复。感谢您的支持。

## 分支维护策略

AscendPyTorch版本分支的维护阶段如下：

| **状态**            | **时间** | **说明**                                         |
| ------------------- | -------- | ------------------------------------------------ |
| 计划                | 1—3 个月 | 计划特性                                         |
| 开发                | 3 个月   | 开发特性                                         |
| 维护                | 6-12 个月| 合入所有已解决的问题并发布版本，针对不同的PyTorch版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护              | 0—3 个月 | 合入所有已解决的问题，无专职维护人员，无版本发布 |
| 生命周期终止（EOL） | N/A      | 分支不再接受任何修改                             |

## PyTorch版本维护策略

| **PyTorch版本** | **维护策略** | **当前状态** | **发布时间**   | **后续状态**         | **EOL日期** |
|---------------|----------|----------|------------|------------------|-----------|
| 2.4.0         | 常规版本     | 维护       | 2024/07/26 | 预计2025/01/26起无维护 |           |
| 2.3.1         | 常规版本     | 维护       | 2024/06/06 | 预计2024/12/06起无维护 |           |
| 2.2.0         | 常规版本     | 维护       | 2024/04/01 | 预计2024/10/15起无维护 |           |
| 2.1.0         | 长期支持     | 维护       | 2023/10/15 | 预计2024/10/15起无维护 |           |
| 2.0.1         | 常规版本     | EOL      | 2023/7/19  |                  | 2024/3/14 |
| 1.11.0        | 长期支持     | 维护       | 2023/4/19  | 预计2024/4/19起无维护  |           |
| 1.8.1         | 长期支持     | EOL      | 2022/4/10  |                  | 2023/4/10 |
| 1.5.0         | 长期支持     | EOL      | 2021/7/29  |                  | 2022/7/29 |

## 安全声明

[Ascend Extension for PyTorch插件 安全声明](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md)

## 参考文档

有关安装指南、模型迁移和训练/推理教程和API列表等更多详细信息，请参考[昇腾社区Ascend Extension for PyTorch](https://www.hiascend.com/software/ai-frameworks?framework=pytorch)。

| 文档名称                   | 文档链接                                                     |
| -------------------------- | ------------------------------------------------------------ |
| 安装指南           | [参考链接](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/configandinstg/instg/insg_0001.html) |
| 网络模型迁移和训练 | [参考链接](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/ptmoddevg/trainingmigrguide/PT_LMTMOG_0003.html) |
| 算子适配           | [参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0048.html) |
| API清单（PyTorch原生接口与自定义接口）            | [参考链接](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000002.html) |

## 许可证

Ascend Extension for PyTorch插件使用BSD许可证。详见[LICENSE](LICENSE)文件。
