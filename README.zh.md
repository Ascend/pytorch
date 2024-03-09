# PyTorch Adapter插件

## 简介

本项目开发了名为**torch_npu**的**PyTorch Adapter**插件，使昇腾NPU可以适配PyTorch框架，为使用PyTorch框架的开发者提供昇腾AI处理器的超强算力。

昇腾为基于华为昇腾处理器和软件的行业应用及服务提供全栈AI计算基础设施。您可以通过访问[昇腾社区](https://www.hiascend.com/zh/)，了解关于昇腾的更多信息。

## 安装

### 使用二进制文件进行安装

我们为用户提供可以快速安装**torch_npu**的whl安装包。在安装**torch_npu**之前，您需要先安装**CANN**软件。[昇腾辅助软件](#昇腾辅助软件)中有更多关于CANN的版本信息。请参考[CANN安装指南](https://www.hiascend.com/zh/software/cann/community)获取**CANN**安装包。

1. **安装PyTorch**

通过 pip 安装 PyTorch。

**aarch64:**

```Python
pip3 install torch==1.11.0
```

**x86:**

```Python
pip3 install torch==1.11.0+cpu  --index-url https://download.pytorch.org/whl/cpu
```

若使用pip命令安装失败，请使用下载链接或进入[PyTorch官方网站](https://pytorch.org/)进行查询下载对应版本。

| 架构    | Python版本 | 下载链接                                                     |
| ------- | ---------- | ------------------------------------------------------------ |
| x86     | Python3.7  | [下载链接](https://download.pytorch.org/whl/cpu/torch-1.11.0%2Bcpu-cp37-cp37m-linux_x86_64.whl#sha256=50008b82004b9d91e036cc199a57f863b6f8978b8a222176f9a4435fce181dd8) |
| x86     | Python3.8  | [下载链接](https://download.pytorch.org/whl/cpu/torch-1.11.0%2Bcpu-cp38-cp38-linux_x86_64.whl#sha256=22997df8f3a3f9faed40ef9e7964d1869cafa0317cc4a5b115bfdf69323e8884) |
| x86     | Python3.9  | [下载链接](https://download.pytorch.org/whl/cpu/torch-1.11.0%2Bcpu-cp39-cp39-linux_x86_64.whl#sha256=544c13ef120531ec2f28a3c858c06e600d514a6dfe09b4dd6fd0262088dd2fa3) |
| x86     | Python3.10 | [下载链接](https://download.pytorch.org/whl/cpu/torch-1.11.0%2Bcpu-cp310-cp310-linux_x86_64.whl#sha256=32fa00d974707c0183bc4dd0c1d69e853d0f15cc60f157b71ac5718847808943) |
| aarch64 | Python3.7  | [下载链接](https://download.pytorch.org/whl/torch-1.11.0-cp37-cp37m-manylinux2014_aarch64.whl) |
| aarch64 | Python3.8  | [下载链接](https://download.pytorch.org/whl/torch-1.11.0-cp38-cp38-manylinux2014_aarch64.whl) |
| aarch64 | Python3.9  | [下载链接](https://download.pytorch.org/whl/torch-1.11.0-cp39-cp39-manylinux2014_aarch64.whl) |
| aarch64 | Python3.10 | [下载链接](https://download.pytorch.org/whl/torch-1.11.0-cp310-cp310-manylinux2014_aarch64.whl) |

2. **安装torch_npu依赖**

运行以下命令安装依赖。

```Python
pip3 install pyyaml
pip3 install setuptools
```

3. **安装torch_npu**

以下使用AArch64和Python 3.8 为例。


```Python
wget https://gitee.com/ascend/pytorch/releases/download/v5.0.0-pytorch1.11.0/torch_npu-1.11.0.post8-cp38-cp38-linux_aarch64.whl

pip3 install torch_npu-1.11.0.post8-cp38-cp38-linux_aarch64.whl
```
如需要保存安装日志，可在pip3 install命令后面加上参数 `--log <PATH>`，并对您指定的目录`<PATH>`做好权限管控。

### 使用源代码进行安装

某些特殊场景下，用户可能需要自行编译**torch_npu**。可以根据[昇腾辅助软件表](#昇腾辅助软件)和[PyTorch与Python版本配套表](#PyTorch与Python版本配套表)选择合适的分支。推荐使用Docker镜像编译**torch_npu**，可以通过以下步骤获取(建议只挂载工作路径，并避开系统路径，以降低安全风险)：

1. **克隆torch_npu代码仓**

   ```
   git clone https://gitee.com/ascend/pytorch.git -b v1.11.0 --depth 1
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

| PyTorch版本   | Python版本                                                   |
| ------------- | :----------------------------------------------------------- |
| PyTorch1.11.0 | Python3.7.x(>=3.7.5), Python3.8.x, Python3.9.x, Python3.10.x |
| PyTorch2.0.1  | Python3.8.x, Python3.9.x, Python3.10.x                       |
| PyTorch2.1.0  | Python3.8.x, Python3.9.x, Python3.10.x                       |

## 昇腾辅助软件

**PyTorch Adapter**版本号采用`{PyTorch版本}-{昇腾版本}`命名规则，前者为**PyTorch Adapter**匹配的PyTorch版本，后者用于匹配CANN版本，详细匹配如下：
| CANN版本         | 支持的PyTorch版本 | 支持的Adapter版本 | Gitee分支          | AscendHub镜像版本/名称([链接](https://ascendhub.huawei.com/#/detail/pytorch-modelzoo)) |
|----------------|--------------|-------------------|-------------------|----------------------|
| CANN 7.0.0     | 2.1.0        | 2.1.0             | v2.1.0-5.0.0      | -                    |
|                | 2.0.1        | 2.0.1.post1       | v2.0.1-5.0.0      | -                    |
|                | 1.11.0       | 1.11.0.post8      | v1.11.0-5.0.0     | -                    |
| CANN 7.0.RC1   | 2.1.0        | 2.1.0.rc1         | v2.1.0-5.0.rc3    | -                    |
|                | 2.0.1        | 2.0.1             | v2.0.1-5.0.rc3    | -                    |
|                | 1.11.0       | 1.11.0.post4      | v1.11.0-5.0.rc3   | -                    |
| CANN 6.3.RC3.1 | 1.11.0       | 1.11.0.post3      | v1.11.0-5.0.rc2.2 | -                    |
| CANN 6.3.RC3   | 1.11.0       | 1.11.0.post2      | v1.11.0-5.0.rc2.1 | -                    |
| CANN 6.3.RC2   | 2.0.1        | 2.0.1.rc1         | v2.0.1-5.0.rc2    | -                    |
|                | 1.11.0       | 1.11.0.post1      | v1.11.0-5.0.rc2   | 23.0.RC1-1.11.0      |
|                | 1.8.1        | 1.8.1.post2       | v1.8.1-5.0.rc2    | 23.0.RC1-1.8.1       |
| CANN 6.3.RC1   | 1.11.0       | 1.11.0            | v1.11.0-5.0.rc1   | -                    |
|                | 1.8.1        | 1.8.1.post1       | v1.8.1-5.0.rc1    | -                    |
| CANN 6.0.1     | 1.5.0        | 1.5.0.post8       | v1.5.0-3.0.0      | 22.0.0               |
|                | 1.8.1        | 1.8.1             | v1.8.1-3.0.0      | 22.0.0-1.8.1         |
|                | 1.11.0       | 1.11.0.rc2（beta) | v1.11.0-3.0.0     | -                    |
| CANN 6.0.RC1   | 1.5.0        | 1.5.0.post7       | v1.5.0-3.0.rc3    | 22.0.RC3             |
|                | 1.8.1        | 1.8.1.rc3         | v1.8.1-3.0.rc3    | 22.0.RC3-1.8.1       |
|                | 1.11.0       | 1.11.0.rc1（beta) | v1.11.0-3.0.rc3   | -                    |
| CANN 5.1.RC2   | 1.5.0        | 1.5.0.post6       | v1.5.0-3.0.rc2    | 22.0.RC2             |
|                | 1.8.1        | 1.8.1.rc2         | v1.8.1-3.0.rc2    | 22.0.RC2-1.8.1       |
| CANN 5.1.RC1   | 1.5.0        | 1.5.0.post5       | v1.5.0-3.0.rc1    | 22.0.RC1             |
|                | 1.8.1        | 1.8.1.rc1         | v1.8.1-3.0.rc1    | -                    |
| CANN 5.0.4     | 1.5.0        | 1.5.0.post4       | 2.0.4.tr5         | 21.0.4               |
| CANN 5.0.3     | 1.8.1        | 1.5.0.post3       | 2.0.3.tr5         | 21.0.3               |
| CANN 5.0.2     | 1.5.0        | 1.5.0.post2       | 2.0.2.tr5         | 21.0.2               |

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
| 生命周期终止（EOL）  | N/A      | 分支不再接受任何修改                             |


## PyTorch版本维护策略

| **PyTorch版本** | **维护策略** | **当前状态** | **发布时间** | **后续状态** | **EOL日期** |
|-----------|-----------|--------|------------|-----------------------|-----------|
| 2.1.0     |  长期支持  | 维护   | 2023/10/15 | 预计2024/10/15起无维护 |           |
| 2.0.1     |  常规版本  | 维护   | 2023/7/19  | 预计2024/1/19起无维护  |           |
| 1.11.0    |  长期支持  | 维护   | 2023/4/19  | 预计2024/4/19起无维护  |           |
| 1.8.1     |  长期支持  | EOL    | 2022/4/10  |                       | 2023/4/10 |
| 1.5.0     |  长期支持  | EOL    | 2021/7/29  |                       | 2022/7/29 |

## 安全声明

[Ascend Adapter for PyTorch插件 安全声明](SECURITYNOTE.md)

## 参考文档

有关安装指南、模型迁移和训练/推理教程和API列表等更多详细信息，请参考[昇腾社区PyTorch Ascend Adapter](https://www.hiascend.com/software/ai-frameworks/commercial)。

| 文档名称                   | 文档链接                                                     |
| -------------------------- | ------------------------------------------------------------ |
| AscendPyTorch 安装指南           | [参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/700/envdeployment/instg/instg_0163.html) |
| AscendPyTorch 网络模型迁移和训练 | [参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/AImpug_000002.html) |
| AscendPyTorch 在线推理           | [参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptonlineinfer/PyTorch_Infer_000001.html) |
| AscendPyTorch 算子适配           | [参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/700/operatordev/tbeaicpudevg/atlasopdev_10_0086.html) |
| AscendPyTorch API清单（PyTorch原生接口与自定义接口）            | [参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/ptaoplist_000002.html) |

## 许可证

PyTorch Ascend Adapter插件使用BSD许可证。详见[LICENSE](LICENSE)文件。
