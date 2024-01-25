# PyTorch Adapter

## Overview

This repository develops the **PyTorch Adapter** named **torch_npu** to adapt **Ascend NPU** to **PyTorch** so that developers who use the **PyTorch** can obtain powerful compute capabilities of **Ascend AI Processors**.

Ascend is a full-stack AI computing infrastructure for industry applications and services based on Huawei Ascend processors and software. For more information about Ascend, see [Ascend Community](https://www.hiascend.com/en/).

## Installation

### From Binary

Provide users with wheel package to quickly install **torch_npu**. Before installing **torch_npu**, complete the installation of **CANN** according to [Ascend Auxiliary Software](#ascend-auxiliary-software). To obtain the **CANN** installation package, refer to the [CANN Installation](https://www.hiascend.com/en/software/cann/community).

1. **Install PyTorch**

Install **PyTorch** through pip.

**For Aarch64:**

```Python
pip3 install torch==2.0.1
```

**For x86:**

```Python
pip3 install torch==2.0.1+cpu  --index-url https://download.pytorch.org/whl/cpu
```

2. **Install torch-npu dependencies**

Run the following command to install dependencies.

```Python
pip3 install pyyaml
pip3 install setuptools
```

If the installation fails, use the download link or visit the [PyTorch official website](https://pytorch.org/) to download the installation package of the corresponding version.

| OS arch | Python version | link                                                         |
| ------- | -------------- | ------------------------------------------------------------ |
| x86     | Python3.8      | [link](https://download.pytorch.org/whl/cpu/torch-2.0.1%2Bcpu-cp38-cp38-linux_x86_64.whl#sha256=8046f49deae5a3d219b9f6059a1f478ae321f232e660249355a8bf6dcaa810c1) |
| x86     | Python3.9      | [link](https://download.pytorch.org/whl/cpu/torch-2.0.1%2Bcpu-cp39-cp39-linux_x86_64.whl#sha256=73482a223d577407c45685fde9d2a74ba42f0d8d9f6e1e95c08071dc55c47d7b) |
| x86     | Python3.10     | [link](https://download.pytorch.org/whl/cpu/torch-2.0.1%2Bcpu-cp310-cp310-linux_x86_64.whl#sha256=fec257249ba014c68629a1994b0c6e7356e20e1afc77a87b9941a40e5095285d) |
| aarch64 | Python3.8      | [link](https://download.pytorch.org/whl/torch-2.0.1-cp38-cp38-manylinux2014_aarch64.whl) |
| aarch64 | Python3.9      | [link](https://download.pytorch.org/whl/torch-2.0.1-cp39-cp39-manylinux2014_aarch64.whl) |
| aarch64 | Python3.10     | [link](https://download.pytorch.org/whl/torch-2.0.1-cp310-cp310-manylinux2014_aarch64.whl) |

3. **Install torch-npu**

Take **Aarch64** architecture and **Python 3.8** as an example.

```Python
wget https://gitee.com/ascend/pytorch/releases/download/v5.0.0-pytorch2.0.1/torch_npu-2.0.1.post1-cp38-cp38-linux_aarch64.whl

pip3 install torch_npu-2.0.1.post1-cp38-cp38-linux_aarch64.whl
```

### From Source

In some special scenarios, users may need to compile **torch-npu** by themselves.Select a branch in table [Ascend Auxiliary Software](#ascend-auxiliary-software) and a Python version in table [PyTorch and Python Version Matching Table](#pytorch-and-python-version-matching-table) first. The docker image is recommended for compiling torch-npu through the following steps(It is recommended to mount the working path only and avoid the system path to reduce security risks):

1. **Clone torch-npu**

   ```
   git clone https://github.com/ascend/pytorch.git -b v2.0.1 --depth 1
   ```

2. **Build Docker Image**

   ```
   cd pytorch/ci/docker/{arch} # {arch} for X86 or ARM
   docker build -t manylinux-builder:v1 .
   ```

3. **Enter Docker Container**

   ```
   docker run -it -v /{code_path}/pytorch:/home/pytorch manylinux-builder:v1 bash
   # {code_path} is the torch_npu source code path
   ```

4. **Compile torch-npu**

   Take **Python 3.8** as an example.

   ```
   cd /home/pytorch
   bash ci/build.sh --python=3.8
   ```

## Getting Started

### Prerequisites

Initialize **CANN** environment variable by running the command as shown below.

```Shell
# Default path, change it if needed.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### Quick Verification

You can quickly experience **Ascend NPU** by the following simple examples.

```Python
import torch
import torch_npu

x = torch.randn(2, 2).npu()
y = torch.randn(2, 2).npu()
z = x.mm(y)

print(z)
```

## PyTorch and Python Version Matching Table

| PyTorch Version | Python Version                                            |
| ------------- | :----------------------------------------------------------- |
| PyTorch1.11.0 | Python3.7.x(>=3.7.5),Python3.8.x,Python3.9.x,Python3.10.x |
| PyTorch2.0.1  | Python3.8.x,Python3.9.x,Python3.10.x                       |
| PyTorch2.1.0  | Python3.8.x,Python3.9.x,Python3.10.x                       |

## Ascend Auxiliary Software

**PyTorch Adapter** versions follow the naming convention `{PyTorch version}-{Ascend version}`, where the former represents the PyTorch version compatible with the **PyTorch Adapter**, and the latter is used to match the CANN version. The detailed matching is as follows:

| CANN Version         | Supported PyTorch Version | Supported Adapter Version | Github Branch          | AscendHub Image Version/Name([Link](https://ascendhub.huawei.com/#/detail/pytorch-modelzoo)) |
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

## Suggestions and Communication

Everyone is welcome to contribute to the community. If you have any questions or suggestions, you can submit [Github Issues](https://github.com/Ascend/pytorch/issues). We will reply to you as soon as possible. Thank you very much.

## Branch Maintenance Policies

The version branches of AscendPyTorch have the following maintenance phases:

| **Status**        | **Duration** | **Description**                                                                                                                |
|-------------------|--------------|--------------------------------------------------------------------------------------------------------------------------------|
| Planning          | 1-3 months   | Plan features.                                                                                                                 |
| Development       | 3 months     | Develop features.                                                                                                              |
| Maintained        | 6-12 months | Allow the incorporation of all resolved issues and release the version, Different versions of PyTorch adopt varying support policies. The maintenance periods for Regular Releases and Long-Term Support versions are 6 months and 12 months, respectively. |
| Unmaintained      | 0-3 months   | Allow the incorporation of all resolved issues. No dedicated maintenance personnel are available. No version will be released. |
| End Of Life (EOL) | N/A          | Do not accept any modification to a branch.                                                                                    |

##  PyTorch Maintenance Policies

| **PyTorch** |  **Maintenance Policies** | **Status** | **Launch Date**       | **Subsequent Status**            | **EOL Date**     |
|-----------|--------------------|--------------|------------|-----------------|-----------|
| 2.1.0     | Long Term Support  | Maintained   | 2023/10/15 | Unmaintained 2024-10-15 estimated |           |
| 2.0.1     | Regular Release    | Maintained   | 2023/7/19  | Unmaintained 2024-1-19 estimated  |           |
| 1.11.0    | Long Term Support  | Maintained   | 2023/4/19  | Unmaintained 2024-4-19 estimated  |           |
| 1.8.1     | Long Term Support  | EOL          | 2022/4/10  |                 | 2023/4/10 |
| 1.5.0     | Long Term Support  | EOL          | 2021/7/29  |                 | 2022/7/29 |

## Reference Documents

For more detailed information on installation guides, model migration, training/inference tutorials, and API lists, please refer to the [PyTorch Ascend Adapter on the HiAI Community](https://www.hiascend.com/software/ai-frameworks/commercial).

| Document Name                    | Document Link                                                 |
| -------------------------------- | ------------------------------------------------------------ |
| AscendPyTorch Installation Guide  | [link](https://www.hiascend.com/document/detail/zh/canncommercial/700/envdeployment/instg/instg_0163.html) |
| AscendPyTorch Network Model Migration and Training | [link](https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/AImpug_000002.html) |
| AscendPyTorch Online Inference    | [link](https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptonlineinfer/PyTorch_Infer_000001.html) |
| AscendPyTorch Operator Adaptation | [link](https://www.hiascend.com/document/detail/zh/canncommercial/700/operatordev/tbeaicpudevg/atlasopdev_10_0086.html) |
| AscendPyTorch API List (PyTorch and Custom Interfaces) | [link](https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/ptaoplist_000002.html) |

## License

PyTorch Ascend Adapter has a BSD-style license, as found in the [LICENSE](LICENSE) file.
