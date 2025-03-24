# Ascend Extension for PyTorch

## Overview

This repository develops the **Ascend Extension for PyTorch** named **torch_npu** to adapt **Ascend NPU** to **PyTorch** so that developers who use the **PyTorch** can obtain powerful compute capabilities of **Ascend AI Processors**.

Ascend is a full-stack AI computing infrastructure for industry applications and services based on Huawei Ascend processors and software. For more information about Ascend, see [Ascend Community](https://www.hiascend.com/en/).

## Installation

### From Binary

Provide users with wheel package to quickly install **torch_npu**. Before installing **torch_npu**, complete the installation of **CANN** according to [Ascend Auxiliary Software](#ascend-auxiliary-software). To obtain the **CANN** installation package, refer to the [CANN Installation](https://www.hiascend.com/en/software/cann/community).

1. **Install PyTorch**

Install **PyTorch** through pip.

**For Aarch64:**

```Shell
pip3 install torch==2.3.1
```

**For x86:**

```Shell
pip3 install torch==2.3.1+cpu  --index-url https://download.pytorch.org/whl/cpu
```

2. **Install torch-npu dependencies**

Run the following command to install dependencies.

```Shell
pip3 install pyyaml
pip3 install setuptools
```

If the installation fails, use the download link or visit the [PyTorch official website](https://pytorch.org/) to download the installation package of the corresponding version.

| OS arch | Python version | link                                                         |
| ------- | -------------- | ------------------------------------------------------------ |
| x86     | Python3.8      | [link](https://download.pytorch.org/whl/cpu/torch-2.3.1%2Bcpu-cp38-cp38-linux_x86_64.whl#sha256=cab92d5101e6db686c5525e04d87cedbcf3a556073d71d07fbe7d1ce09630ffb) |
| x86     | Python3.9      | [link](https://download.pytorch.org/whl/cpu/torch-2.3.1%2Bcpu-cp39-cp39-linux_x86_64.whl#sha256=a3cb8e61ba311cee1bb7463cbdcf3ebdfd071e2091e74c5785e3687eb02819f9) |
| x86     | Python3.10     | [link](https://download.pytorch.org/whl/cpu/torch-2.3.1%2Bcpu-cp310-cp310-linux_x86_64.whl#sha256=d679e21d871982b9234444331a26350902cfd2d5ca44ce6f49896af8b3a3087d) |
| x86     | Python3.11     | [link](https://download.pytorch.org/whl/cpu/torch-2.3.1%2Bcpu-cp311-cp311-linux_x86_64.whl#sha256=a272defe305dbd944aa28a91cc3db0f0149495b3ebec2e39723a7224fa05dc57) |
| aarch64 | Python3.8      | [link](https://download.pytorch.org/whl/cpu/torch-2.3.1-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl#sha256=3b7c1498f904f67eb1e331f2ebe8742771a2ce71b9ee9bc01de967257e881c7d) |
| aarch64 | Python3.9      | [link](https://download.pytorch.org/whl/cpu/torch-2.3.1-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl#sha256=db6bff4ba6273b59ae443de04b5adc36d6a40bb2898866133bff2d52f276eafe) |
| aarch64 | Python3.10     | [link](https://download.pytorch.org/whl/cpu/torch-2.3.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl#sha256=6544fdf29018668c0a6d4a1bcc955982c1ada70806281b010cba93bdcfbdcf22) |
| aarch64 | Python3.11     | [link](https://download.pytorch.org/whl/cpu/torch-2.3.1-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl#sha256=2aaf0e37734cbc5fe6bfcc81ada36ecbb899d4ddbe13498bd84aaca8a91c8628) |

3. **Install torch-npu**

```
pip3 install torch-npu==2.3.1.post4
```

### From Source

In some special scenarios, users may need to compile **torch-npu** by themselves.Select a branch in table [Ascend Auxiliary Software](#ascend-auxiliary-software) and a Python version in table [PyTorch and Python Version Matching Table](#pytorch-and-python-version-matching-table) first. The docker image is recommended for compiling torch-npu through the following steps(It is recommended to mount the working path only and avoid the system path to reduce security risks.), the generated .whl file path is ./dist/. Note that gcc version has the following constraints if you try to compile without using docker image: we recommend to use gcc 10.2 for ARM and gcc 9.3.1 for X86.

1. **Clone torch-npu**

   ```
   git clone https://github.com/ascend/pytorch.git -b v2.3.1 --depth 1
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

**Tips**

   If you would like to compile with new C++ ABI, then first run this command, at this point, the recommended compilation environment is same to community torch package: glibc 2.28, gcc 11.2.1
   
   ```
   export _GLIBCXX_USE_CXX11_ABI=1
   ```

   Meanwhile, we support configuring -fabi-version using the following variables，require consistency with the community torch package

   ```
   export _ABI_VERSION=16
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

## User Manual

Refer to [API of Ascend Extension for PyTorch](docs/api/torch_npu_apis.md) for more detailed informations.

## PyTorch and Python Version Matching Table

| PyTorch Version | Python Version                                            |
|-----------------|:----------------------------------------------------------|
| PyTorch1.11.0   | Python3.7.x(>=3.7.5),Python3.8.x,Python3.9.x,Python3.10.x |
| PyTorch2.1.0    | Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x                      |
| PyTorch2.2.0    | Python3.8.x,Python3.9.x,Python3.10.x                       |
| PyTorch2.3.1    | Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x                       |
| PyTorch2.4.0    | Python3.8.x,Python3.9.x,Python3.10.x,Python3.11.x                       |

## Ascend Auxiliary Software

**PyTorch Extension** versions follow the naming convention `{PyTorch version}-{Ascend version}`, where the former represents the PyTorch version compatible with the **PyTorch Extension**, and the latter is used to match the CANN version. The detailed matching is as follows:

| CANN Version          | Supported PyTorch Version | Supported Extension Version | Github Branch     |
|-----------------------|---------------------------|-----------------------------|-------------------|
| CANN 8.1.RC1          | 2.5.1                     | 2.5.0                       | v2.5.1-7.0.0      |
|                       | 2.4.0                     | 2.4.0.post3                 | v2.4.0-7.0.0      |
|                       | 2.3.1                     | 2.3.1.post5                 | v2.3.1-7.0.0      |
|                       | 2.1.0                     | 2.1.0.post11                | v2.1.0-7.0.0      |
| CANN 8.0.0            | 2.4.0                     | 2.4.0.post2                 | v2.4.0-6.0.0      | 
|                       | 2.3.1                     | 2.3.1.post4                 | v2.3.1-6.0.0      |
|                       | 2.1.0                     | 2.1.0.post10                | v2.1.0-6.0.0      |
| CANN 8.0.RC3          | 2.4.0                     | 2.4.0                       | v2.4.0-6.0.rc3    |
|                       | 2.3.1                     | 2.3.1.post2                 | v2.3.1-6.0.rc3    |
|                       | 2.1.0                     | 2.1.0.post8                 | v2.1.0-6.0.rc3    |
| CANN 8.0.RC2          | 2.3.1                     | 2.3.1                       | v2.3.1-6.0.rc2    |
|                       | 2.2.0                     | 2.2.0.post2                 | v2.2.0-6.0.rc2    |
|                       | 2.1.0                     | 2.1.0.post6                 | v2.1.0-6.0.rc2    |
|                       | 1.11.0                    | 1.11.0.post14               | v1.11.0-6.0.rc2   |
| CANN 8.0.RC2.alpha002 | 2.3.1                     | 2.3.1rc1                    | v2.3.1            |
| CANN 8.0.RC1          | 2.2.0                     | 2.2.0                       | v2.2.0-6.0.rc1    |
|                       | 2.1.0                     | 2.1.0.post4                 | v2.1.0-6.0.rc1    |
|                       | 1.11.0                    | 1.11.0.post11               | v1.11.0-6.0.rc1   |
| CANN 7.0.0            | 2.1.0                     | 2.1.0                       | v2.1.0-5.0.0      |
|                       | 2.0.1                     | 2.0.1.post1                 | v2.0.1-5.0.0      |
|                       | 1.11.0                    | 1.11.0.post8                | v1.11.0-5.0.0     |
| CANN 7.0.RC1          | 2.1.0                     | 2.1.0.rc1                   | v2.1.0-5.0.rc3    |
|                       | 2.0.1                     | 2.0.1                       | v2.0.1-5.0.rc3    |
|                       | 1.11.0                    | 1.11.0.post4                | v1.11.0-5.0.rc3   |
| CANN 6.3.RC3.1        | 1.11.0                    | 1.11.0.post3                | v1.11.0-5.0.rc2.2 |
| CANN 6.3.RC3          | 1.11.0                    | 1.11.0.post2                | v1.11.0-5.0.rc2.1 |
| CANN 6.3.RC2          | 2.0.1                     | 2.0.1.rc1                   | v2.0.1-5.0.rc2    |
|                       | 1.11.0                    | 1.11.0.post1                | v1.11.0-5.0.rc2   |
|                       | 1.8.1                     | 1.8.1.post2                 | v1.8.1-5.0.rc2    |
| CANN 6.3.RC1          | 1.11.0                    | 1.11.0                      | v1.11.0-5.0.rc1   |
|                       | 1.8.1                     | 1.8.1.post1                 | v1.8.1-5.0.rc1    |
| CANN 6.0.1            | 1.5.0                     | 1.5.0.post8                 | v1.5.0-3.0.0      |
|                       | 1.8.1                     | 1.8.1                       | v1.8.1-3.0.0      |
|                       | 1.11.0                    | 1.11.0.rc2（beta)            | v1.11.0-3.0.0     |
| CANN 6.0.RC1          | 1.5.0                     | 1.5.0.post7                 | v1.5.0-3.0.rc3    |
|                       | 1.8.1                     | 1.8.1.rc3                   | v1.8.1-3.0.rc3    |
|                       | 1.11.0                    | 1.11.0.rc1（beta)            | v1.11.0-3.0.rc3   |
| CANN 5.1.RC2          | 1.5.0                     | 1.5.0.post6                 | v1.5.0-3.0.rc2    | 
|                       | 1.8.1                     | 1.8.1.rc2                   | v1.8.1-3.0.rc2    | 
| CANN 5.1.RC1          | 1.5.0                     | 1.5.0.post5                 | v1.5.0-3.0.rc1    |
|                       | 1.8.1                     | 1.8.1.rc1                   | v1.8.1-3.0.rc1    |
| CANN 5.0.4            | 1.5.0                     | 1.5.0.post4                 | 2.0.4.tr5         |
| CANN 5.0.3            | 1.8.1                     | 1.5.0.post3                 | 2.0.3.tr5         |
| CANN 5.0.2            | 1.5.0                     | 1.5.0.post2                 | 2.0.2.tr5         |

## Hardware support

The Ascend training device includes the following models, all of which can be used as training environments for PyTorch models
| Product series        | Product model                    |
|-----------------------|----------------------------------|
| Atlas Training series products     | Atlas 800（model: 9000） |
|                       | Atlas 800（model：9010）          |
|                       | Atlas 900 PoD（model：9000）      |
|                       | Atlas 300T（model：9000）         |
|                       | Atlas 300T Pro（model：9000）     |
| Atlas A2 Training series products  | Atlas 800T A2       |
|                       | Atlas 900 A2 PoD                 |
|                       | Atlas 200T A2 Box16              |
|                       | Atlas 300T A2                    |

The Ascend inference device includes the following models, all of which can be used as inference environments for large models
| Product series        | Product model                        |
|-----------------------|----------------------------------|
| Atlas 800I A2 Inference product  | Atlas 800I A2         |

## Pipeline Status

Due to the asynchronous development mechanism of upstream and downstream, incompatible modifications in upstream may cause some functions of **torch_npu** to be unavailable (only upstream and downstream development branches are involved, excluding stable branches). Therefore, we built a set of daily tasks that make it easy to detect relevant issues in time and fix them within 48 hours (under normal circumstances), providing users with the latest features and stable quality.

|       **OS**        |             **CANN Version(Docker Image)**              |                 **Upstream Branch**                  |                  **Downstream Branch**                  |   **Period**   |                                                                          **Status**                                                                           |
|:-------------------:|:-------------------------------------------------------:|:----------------------------------------------------:|:-------------------------------------------------------:|:--------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| openEuler 22.03 SP2 | [CANN 7.1](https://hub.docker.com/r/ascendai/cann/tags) | [main](https://github.com/pytorch/pytorch/tree/main) | [master](https://github.com/Ascend/pytorch/tree/master) | UTC 1200 daily | [![Ascend NPU](https://github.com/Ascend/pytorch/actions/workflows/periodic.yml/badge.svg)](https://github.com/Ascend/pytorch/actions/workflows/periodic.yml) |

## Suggestions and Communication

Everyone is welcome to contribute to the community. If you have any questions or suggestions, you can submit [Github Issues](https://github.com/Ascend/pytorch/issues). We will reply to you as soon as possible. Thank you very much.

## Branch Maintenance Policies

The version branches of AscendPyTorch have the following maintenance phases:

| **Status**        | **Duration** | **Description**                                                                                                                |
|-------------------|--------------|--------------------------------------------------------------------------------------------------------------------------------|
| Planning          | 1-3 months   | Plan features.                                                                                                                 |
| Development       | 6-12 months     | Develop new features and fix issues, regularly release new versions. Different strategies are adopted for different versions of PyTorch, with a regular branch development cycle of 6 months and a long-term support branch development cycle of 12 months.                                                                                                             |
| Maintained        | 1 year/3.5 years | Regular Release branch for 1 year, Long Term Support branch maintenance for 3.5 years. Fix major issues, do not incorporate new features, and release patch versions based on the impact of fixed bugs. |
| End Of Life (EOL) | N/A          | Do not accept any modification to a branch.                                                                                    |

##  PyTorch Maintenance Policies

| **PyTorch** |  **Maintenance Policies** | **Status** | **Launch Date**       | **Subsequent Status**            | **EOL Date**     |
|-----------|--------------------|--------------|------------|-----------------|-----------|
| 2.4.0     |  Regular Release  | Development  | 2024/10/15 |Expected to enter maintenance status from  June 15, 2025 |           |
| 2.3.1     |  Regular Release  | Development   | 2024/06/06 | Expected to enter maintenance status from  June 7th, 2025 |           |
| 2.2.0     |  Regular Release  | Maintained   | 2024/04/01 | Expected to enter maintenance free status from September 10th, 2025|           |
| 2.1.0     | Long Term Support  | Development   | 2023/10/15 | Expected to enter maintenance status from September 15th, 2025 |           |
| 2.0.1     | Regular Release    | EOL   | 2023/7/19  |   | 2024/3/14          |
| 1.11.0    | Long Term Support  | Maintained   | 2023/4/19  | Expected to enter maintenance free status from September 10th, 2025  |           |
| 1.8.1     | Long Term Support  | EOL          | 2022/4/10  |                 | 2023/4/10 |
| 1.5.0     | Long Term Support  | EOL          | 2021/7/29  |                 | 2022/7/29 |

## Reference Documents

For more detailed information on installation guides, model migration, training/inference tutorials, and API lists, please refer to the [Ascend Extension for PyTorch on the HiAI Community](https://www.hiascend.com/software/ai-frameworks?framework=pytorch).

| Document Name                            | Document Link                                                                                                           |
|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| Installation Guide                       | [link](https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html)                                      |
| Network Model Migration and Training     | [link](https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/ptmoddevg/trainingmigrguide/PT_LMTMOG_0003.html)       |
| Operator Adaptation                      | [link](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/operatordev/tbeaicpudevg/atlasopdev_10_0086.html) |
| API List (PyTorch and Custom Interfaces) | [link](https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/apiref/apilist/ptaoplist_000002.html)                  |

## License

Ascend Extension for PyTorch has a BSD-style license, as found in the [LICENSE](LICENSE) file.
