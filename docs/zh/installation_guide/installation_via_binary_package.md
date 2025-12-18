# 方式一：二进制软件包安装

通过wheel格式的二进制软件包直接安装PyTorch。

执行安装命令前，请参见[安装前准备](preparing_installation.md)完成环境变量配置及其他环境准备。

## 安装PyTorch


|PyTorch版本<!-- class: installation_torch_npu -->|torch_npu插件版本|Python版本|系统架构|CANN版本|安装方式|安装命令|
|--|--|--|--|--|--|--|
|2.6.0|7.3.0|Python 3.9|AArch64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp39-cp39-manylinux_2_28_aarch64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.6.0/torch_npu-2.6.0.post3-cp39-cp39-manylinux_2_28_aarch64.whl<br># 安装命令<br>pip3 install torch-2.6.0+cpu-cp39-cp39-manylinux_2_28_aarch64.whl <br>pip3 install torch_npu-2.6.0.post3-cp39-cp39-manylinux_2_28_aarch64.whl</copy>|
|2.6.0|7.3.0|Python 3.9|X86_64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp39-cp39-linux_x86_64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.6.0/torch_npu-2.6.0.post3-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl<br># 安装命令<br>pip3 install torch-2.6.0+cpu-cp39-cp39-linux_x86_64.whl<br>pip3 install torch_npu-2.6.0.post3-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl</copy>|
|2.6.0|7.3.0|Python 3.10|AArch64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.6.0/torch_npu-2.6.0.post3-cp310-cp310-manylinux_2_28_aarch64.whl <br># 安装命令<br>pip3 install torch-2.6.0+cpu-cp310-cp310-manylinux_2_28_aarch64.whl<br>pip3 install torch_npu-2.6.0.post3-cp310-cp310-manylinux_2_28_aarch64.whl</copy>|
|2.6.0|7.3.0|Python 3.10|X86_64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp310-cp310-linux_x86_64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.6.0/torch_npu-2.6.0.post3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl<br># 安装命令<br>pip3 install torch-2.6.0+cpu-cp310-cp310-linux_x86_64.whl <br>pip3 install torch_npu-2.6.0.post3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl </copy>|
|2.6.0|7.3.0|Python 3.11|AArch64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.6.0/torch_npu-2.6.0.post3-cp311-cp311-manylinux_2_28_aarch64.whl<br># 安装命令<br>pip3 install torch-2.6.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl<br>pip3 install torch_npu-2.6.0.post3-cp311-cp311-manylinux_2_28_aarch64.whl</copy>|
|2.6.0|7.3.0|Python 3.11|X86_64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-linux_x86_64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.6.0/torch_npu-2.6.0.post3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl<br># 安装命令<br>pip3 install torch-2.6.0+cpu-cp311-cp311-linux_x86_64.whl<br>pip3 install torch_npu-2.6.0.post3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl</copy>|
|2.6.0|7.3.0|Python 3.12|AArch64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.6.0/torch_npu-2.6.0.post3-cp312-cp312-manylinux_2_28_aarch64.whl<br># 安装命令<br>pip3 install torch-2.6.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl<br>pip3 install torch_npu-2.6.0.post3-cp312-cp312-manylinux_2_28_aarch64.whl</copy>|
|2.6.0|7.3.0|Python 3.12|X86_64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp312-cp312-linux_x86_64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.6.0/torch_npu-2.6.0.post3-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl<br># 安装命令<br>pip3 install torch-2.6.0+cpu-cp312-cp312-linux_x86_64.whl<br>pip3 install torch_npu-2.6.0.post3-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl</copy>|
|2.7.1|7.3.0|Python 3.9|AArch64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp39-cp39-manylinux_2_28_aarch64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.7.1/torch_npu-2.7.1-cp39-cp39-manylinux_2_28_aarch64.whl<br># 安装命令<br>pip3 install torch-2.7.1+cpu-cp39-cp39-manylinux_2_28_aarch64.whl<br>pip3 install torch_npu-2.7.1-cp39-cp39-manylinux_2_28_aarch64.whl</copy>|
|2.7.1|7.3.0|Python 3.9|X86_64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp39-cp39-manylinux_2_28_x86_64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.7.1/torch_npu-2.7.1-cp39-cp39-manylinux_2_28_x86_64.whl<br># 安装命令<br>pip3 install torch-2.7.1+cpu-cp39-cp39-manylinux_2_28_x86_64.whl<br>pip3 install torch_npu-2.7.1-cp39-cp39-manylinux_2_28_x86_64.whl</copy>|
|2.7.1|7.3.0|Python 3.10|AArch64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.7.1/torch_npu-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl<br># 安装命令<br>pip3 install torch-2.7.1+cpu-cp310-cp310-manylinux_2_28_aarch64.whl<br>pip3 install torch_npu-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl</copy>|
|2.7.1|7.3.0|Python 3.10|X86_64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.7.1/torch_npu-2.7.1-cp310-cp310-manylinux_2_28_x86_64.whl<br># 安装命令<br>pip3 install torch-2.7.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl<br>pip3 install torch_npu-2.7.1-cp310-cp310-manylinux_2_28_x86_64.whl</copy>|
|2.7.1|7.3.0|Python 3.11|AArch64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.7.1/torch_npu-2.7.1-cp311-cp311-manylinux_2_28_aarch64.whl<br># 安装命令<br>pip3 install torch-2.7.1+cpu-cp311-cp311-manylinux_2_28_aarch64.whl<br>pip3 install torch_npu-2.7.1-cp311-cp311-manylinux_2_28_aarch64.whl</copy>|
|2.7.1|7.3.0|Python 3.11|X86_64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.7.1/torch_npu-2.7.1-cp311-cp311-manylinux_2_28_x86_64.whl<br># 安装命令<br>pip3 install torch-2.7.1+cpu-cp311-cp311-manylinux_2_28_x86_64.whl<br>pip3 install torch_npu-2.7.1-cp311-cp311-manylinux_2_28_x86_64.whl</copy>|
|2.7.1|7.3.0|Python 3.12|AArch64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.7.1/torch_npu-2.7.1-cp312-cp312-manylinux_2_28_aarch64.whl<br># 安装命令<br>pip3 install torch-2.7.1+cpu-cp312-cp312-manylinux_2_28_aarch64.whl<br>pip3 install torch_npu-2.7.1-cp312-cp312-manylinux_2_28_aarch64.whl</copy>|
|2.7.1|7.3.0|Python 3.12|X86_64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.7.1/torch_npu-2.7.1-cp312-cp312-manylinux_2_28_x86_64.whl<br># 安装命令<br>pip3 install torch-2.7.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl<br>pip3 install torch_npu-2.7.1-cp312-cp312-manylinux_2_28_x86_64.whl</copy>|
|2.8.0|7.3.0|Python 3.9|AArch64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp39-cp39-manylinux_2_28_aarch64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.8.0/torch_npu-2.8.0-cp39-cp39-manylinux_2_28_aarch64.whl<br># 安装命令<br>pip3 install torch-2.8.0+cpu-cp39-cp39-manylinux_2_28_aarch64.whl<br>pip3 install torch_npu-2.8.0-cp39-cp39-manylinux_2_28_aarch64.whl</copy>|
|2.8.0|7.3.0|Python 3.9|X86_64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp39-cp39-manylinux_2_28_x86_64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.8.0/torch_npu-2.8.0-cp39-cp39-manylinux_2_28_x86_64.whl<br># 安装命令<br>pip3 install torch-2.8.0+cpu-cp39-cp39-manylinux_2_28_x86_64.whl<br>pip3 install torch_npu-2.8.0-cp39-cp39-manylinux_2_28_x86_64.whl</copy>|
|2.8.0|7.3.0|Python 3.10|AArch64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.8.0/torch_npu-2.8.0-cp310-cp310-manylinux_2_28_aarch64.whl<br># 安装命令<br>pip3 install torch-2.8.0+cpu-cp310-cp310-manylinux_2_28_aarch64.whl<br>pip3 install torch_npu-2.8.0-cp310-cp310-manylinux_2_28_aarch64.whl</copy>|
|2.8.0|7.3.0|Python 3.10|X86_64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.8.0/torch_npu-2.8.0-cp310-cp310-manylinux_2_28_x86_64.whl<br># 安装命令<br>pip3 install torch-2.8.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl<br>pip3 install torch_npu-2.8.0-cp310-cp310-manylinux_2_28_x86_64.whl</copy>|
|2.8.0|7.3.0|Python 3.11|AArch64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.8.0/torch_npu-2.8.0-cp311-cp311-manylinux_2_28_aarch64.whl<br># 安装命令<br>pip3 install torch-2.8.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl<br>pip3 install torch_npu-2.8.0-cp311-cp311-manylinux_2_28_aarch64.whl</copy>|
|2.8.0|7.3.0|Python 3.11|X86_64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.8.0/torch_npu-2.8.0-cp311-cp311-manylinux_2_28_x86_64.whl<br># 安装命令<br>pip3 install torch-2.8.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl<br>pip3 install torch_npu-2.8.0-cp311-cp311-manylinux_2_28_x86_64.whl</copy>|
|2.8.0|7.3.0|Python 3.12|AArch64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.8.0/torch_npu-2.8.0-cp312-cp312-manylinux_2_28_aarch64.whl<br># 安装命令<br>pip3 install torch-2.8.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl<br>pip3 install torch_npu-2.8.0-cp312-cp312-manylinux_2_28_aarch64.whl</copy>|
|2.8.0|7.3.0|Python 3.12|X86_64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.8.0/torch_npu-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl<br># 安装命令<br>pip3 install torch-2.8.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl<br>pip3 install torch_npu-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl</copy>|
|2.9.0|7.3.0|Python 3.9|AArch64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp39-cp39-manylinux_2_28_aarch64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.9.0/torch_npu-2.9.0-cp39-cp39-manylinux_2_28_aarch64.whl<br># 安装命令<br>pip3 install torch-2.9.0+cpu-cp39-cp39-manylinux_2_28_aarch64.whl<br>pip3 install torch_npu-2.9.0-cp39-cp39-manylinux_2_28_aarch64.whl</copy>|
|2.9.0|7.3.0|Python 3.9|X86_64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp39-cp39-manylinux_2_28_x86_64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.9.0/torch_npu-2.9.0-cp39-cp39-manylinux_2_28_x86_64.whl<br># 安装命令<br>pip3 install torch-2.9.0+cpu-cp39-cp39-manylinux_2_28_x86_64.whl<br>pip3 install torch_npu-2.9.0-cp39-cp39-manylinux_2_28_x86_64.whl</copy>|
|2.9.0|7.3.0|Python 3.10|AArch64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.9.0/torch_npu-2.9.0-cp310-cp310-manylinux_2_28_aarch64.whl<br># 安装命令<br>pip3 install torch-2.9.0+cpu-cp310-cp310-manylinux_2_28_aarch64.whl<br>pip3 install torch_npu-2.9.0-cp310-cp310-manylinux_2_28_aarch64.whl</copy>|
|2.9.0|7.3.0|Python 3.10|X86_64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.9.0/torch_npu-2.9.0-cp310-cp310-manylinux_2_28_x86_64.whl<br># 安装命令<br>pip3 install torch-2.9.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl<br>pip3 install torch_npu-2.9.0-cp310-cp310-manylinux_2_28_x86_64.whl</copy>|
|2.9.0|7.3.0|Python 3.11|AArch64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.9.0/torch_npu-2.9.0-cp311-cp311-manylinux_2_28_aarch64.whl<br># 安装命令<br>pip3 install torch-2.9.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl<br>pip3 install torch_npu-2.9.0-cp311-cp311-manylinux_2_28_aarch64.whl</copy>|
|2.9.0|7.3.0|Python 3.11|X86_64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.9.0/torch_npu-2.9.0-cp311-cp311-manylinux_2_28_x86_64.whl<br># 安装命令<br>pip3 install torch-2.9.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl<br>pip3 install torch_npu-2.9.0-cp311-cp311-manylinux_2_28_x86_64.whl</copy>|
|2.9.0|7.3.0|Python 3.12|AArch64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.9.0/torch_npu-2.9.0-cp312-cp312-manylinux_2_28_aarch64.whl<br># 安装命令<br>pip3 install torch-2.9.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl<br>pip3 install torch_npu-2.9.0-cp312-cp312-manylinux_2_28_aarch64.whl</copy>|
|2.9.0|7.3.0|Python 3.12|X86_64|8.5.0|Pip|<copy># 下载软件包<br>wget https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl<br>wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.9.0/torch_npu-2.9.0-cp312-cp312-manylinux_2_28_x86_64.whl<br># 安装命令<br>pip3 install torch-2.9.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl<br>pip3 install torch_npu-2.9.0-cp312-cp312-manylinux_2_28_x86_64.whl</copy>|

> [!NOTE]
> -   出现“找不到google或protobuf，或者protobuf版本过高”报错时，需执行如下命令：
>       ```
>       pip3 install protobuf==3.20
>       ```
> -   更多PyTorch版本可单击[Link](https://download.pytorch.org/whl/torch/)进行查询。
> -   更多torch\_npu插件版本可单击[Link](https://gitcode.com/Ascend/pytorch/releases)查询。


## 安装后验证

执行以下命令可检查PyTorch框架和torch\_npu插件是否已成功安装。

-   方法一

    ```Python
    python3 -c "import torch;import torch_npu; a = torch.randn(3, 4).npu(); print(a + a);"
    ```

    输出如下类似信息说明安装成功。

    ```ColdFusion
    tensor([[-0.6066,  6.3385,  0.0379,  3.3356],
            [ 2.9243,  3.3134, -1.5465,  0.1916],
            [-2.1807,  0.2008, -1.1431,  2.1523]], device='npu:0')
    ```

-   方法二

    ```Python
    import torch
    import torch_npu
    
    x = torch.randn(2, 2).npu()
    y = torch.randn(2, 2).npu()
    z = x.mm(y)
    
    print(z)
    ```

    输出如下类似信息说明安装成功。

    ```ColdFusion
    tensor([[-0.0515,  0.3664],
            [-0.1258, -0.5425]], device='npu:0')
    ```

