# PROXY\_AGENT\_PID\_USE\_LOCAL\_SOCKET\_PATH

## 功能描述

通过此环境变量可将agent进程的PID传递给ParallelStore，用于控制本地socket路径的生成。在`torch_npu_run`的`_create_parallel_handler`中，通过`setdefault`将当前进程PID写入此变量。

- 默认值“-1”：表示未设置，ParallelStore不启用基于agent PID的本地socket路径生成。
- 配置为指定PID：ParallelStore使用该PID生成agent本地socket路径，用于agent与worker之间的本地通信寻址。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

通常无需手动配置，由`torch_npu_run`自动设置。

```bash
export PROXY_AGENT_PID_USE_LOCAL_SOCKET_PATH=12345
```

## 使用约束

无

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3训练系列产品</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT系列产品</term>
<!-- end id4 -->
