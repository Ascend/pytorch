# PROXY\_AGENT\_PID\_USE\_LOCAL\_SOCKET\_PATH

## 功能描述

通过此环境变量可将agent进程的PID传递给ParallelStore，用于控制本地socket路径的生成。

默认值：-1（表示未设置）。

在`torch_npu_run`的`_create_parallel_handler`中，通过`setdefault`将当前进程PID写入此变量。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

通常无需手动配置，由`torch_npu_run`自动设置。

```bash
export PROXY_AGENT_PID_USE_LOCAL_SOCKET_PATH=12345
```

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
