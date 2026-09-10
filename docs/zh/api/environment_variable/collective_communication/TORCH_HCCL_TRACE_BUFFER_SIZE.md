# TORCH\_HCCL\_TRACE\_BUFFER\_SIZE

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可配置HCCL Flight Recorder环形缓冲区中最大可记录的事件数。每个事件对应一次集合通信操作的开始或结束。

- 配置为0或负数时：关闭Flight Recorder记录功能，不记录任何事件，导出dump时也不包含trace内容。
- 配置为正整数时：开启Flight Recorder记录功能，缓冲区最多保存该数量的事件记录。

单位为个数，默认值为0。

> [!NOTE]
>
> - 此环境变量在`HCCLTraceBuffer`单例首次构造时读取并缓存，运行中修改不会生效。
> - 仅当`TRACE_BUFFER_SIZE > 0`时，`TORCH_HCCL_DUMP_ON_TIMEOUT`等超时dump功能才会产生有效内容。

该变量对应PyTorch的`TORCH_FR_BUFFER_SIZE`。PyTorch同时兼容`TORCH_NCCL_TRACE_BUFFER_SIZE`名称。

> [!NOTE]
>
> PyTorch默认为2000，TorchNPU默认为0（关闭记录）。

## 配置示例

```bash
export TORCH_HCCL_TRACE_BUFFER_SIZE=2000
```

## 使用约束

- 此环境变量需在加载`torch_npu`扩展前设置，运行中修改无效。
- 缓冲区大小直接影响内存占用，建议根据实际collective数量合理配置。
- 开启记录后，HCCL watchDog线程会周期性处理缓冲区中的事件。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
