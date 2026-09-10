# TORCH\_HCCL\_DUMP\_ON\_TIMEOUT

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可控制发生HCCL超时或错误时是否自动触发Flight Recorder调试信息dump。

- 0：超时或错误时不触发dump。
- 1：超时或错误时触发本rank的dump，并通过Store协调其他rank同步dump。

默认值：0。

> [!NOTE]
>
> - 开启此功能需要`TORCH_HCCL_TRACE_BUFFER_SIZE > 0`，否则无事件可dump。
> - 开启此功能后，建议同时开启`TORCH_HCCL_ENABLE_MONITORING`，以确保watchDog卡死时也能触发dump。

该变量对应PyTorch的`TORCH_FR_DUMP_ON_TIMEOUT`。PyTorch同时兼容`TORCH_NCCL_DUMP_ON_TIMEOUT`名称。

> [!NOTE]
>
> PyTorch进程组默认开启，TorchNPU默认关闭。

## 配置示例

```bash
export TORCH_HCCL_DUMP_ON_TIMEOUT=1
```

## 使用约束

- 建议配合`TORCH_HCCL_TRACE_BUFFER_SIZE > 0`和`TORCH_HCCL_ENABLE_MONITORING=1`使用。
- dump文件路径由`TORCH_HCCL_DEBUG_INFO_TEMP_FILE`控制，默认输出到`/tmp/hccl_trace_rank_<rank>`。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
