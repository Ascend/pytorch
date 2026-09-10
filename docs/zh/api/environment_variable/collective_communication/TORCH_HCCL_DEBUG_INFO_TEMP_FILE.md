# TORCH\_HCCL\_DEBUG\_INFO\_TEMP\_FILE

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可配置dump文件的名称前缀，HCCL debug信息将输出到该文件。

默认值为`/tmp/hccl_trace_rank_`，最终文件名为`<前缀><rank>`。

例如，默认配置下rank 0的dump文件为`/tmp/hccl_trace_rank_0`。

> [!NOTE]
>
> - 仅当`TORCH_HCCL_TRACE_BUFFER_SIZE > 0`且有dump触发时才会生成文件。

该变量对应PyTorch的`TORCH_FR_DUMP_TEMP_FILE`。PyTorch同时兼容`TORCH_NCCL_DEBUG_INFO_TEMP_FILE`名称。

> [!NOTE]
>
> PyTorch默认为`$XDG_CACHE_HOME/torch/comm_lib_trace_rank_`，未设置`XDG_CACHE_HOME`时使用`$HOME/.cache/torch/comm_lib_trace_rank_`；TorchNPU默认为`/tmp/hccl_trace_rank_`。

## 配置示例

```bash
export TORCH_HCCL_DEBUG_INFO_TEMP_FILE=/data/hccl_dumps/trace_
```

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
