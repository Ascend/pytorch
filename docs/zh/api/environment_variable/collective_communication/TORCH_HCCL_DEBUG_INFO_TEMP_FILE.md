# TORCH\_HCCL\_DEBUG\_INFO\_TEMP\_FILE

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可配置dump文件的转储路径及文件名前缀，每个rank对应一个文件，HCCL debug信息将输出到该文件。

- 默认值为`/tmp/hccl_trace_rank_`，最终文件名为`<前缀><rank序号>`。例如，默认配置下第0个rank的dump文件为`/tmp/hccl_trace_rank_0`。
- 配置指定路径前缀：以指定路径作为文件名前缀，dump文件输出到该前缀所在目录，文件名为`<前缀><rank序号>`。

该变量对应PyTorch的[`TORCH_FR_DUMP_TEMP_FILE`](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html#enabling-flight-recorder)。PyTorch同时兼容[`TORCH_NCCL_DEBUG_INFO_TEMP_FILE`](https://docs.pytorch.org/docs/2.14/torch_nccl_environment_variables.html)名称。

> [!NOTE]
>
> PyTorch默认为`$XDG_CACHE_HOME/torch/comm_lib_trace_rank_`，未设置`XDG_CACHE_HOME`时使用`$HOME/.cache/torch/comm_lib_trace_rank_`；TorchNPU默认为`/tmp/hccl_trace_rank_`。

## 配置示例

配置指定路径

```bash
export TORCH_HCCL_DEBUG_INFO_TEMP_FILE=/data/hccl_trace_rank_
```

## 使用约束

仅当`TORCH_HCCL_TRACE_BUFFER_SIZE > 0`且有dump触发时才会生成文件。

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
