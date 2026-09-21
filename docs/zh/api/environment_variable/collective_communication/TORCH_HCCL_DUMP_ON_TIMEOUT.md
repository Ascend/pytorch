# TORCH\_HCCL\_DUMP\_ON\_TIMEOUT

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可控制发生HCCL超时或错误时是否自动触发Flight Recorder调试信息dump。

- 配置为“0”：超时或错误时不触发dump。
- 配置为“1”：超时或错误时触发本rank的dump，并通过Store协调其他rank同步dump。

该环境变量默认值配置为“0”。

该变量对应PyTorch的`TORCH_FR_DUMP_ON_TIMEOUT`(PyTorch 2.15版本及以上)，PyTorch兼容历史版本变量[`TORCH_NCCL_DUMP_ON_TIMEOUT`](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html#enabling-flight-recorder)（PyTorch 2.14版本及以下），PyTorch默认开启。

## 配置示例

```bash
export TORCH_HCCL_DUMP_ON_TIMEOUT=1
```

## 使用约束

- 建议配合`TORCH_HCCL_TRACE_BUFFER_SIZE > 0`和`TORCH_HCCL_ENABLE_MONITORING=1`使用。
- dump文件路径由[TORCH_HCCL_DEBUG_INFO_TEMP_FILE](TORCH_HCCL_DEBUG_INFO_TEMP_FILE.md)控制，默认输出到`/tmp/hccl_trace_rank_<rank>`。

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas 训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 训练系列产品</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT</term>
<!-- end id4 -->
