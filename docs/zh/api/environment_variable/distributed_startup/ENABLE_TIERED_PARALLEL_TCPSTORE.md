# ENABLE\_TIERED\_PARALLEL\_TCPSTORE

## 功能描述

通过此环境变量可控制ParallelStore是否启用分层建链（tiered）优化模式。分层建链将建链复杂度从O(n)降低到O(sqrt(n))，在大规模集群场景下显著提升建链性能。

- 配置为字符串`true`（不区分大小写）：开启分层建链优化模式。
- 配置为其他值：关闭分层建链优化模式。
- 默认值：`torch_npu_run`命令行参数`--enable_tiered_parallel_tcpstore`的默认值，通常为“false”，即默认关闭分层建链。

> [!NOTE]
>
> 此环境变量在`torch_npu_run`的`_create_parallel_handler`中通过`setdefault`写入，通常由`torch_npu_run`的`--enable_tiered_parallel_tcpstore`参数控制，详细说明可参考[torch_npu_run使用指导](../../../developer_notes/distributed/startup_and_fault_tolerance/torch_npu_run.md)。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export ENABLE_TIERED_PARALLEL_TCPSTORE=true
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
