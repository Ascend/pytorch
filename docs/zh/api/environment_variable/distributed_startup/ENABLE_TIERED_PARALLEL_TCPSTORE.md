# ENABLE\_TIERED\_PARALLEL\_TCPSTORE

## 功能描述

通过此环境变量可控制ParallelStore是否启用分层建链（tiered）优化模式。分层建链将建链复杂度从O(n)降低到O(sqrt(n))，在大规模集群场景下显著提升建链性能。

- 字符串"true"（大小写不敏感）：开启分层建链。
- 其他值：关闭。

默认值：`torch_npu_run`命令行参数`--enable_tiered_parallel_tcpstore`的默认值，通常为"false"。

> [!NOTE]
>
> - 此环境变量在`torch_npu_run`的`_create_parallel_handler`中通过`setdefault`写入，通常由`torch_npu_run`的`--enable_tiered_parallel_tcpstore`参数控制，详细说明可参考[torch_npu_run使用指导](../../../developer_notes/distributed/torch_npu_run.md#使用指导)。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export ENABLE_TIERED_PARALLEL_TCPSTORE=true
```

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
