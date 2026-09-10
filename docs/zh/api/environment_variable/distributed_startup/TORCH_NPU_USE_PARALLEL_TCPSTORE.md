# TORCH\_NPU\_USE\_PARALLEL\_TCPSTORE

## 功能描述

通过此环境变量可控制是否将`env://` rendezvous URL改为`parallel://`，从而启用ParallelStore作为分布式存储后端，提升建链性能。

- 精确等于字符串"True"（大小写敏感）：启用ParallelStore，将`env://`改写为`parallel://<MASTER_ADDR>:<MASTER_PORT>`。
- 其他值（包括"true"、"TRUE"）：视为关闭。

默认值："False"。

> [!NOTE]
>
> - 此环境变量仅在`torch_npu`分布式初始化时通过`torch_npu.distributed`的`_trigger_rendezvous_decorator`读取。
> - 使用`torch_npu_run`启动时，默认会设置此环境变量，通常无需手动配置。
> - 启用后，`MASTER_ADDR`缺省值为`127.0.0.1`，`MASTER_PORT`缺省值为`29500`。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export TORCH_NPU_USE_PARALLEL_TCPSTORE=True
```

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
