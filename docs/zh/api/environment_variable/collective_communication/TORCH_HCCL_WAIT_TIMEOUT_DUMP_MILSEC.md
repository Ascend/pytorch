# TORCH\_HCCL\_WAIT\_TIMEOUT\_DUMP\_MILSEC

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可设置heartbeat monitor等待异步dump完成的最大时间。

单位为ms，默认值为60000（60秒）。

当发生超时触发dump时，monitor会等待其他rank的dump完成。超过此时间后，无论dump是否完成，monitor都会终止进程。

> [!NOTE]
>
> - 此环境变量仅在`TORCH_HCCL_ENABLE_MONITORING=1`时生效。

该变量对应PyTorch的`TORCH_FR_WAIT_TIMEOUT_DUMP_MILSEC`。PyTorch同时兼容`TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC`名称。

> [!NOTE]
>
> PyTorch默认为15000ms（15秒），TorchNPU默认为60000ms（60秒）。

## 配置示例

```bash
export TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC=30000
```

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
