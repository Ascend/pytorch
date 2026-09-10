# TORCH\_HCCL\_HEARTBEAT\_TIMEOUT\_SEC

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可设置heartbeat monitor判定WatchDog卡死的超时时间。

单位为秒，默认值为600秒（10分钟）。

当WatchDog线程的心跳停止超过此时间时，monitor认为WatchDog已卡死，将触发dump并终止进程。

> [!NOTE]
>
> - 此环境变量仅在`TORCH_HCCL_ENABLE_MONITORING=1`时生效。

该变量对应PyTorch的`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`。

> [!NOTE]
>
> PyTorch默认为480秒，TorchNPU默认为600秒。

## 配置示例

```bash
export TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC=300
```

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
