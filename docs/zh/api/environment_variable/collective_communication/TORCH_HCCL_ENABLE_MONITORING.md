# TORCH\_HCCL\_ENABLE\_MONITORING

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可控制是否启动heartbeat monitor线程。该线程用于检测HCCL WatchDog线程是否卡死，并在WatchDog无心跳时自动dump调试信息并终止进程。

- 0：不启动monitor线程。
- 1：启动monitor线程，WatchDog卡死时将触发`TORCH_HCCL_DUMP_ON_TIMEOUT`逻辑并终止进程。

默认值：0。

> [!NOTE]
>
> - 当前版本同时兼容旧名称`HCCL_ENABLE_MONITORING`。
> - WatchDog心跳超时时间由`TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC`控制，默认600秒。
> - 建议开启此功能以预防训练任务因HCCL API卡死而长时间占用集群资源。

该变量对应PyTorch的`TORCH_NCCL_ENABLE_MONITORING`。

> [!NOTE]
>
> PyTorch默认开启，TorchNPU默认关闭。

## 配置示例

```bash
export TORCH_HCCL_ENABLE_MONITORING=1
```

## 使用约束

- 建议同时设置`TORCH_HCCL_TRACE_BUFFER_SIZE > 0`以便在卡死时dump有效的调试信息。
- Monitor线程在ProcessGroupHCCL构造时启动，需在创建PG前设置环境变量。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
