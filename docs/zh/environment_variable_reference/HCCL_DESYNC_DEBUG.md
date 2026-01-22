# HCCL\_DESYNC\_DEBUG

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可控制是否进行通信超时分析。

-   0：不开启通信超时分析。
-   1：开启通信超时分析。

默认值：0。

> [!NOTE]  
> -   当前版本，仅打印超时分析结果，不会终止进程。
> -   当集群组网规模较大时，若启用此环境变量，可能会出现训练进程异常卡死的情况。

## 配置示例

```
export HCCL_DESYNC_DEBUG=1
```

## 使用约束

PyTorch版本为1.11.0时，此环境变量需要与[HCCL\_ASYNC\_ERROR\_HANDLING](HCCL_ASYNC_ERROR_HANDLING.md)同时使用，即若HCCL\_DESYNC\_DEBUG配置为1，HCCL\_ASYNC\_ERROR\_HANDLING需要同步配置为1。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>
-   <term>Atlas 推理系列产品</term>

