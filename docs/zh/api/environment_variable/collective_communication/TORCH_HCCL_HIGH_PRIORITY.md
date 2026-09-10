# TORCH\_HCCL\_HIGH\_PRIORITY

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可控制是否强制从高优先级NPU stream pool中获取通信流。

- 0：不强制使用高优先级stream，PG option中的优先级设置可能生效。
- 1：强制使用高优先级NPU stream pool中的通信流。

默认值：0。

> [!NOTE]
>
> - 当前版本同时兼容旧名称`HCCL_HIGH_PRIORITY`。
> - 高优先级stream可能减少通信延迟，但可能影响计算与通信的并发调度。

该变量对应PyTorch的`TORCH_NCCL_HIGH_PRIORITY`。配置方式一致，默认均关闭。

## 配置示例

```bash
export TORCH_HCCL_HIGH_PRIORITY=1
```

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
