# TORCH\_HCCL\_COORD\_CHECK\_MILSEC

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可设置WatchDog线程中轮询检查Store dump signal的时间间隔。WatchDog线程会以此间隔检查其他rank通过Store发出的dump信号，用于跨rank协调的dump通知。

该环境变量的默认值为1000ms，单位为毫秒。

> [!NOTE]
>
> 减小此值可以更快响应dump信号，但会增加对Store的轮询频率。

该变量对应PyTorch的[`TORCH_NCCL_COORD_CHECK_MILSEC`](https://docs.pytorch.org/docs/2.14/torch_nccl_environment_variables.html)。配置方式一致，默认值均为1000ms。

## 配置示例

```bash
export TORCH_HCCL_COORD_CHECK_MILSEC=500
```

## 使用约束

此环境变量仅在`TORCH_HCCL_ENABLE_MONITORING=1`时生效。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
