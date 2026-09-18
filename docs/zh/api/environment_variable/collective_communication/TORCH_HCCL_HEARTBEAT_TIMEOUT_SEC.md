# TORCH\_HCCL\_HEARTBEAT\_TIMEOUT\_SEC

## 功能描述

在使用HCCL作为通信后端时，通过此环境变量可设置heartbeat monitor判定WatchDog无响应的超时时间。当WatchDog线程停止响应超过该时长时，monitor将判定其无响应，并触发dump及进程终止。

该环境变量默认值为600s，单位为秒。

该变量对应PyTorch的[`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`](https://docs.pytorch.org/docs/2.14/torch_nccl_environment_variables.html)，PyTorch默认为480秒。

## 配置示例

```bash
export TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC=300
```

## 使用约束

此环境变量仅在`TORCH_HCCL_ENABLE_MONITORING=1`时生效。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
