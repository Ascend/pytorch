# TORCH\_HCCL\_STATUS\_SAVE\_INTERVAL

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可配置HCCL进程组状态保存的间隔时间。

- 该环境变量默认值为2，单位为s。
- 如果配置为小于等于0的值，会被重置为2。

> [!NOTE]
>
> 此环境变量在首次调用时读取并缓存。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export TORCH_HCCL_STATUS_SAVE_INTERVAL=5
```

## 使用约束

仅当`TORCH_HCCL_STATUS_SAVE_ENABLE=1`时生效。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
