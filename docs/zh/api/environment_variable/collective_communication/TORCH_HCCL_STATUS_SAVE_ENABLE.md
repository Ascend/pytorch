# TORCH\_HCCL\_STATUS\_SAVE\_ENABLE

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可控制HCCL进程组状态信息的周期性保存。状态文件包含collective操作的seq、op_type、pg_id、comm_ids和status等信息，以及异常退出时的错误信息。

- 0：不保存状态信息。
- 1：启用状态保存。

默认值：0。

> [!NOTE]
>
> - 此环境变量在`ProcessGroupHCCL.cpp`文件级全局初始化时读取，**应在加载`torch_npu`扩展前设置**。
> - 状态文件保存在`TORCH_HCCL_STATUS_SAVE_PATH`指定目录下，保存间隔由`TORCH_HCCL_STATUS_SAVE_INTERVAL`控制。
> - 状态保存在WatchDog线程中进行，WatchDog异常退出时会自动追加错误状态。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export TORCH_HCCL_STATUS_SAVE_ENABLE=1
```

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
