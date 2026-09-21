# TORCH\_HCCL\_STATUS\_SAVE\_ENABLE

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可控制HCCL进程组状态信息的周期性保存。状态文件包含collective操作的`seq`、`op_type`、`pg_id`、`comm_ids`和`status`等信息，以及异常退出时的错误信息。

- 配置为“0”：不保存状态信息。
- 配置为“1”：启用状态保存。

该环境变量默认值为“0”。

> [!NOTE]
>
> - 此环境变量在`ProcessGroupHCCL.cpp`的全局初始化阶段读取，需在加载`torch_npu`扩展前设置。
> - 状态文件保存在`TORCH_HCCL_STATUS_SAVE_PATH`指定目录下，保存间隔由`TORCH_HCCL_STATUS_SAVE_INTERVAL`控制。
> - 状态保存在WatchDog线程中进行，WatchDog异常退出时会自动追加错误状态。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export TORCH_HCCL_STATUS_SAVE_ENABLE=1
```

## 使用约束

无

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas 训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 训练系列产品</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT</term>
<!-- end id4 -->
