# TORCH\_HCCL\_STATUS\_SAVE\_PATH

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可配置HCCL进程组状态文件的保存目录。

默认值：`/tmp`。

状态文件名格式为：`torch_hccl_status-<global_rank>_<master_addr>_<deviceId>_<numRanks>_<pid>_<timestamp>.log`。

> [!NOTE]
>
> - 此环境变量在`ProcessGroupHCCL.cpp`文件级全局初始化时读取，**应在加载`torch_npu`扩展前设置**。
> - 仅当`TORCH_HCCL_STATUS_SAVE_ENABLE=1`时，状态文件才会写入该目录。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export TORCH_HCCL_STATUS_SAVE_PATH=/data/hccl_status
```

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
