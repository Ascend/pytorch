# TORCH\_HCCL\_STATUS\_SAVE\_PATH

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可配置HCCL进程组状态文件的保存目录。状态文件以固定命名规则创建在该目录下，文件名格式为`torch_hccl_status-<global_rank>_<master_addr>_<deviceId>_<numRanks>_<pid>_<timestamp>.log`。

- 默认值：`/tmp`，状态文件保存在`/tmp`目录下。
- 配置为指定路径：状态文件保存在该指定目录下，文件名格式不变。

> [!NOTE]
>
> 此环境变量在`ProcessGroupHCCL.cpp`的全局初始化阶段读取，应在加载`torch_npu`扩展前设置。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export TORCH_HCCL_STATUS_SAVE_PATH=/data/hccl_status
```

## 使用约束

仅当`TORCH_HCCL_STATUS_SAVE_ENABLE=1`时，状态文件才会写入目录。

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3训练系列产品</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT系列产品</term>
<!-- end id4 -->
