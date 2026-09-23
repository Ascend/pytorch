# NPU\_SHMEM\_SYMMETRIC\_SIZE

## 功能描述

通过此环境变量可配置NPU对称内存（symmetric memory）的堆大小。对称内存用于NPU设备间的直接内存访问，是NPUSHMEM功能的基础。

- 默认值：1 GiB，分配1 GiB对称内存堆大小。
- 配置为纯数字（字节）：例如`1073741824`，分配1073741824字节对称内存堆大小。
- 配置为数字加后缀：`k/K`（KB）、`m/M`（MB）、`g/G`（GB）、`t/T`（TB），例如`2G`分配2 GiB对称内存堆大小，`512M`分配512 MB对称内存堆大小。

配置更大的堆可容纳更多设备间直接访问的对称内存数据，配置过小可能导致对称内存分配失败。

> [!NOTE]
>
> - 此环境变量在首次调用`OptionsManager::GetShmemSymmetricSize`时读取并缓存，运行中修改不会生效。
> - 该值传递给`aclshmemx_init_attr`或`shmem_set_attr`用于初始化对称内存区域。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
# 配置为2 GiB
export NPU_SHMEM_SYMMETRIC_SIZE=2G

# 配置为512 MB
export NPU_SHMEM_SYMMETRIC_SIZE=512M
```

## 使用约束

配置值必须为正数，非法格式会抛出`NPU_SHMEM_SYMMETRIC_SIZE is invalid`错误。

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
