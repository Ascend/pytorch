# NPU\_SHMEM\_SYMMETRIC\_SIZE

## 功能描述

通过此环境变量可配置NPU对称内存（symmetric memory）的堆大小。对称内存用于NPU设备间的直接内存访问，是NPUSHMEM功能的基础。

默认值为1 GiB。支持以下格式：

- 纯数字（字节）：例如`1073741824`表示1 GiB。
- 数字加后缀：`k/K`（KB）、`m/M`（MB）、`g/G`（GB）、`t/T`（TB）。

> [!NOTE]
>
> - 此环境变量在首次调用`OptionsManager::GetShmemSymmetricSize`时读取并缓存，运行中修改不会生效。
> - 该值传递给`aclshmemx_init_attr`或`shmem_set_attr`用于初始化对称内存区域。
> - 配置值必须为正数，非法格式会抛出`NPU_SHMEM_SYMMETRIC_SIZE is invalid`错误。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
# 配置为2 GiB
export NPU_SHMEM_SYMMETRIC_SIZE=2G

# 配置为512 MB
export NPU_SHMEM_SYMMETRIC_SIZE=512M
```

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
