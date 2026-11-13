# HCCL对称内存池

## 简介

本文档介绍HCCL对称内存池，TorchNPU支持使用HCCL内存分配器创建`torch.npu.MemPool`并将池中的内存段注册到对称内存窗口用于集合通信，适用于需要在HCCL分配的内存上执行AllReduce等集合通信并统一管理通信缓冲区的用户。注册后，新增和释放的内存段会自动注册和注销。

## 使用场景

需要在HCCL分配的内存上执行AllReduce等集合通信，并统一管理通信缓冲区的分配与注册时，可以使用本特性。

## 使用指导

通过目标进程组获取HCCL后端，使用以下接口管理内存池：

| 接口 | 说明 |
| --- | --- |
| `backend.mem_allocator` | 获取HCCL内存分配器，用于创建`torch.npu.MemPool`。 |
| `backend.register_mem_pool(pool, symm=True)` | 将池中的内存段注册到对称内存窗口。`symm`默认为`False`，当前仅支持`True`，需显式指定。 |
| `backend.deregister_mem_pool(pool)` | 注销内存池，不释放池中的Tensor。 |

使用`torch.npu.use_mem_pool(pool)`将当前线程在上下文中的分配路由到该池，已有Tensor不会自动迁入池中。注册对称内存池前，需初始化目标进程组在该设备上的HCCL通信域。

对称窗口大小可通过`pg_options.hccl_config`中的`hccl_sym_win_max_mem_size_per_rank`配置，具体操作请参见[通过pg_options配置HCCL通信域参数](../parameter_configuration/setting_HCCL_communicator_parameter.md)。

## 使用样例

以下示例采用单机多进程、每个进程使用一张NPU的方式，在内存池上执行AllReduce。将代码保存为`hccl_mem_pool_example.py`：

```python
import os

import torch
import torch.distributed as dist
import torch_npu


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl")

    # 预热，完成当前设备上的通信域初始化。
    warmup = torch.zeros(1, device=device)
    dist.all_reduce(warmup)
    torch.npu.synchronize(device)
    del warmup

    # 创建内存池，分配并注册通信缓冲区。
    backend = dist.group.WORLD._get_backend(device)
    pool = torch.npu.MemPool(backend.mem_allocator)
    with torch.npu.use_mem_pool(pool, device=device):
        tensor = torch.ones(2 * 1024 * 1024, device=device)
    torch.npu.synchronize(device)
    backend.register_mem_pool(pool, symm=True)

    try:
        dist.all_reduce(tensor)
        torch.npu.synchronize(device)
        assert torch.all(tensor == dist.get_world_size()).item()
    finally:
        # 等待设备操作完成后注销，再释放Tensor和内存池。
        torch.npu.synchronize(device)
        backend.deregister_mem_pool(pool)

    del tensor, pool
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

> [!NOTE]
>
> 需在已完成CANN环境初始化且具备至少两张可用NPU的环境中执行：
>
> ```bash
> torchrun --standalone --nproc_per_node=2 hccl_mem_pool_example.py
> ```

## 约束说明

- 需要CANN 9.2.0及以上版本，硬件和通信算子支持范围以所用CANN/HCCL版本的说明为准。
- 需启用NPU缓存分配器，并关闭`expandable_segments`，具体请参见[PYTORCH_NPU_ALLOC_CONF](../../../api/environment_variable/memory_management/PYTORCH_NPU_ALLOC_CONF.md)。
- 各rank应保持对应内存分配的大小和顺序一致，并按一致顺序注册和注销。AllGather、ReduceScatter等操作的输入、输出缓冲区均应在池中分配。
- 注册和注销应使用同一设备、同一HCCL后端和内存池。注销前需等待通信及设备操作完成；随后释放Tensor和内存池，最后销毁进程组。
<!-- npu="A3,950" id1 -->
- 仅<term>Atlas A3系列产品</term>和<term>Ascend 950PR&950DT系列产品</term>支持HCCL对称内存池。
<!-- end id1 -->
