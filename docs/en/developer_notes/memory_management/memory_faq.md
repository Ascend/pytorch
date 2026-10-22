# FAQ

## Inconsistent Memory Usage Results between the npu-smi info Command and the torch.npu.memory_allocated() and torch.npu.memory_reserved() APIs

### Problem Description

When users check NPU memory usage with the npu-smi info command and the torch.npu.memory_allocated() and torch.npu.memory_reserved() APIs, the results are inconsistent.

### Possible Causes

Inconsistent memory query results are normal and are mainly caused by differences in the statistics scope and levels of the tools/APIs:

- `npu-smi info`: Shows the total memory usage at the NPU hardware level (HBM-Usage). It has the widest statistics scope, including not only the memory occupied by the TorchNPU framework but also the memory occupied by underlying system components such as the driver and HCCL communication components.
- `torch.npu.memory_reserved()`: Queries the total amount of memory managed by the TorchNPU cache allocator (Cache Allocator). It includes the memory allocated to tensors and the memory that has been cached but not yet released. Its scope is smaller than npu-smi but larger than the actual usage.
- `torch.npu.memory_allocated()`: Queries the NPU memory size actually occupied by tensors on the current device (in bytes). It only reflects the amount of memory currently being directly used by business logic, excluding cache and underlying overhead.

**Size Relationship**:

`npu-smi info` query result > `torch.npu.memory_reserved()` query result > `torch.npu.memory_allocated()` query result.

**Case Description**:

After `torch.npu.empty_cache()` is executed, the `memory_reserved()` value decreases, but the usage shown by npu-smi info may still be high. This is because `empty_cache` only releases the memory managed by the TorchNPU cache allocator and cannot release the memory occupied by underlying components such as the driver and HCCL. Therefore, it is normal for npu-smi to still show usage.

### Solution

This is a normal system mechanism and requires no additional handling.
