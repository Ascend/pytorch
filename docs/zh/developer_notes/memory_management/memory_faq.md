# 和内存相关的常见问题

## `npu-smi info`与`torch.npu.memory_allocated()`/`torch.npu.memory_reserved()`查看内存占用结果不一致

这三种方式查到的内存占用情况不一致，是正常现象，原因如下：

- `npu-smi info`命令查看的是当前NPU的内存占用情况（HBM-Usage），包括已分配内存和总内存。其中已分配内存里除了TorchNPU外还有驱动、HCCL等其他组件，也会占用内存。
- `torch.npu.memory_allocated()`查询的是当前设备上，张量（tensors）实际占用的NPU内存大小，单位为字节，它反映的是当前正在被使用的内存量。
- `torch.npu.memory_reserved()`查询的是TorchNPU缓存分配器管理下的内存总量，单位为字节，包括了已分配和已缓存但尚未释放内存。
- 三种方式查询到的结果大小关系为：`npu-smi info`查询的HBM-Usage > `torch.npu.memory_reserved()`查询的结果 > `torch.npu.memory_allocated()`查询的结果（其中符号`>`是大于号）。

典型案例：`torch.npu.empty_cache()`后，用户可观察到`memory_reserved()`下降，但`npu-smi`还是有一定占用：这是正常现象，`torch.npu.empty_cache()`仅释放TorchNPU缓存分配器管理下的内存，无法释放驱动、HCCL等组件占用。
