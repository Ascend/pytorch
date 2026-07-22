# 和内存相关的常见问题

## 使用`npu-smi info`命令查看内存占用情况和使用`torch.npu.memory_allocated()`/`torch.npu.memory_reserved()`查看内存占用情况不一致

这三种方式查到的内存占用情况不一致，是正常现象，原因如下：

- `npu-smi info`命令查看的是当前NPU的内存占用情况（HBM-Usage），包括已分配内存和总内存。其中已分配内存里除了TorchNPU外还有驱动、HCCL等其他组件，也会占用内存。
- `torch.npu.memory_allocated()`查询的是当前设备上，张量（tensors）实际占用的NPU内存大小，单位为字节，它反映的是当前正在被使用的内存量。
- `torch.npu.memory_reserved()`查询的是TorchNPU缓存分配器管理下的内存总量，单位为字节，包括了已分配和已缓存但尚未释放内存。
- 查询到的结果大小情况为：`npu-smi info`查询的HBM-Usage > `torch.npu.memory_reserved()`查询的结果 > `torch.npu.memory_allocated()`查询的结果
