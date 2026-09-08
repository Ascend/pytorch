# FAQ

## npu-smi info命令与torch.npu.memory_allocated()、torch.npu.memory_reserved()接口查看内存占用结果不一致

### 问题描述

用户使用npu-smi info命令与torch.npu.memory_allocated()、torch.npu.memory_reserved()接口查看NPU内存占用时，发现结果不一致。

### 可能原因

内存查询结果不一致属于正常现象，主要由各工具/接口的统计范围及层级差异导致：

    - npu-smi info：查看的是NPU硬件层面的总内存占用（HBM-Usage）。其统计范围最广，不仅包含 TorchNPU 框架占用的内存，还包含驱动、HCCL 通信组件等底层系统组件占用的内存。
    - torch.npu.memory_reserved()：查询的是TorchNPU缓存分配器（Cache Allocator）管理的内存总量。它包含了已分配给张量的内存以及已缓存但尚未释放的内存，范围小于 npu-smi，但大于实际使用量。
    - torch.npu.memory_allocated()：查询的是当前设备上张量（Tensors）实际占用的 NPU 内存大小（单位：字节）。它仅反映当前正在被业务逻辑直接使用的内存量，不包含缓存及底层开销。

**大小关系**：

npu-smi info 查询结果 > torch.npu.memory_reserved() 查询结果 > torch.npu.memory_allocated() 查询结果。

**案例说明**:

执行torch.npu.empty_cache() 后，memory_reserved() 数值会下降，但npu-smi info显示的占用仍可能较高。这是因为empty_cache仅释放TorchNPU缓存分配器管理的内存，无法释放驱动、HCCL 等底层组件占用的内存，因此npu-smi仍有占用属正常现象。

### 解决措施

此现象为系统正常机制，无需进行额外处理。
