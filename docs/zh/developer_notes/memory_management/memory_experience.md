# TorchNPU 内存问题定位与调优指南

## 1. 内存状态监控与信息采集

通过内存状态查询可初步判断问题。建议按以下步骤定位问题根因：

- 步骤 1：执行`npu-smi info`命令，查看当前`NPU`的内存占用情况（`HBM Usage`），包括已分配内存和总内存。其中已分配内存里除了`TorchNPU`外还有驱动、`HCCL`等其他组件，也会占用内存。该功能作为代码零改动的监控策略，可以作为最基础的判断条件，判断用户网络是否需要降低`Batch Size`或者采用模型并行等策略以减少内存使用
- 步骤 2：在代码中调用`torch.npu.memory_reserved()`、`torch.npu.memory_allocated()`、`torch.npu.max_memory_allocated()`、`torch.npu.max_memory_reserved()`这些常用的内存采集接口细化分析。如需更多信息，可新增以下接口采集信息：
    - `torch.npu.memory_summary()` : 返回当前设备内存分配器的统计信息摘要（文本格式）。该摘要适用于训练过程中的周期性监控或 OOM 异常分析。参考 PyTorch 官方文档；
    - `torch.npu.memory_stats()` : 返回包含`NPU`内存分配器详细统计信息的字典（`memory_reserved()` 等接口数据为其子集）。统计信息以非负整数呈现，适用于深入分析内存分配器行为。参考`PyTorch`官方文档。

若开启`expandable_segments`，可通过观察`torch.npu.memory_allocated()`和`torch.npu.memory_reserved()`的变化曲线进行调优：

- 若`memory_reserved`曲线波动，表明内存存在压力。建议调整代码以降低内存使用，使曲线趋于平缓。此举可能提升性能，但需权衡减少内存使用带来的性能损失；
- 若`memory_reserved`曲线平缓且`memory_allocated`远小于`memory_reserved`：表明存在空闲内存，可尝试提升`Batch Size`加速训推；
- 如果`memory_reserved`曲线平缓，`memory_allocated`接近`memory_reserved`且两者都接近设备内存上限，表明内存使用已接近最优状态；
- 其余场景：需结合更多额外信息判断，如多次调参内存状态变化曲线。

如果不开启`expandable_segments`，上述曲线分析的适用性受限，需结合更多额外信息判断（如多次调参内存状态变化曲线）。

## 2. 深度问题定位与分析

若基础信息不足以定位问题，可采用以下高阶手段。使用这些手段需具备`C++`基础，并熟悉`PyTorch`内存管理机制（如调用栈、底层分配策略）。

- [内存快照特性](memory_snapshot.md)：通过`Python`和`C++`调用栈机制，判断内存调用栈是否符合预期。建议结合`PyTorch`社区文档深入理解；
- [MindStudio Insight内存调优](https://www.hiascend.com/document/detail/zh/mindstudio/2600/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/memory_tuning.md)：使用昇腾自带的`Profiling`工具也可以获取详细的内存信息，该工具和内存快照特性互补；
- `PyTorch`内存使用指南 [Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/2.12/torch_cuda_memory.html)：参考社区提供的内存调优与问题定位指导文档；
- 内存日志：执行`export TORCH_NPU_LOGS=+memory`获取内存日志，搜索`NPUCachingAllocator`以获取详细调用栈。高级用户可修改`TorchNPU`源码新增日志，并通过源码编译重新调测。

## 3. OOM问题排查与优化

`OOM（Out Of Memory）`是常见的内存问题。建议按以下步骤排查与优化：

- 调整配置或代码：如通过`PYTORCH_NPU_ALLOC_CONF`设置`max_split_size_mb`或开启`expandable_segments`，尝试规避`OOM`问题；
- 分析问题根因，寻找最优解。

典型报错示例：
```
  NPU out of memory. 
  Tried to allocate 91.44 GiB (NPU 0; 
  60.96 GiB total capacity; 
  20.00 MiB already allocated; 
  20.00 MiB current active; 
  60.51 GiB free; 
  20.00 MiB reserved in total by PyTorch). 
  If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation. 
```
若`reserved`远大于`allocated`，表明存在内存碎片化问题；若`reserved`约等于`allocated`，优先检查剩余内存大小是否足以满足用户需求。以上用例，结合分配内存大小、设备内存上限以及`reserved`约等于`allocated`的现象，可以判定为内存不够，需要优化代码解决该问题。

对于更为复杂的问题，如果需要使用内存快照采集信息，并且OOM发生时手动调用`_record_memory_history()`无法采集到信息，则建议通过`OOM_SNAPSHOT_ENABLE`环境变量自动生成快照。

### 3.1 OOM临界点识别

识别`OOM`临界点需熟悉内存管理机制及相关术语。
```
内存分配过程中，如果找不到可复用的空闲Block，会按如下步骤多次向驱动申请内存：
    1. 打印日志：No existing block found on device %d, attempting to allocate new block。
    2. 第一次尝试向驱动申请内存。 
    3. 申请失败后，打印日志：Releasing available cached blocks: size=%zu, device=%d。
    4. （可选）清空TaskQueue并同步设备，然后在对应流上按需向驱动释放部分空闲内存（满足请求的大小即可）。第二次尝试向驱动申请内存。开启max_split_size_mb选项后才执行此步骤。 
    5. 申请失败后，打印'Get a block from the existing pool failed. Try to free cached blocks and reallocate. This error log can be ignored.'。
    6. 清空 TaskQueue 并同步设备，向驱动释放所有空闲内存。 
    7. 第三次尝试向驱动申请内存。 
```
若第一次申请失败但第三次成功，虽未触发最终`OOM`，但清空`TaskQueue`、设备同步及`ACL`接口调用等操作会降低整体性能。建议根据日志、`Profiling`或内存快照提前识别该场景，通过修改配置或参数避免性能损耗及潜在的随机`OOM`。
