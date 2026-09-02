# 队列清空接口列表

本附录为 [队列清空](host_taskqueue_parallel_delivery.md#队列清空) 的完整接口列表，供排查时查阅。

## 触发队列清空接口列表

| 接口名 | 接口作用 |
|------|------|
| `torch.npu.synchronize` | 设备同步的语义要求设备上所有算子下发和执行完成 |
| `torch.npu.Stream.synchronize` | 流同步的语义要求流上所有算子下发和执行完成 |
| `torch.npu._C._npu_getCurrentRawStream` | 获取aclrt stream触发隐式同步 |
| `torch.npu.Stream.npu_stream` | 获取流，触发隐式清空队列 |
| `torch.npu.Stream._as_parameter_` | 获取流属性，触发隐式清空队列 |
| `torch.npu.Stream.__hash__()` | 获取流属性，触发隐式清空队列 |
| `torch.npu.Stream.__repr__()` | 获取流属性，触发隐式清空队列 |
| `torch.npu.Stream.query` | 查询流上状态，触发隐式清空队列 |
| `torch.npu.set_stream_limit` | 设置指定Stream的Device资源限制 |
| `torch.npu.reset_stream_limit` | 重置指定Stream的Device资源限制，恢复默认配置 |
| `torch.npu.Event.elapsed_time` | 统计两个event间的时间需要做event同步，event在队列中 |
| `torch.npu.Event.recorded_time` | 获取NPU Event对象在设备上被记录的时间，event在队列中 |
| `torch.npu.ExternalEvent.reset` | 重置事件，触发隐式清空队列 |
| `torch.npu.memory._set_allocator_settings` | 设置allocator配置 |
| `tensor.item` | 获取元素的值触发设备同步 |
| `torch.npu.empty_cache` | 缓存释放接口 |
| `torch.npu.empty_virt_addr_cache` | 轻量化empty_cache，只释放虚拟内存 |
| `torch.npu.finalize_dump` | 结束dump |
| `torch.npu.set_op_timeout_ms` | 设置NPU上算子的执行超时时间 |
| `torch.npu.graphs.graph.__enter__` | aclgraph图捕获开始 |

## 未触发队列清空接口列表

| 接口名 | 接口作用 |
|------|------|
| `torch.npu.set_device` | 设置指定设备 |
| `torch.npu.current_device` | 获取当前设备 |
| `torch.npu.device_count` | 获取可用设备数量 |
| `torch.npu.set_stream` | 设置流 |
| `torch.npu.current_stream` | 获取当前流，返回的是NPUStream对象 |
| `torch.npu._C._npu_getCurrentRawStreamNoWait` | 获取当前aclrt stream（不等待队列清空） |
| `torch.npu.default_stream` | 获取默认流，返回的是NPUStream对象 |
| `torch.npu.Event.record` | 事件记录 |
| `torch.npu.Event.query` | 不会清空队列；若事件记录尚未完成下发，立即返回false，否则阻塞查询runtime事件状态 |
| `torch.npu.Event.synchronize` | 事件同步，不会清空队列，但是阻塞等到对应事件记录下发完成 |
| `torch.npu.graphs.graph.__exit__` | aclgraph图捕获结束 |
