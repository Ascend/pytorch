# 常见问题排查

## 问题1：调用栈不准确

**问题描述**

TaskQueue环境变量设置为“1”或“2”时，异步操作会导致报错发生位置与主线程的调用点解耦，当Host API在算子下发线程执行时发生报错，主线程获取的Python调用栈可能无法准确反映错误发生位置，导致调用栈信息不准确的情况。

| 错误发生位置 | GPU报错堆栈是否准确 | NPU报错堆栈是否准确（值为1时） |
|------|------|------|
| Python、Aten | 是 | 是 |
| Host API | 是 | 否 |
| device kernel | 否 | 否 |

**处理方法**

- **Host API层报错**（典型为`aclnnXxx`执行报错）：关闭TaskQueue以获取准确调用栈。

```shell
# 关闭TaskQueue
export TASK_QUEUE_ENABLE=0
```

- **device kernel层报错**：开启强制同步模式（关闭TaskQueue+device synchronize）以明确调用栈。

```shell
# 开启强制同步模式
export ASCEND_LAUNCH_BLOCKING=1
```

## 问题2：队列清空

**问题描述**

队列清空是指CPU侧将缓存在队列中的所有算子完成下发的动作（不一定完成device执行），属于高消耗操作，对Host性能影响较大。

常见Python接口按是否触发队列清空分为两类，完整列表见[队列清空接口列表](#队列清空接口列表)。简要归类如下：

- **触发清空**：设备/流同步、流对象隐式转换（`npu_stream`、`__hash__`、`__repr__`等）、Event时间统计、内存池配置、`tensor.item`、`empty_cache`、dump结束、算子超时设置、aclgraph图捕获开始等。
- **未触发清空**：设备查询与设置、流设置、`current_stream`/`default_stream`获取NPUStream对象、`Event.record`、`Event.query`、`Event.synchronize`、aclgraph图捕获结束等。

**处理方法**

对延迟敏感的关键路径，应避免频繁调用会触发队列清空的接口。若必须获取`aclrtStream`可以采用以下方法：

- 使用`torch.npu._C._npu_getCurrentRawStreamNoWait`。
- 使用显式调用 stream(false) 获取 aclrtStream，不触发TaskQueue清空：

    ```cpp
    // 通过stream(false) 获取aclrtStream，不触发TaskQueue清空
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    ```

### 队列清空接口列表

**触发队列清空接口列表**

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

**未触发队列清空接口列表**

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
