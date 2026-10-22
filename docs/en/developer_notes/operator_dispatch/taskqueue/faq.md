# FAQs

## Problem 1: Inaccurate Call Stack

**Problem Description**

When the TaskQueue environment variable is set to "1" or "2", asynchronous operations decouple the location where an error occurs from the call point of the main thread. When a Host API error occurs during execution on the operator dispatch thread, the Python call stack obtained by the main thread may not accurately reflect the error location, resulting in inaccurate call stack information.

| Error Location | Is the GPU Error Stack Accurate? | Is the NPU Error Stack Accurate? (when the value is 1) |
|------|------|------|
| Python, Aten | Yes | Yes |
| Host API | Yes | No |
| device kernel | No | No |

**Solution**

- **Host API layer errors** (typically `aclnnXxx` execution errors): Disable TaskQueue to obtain an accurate call stack.

```shell
# Disable TaskQueue
export TASK_QUEUE_ENABLE=0
```

- **Device kernel layer errors**: Enable the forced synchronization mode (disable TaskQueue + device synchronize) to clarify the call stack.

```shell
# Enable forced synchronization mode
export ASCEND_LAUNCH_BLOCKING=1
```

## Problem 2: Queue Flushing

**Problem Description**

Queue flushing refers to the action on the CPU side of completing the dispatch of all operators cached in the queue (not necessarily completing device execution). It is a high-cost operation that significantly affects Host performance.

Common Python interfaces are divided into two categories based on whether they trigger queue flushing. For the complete list, see [Queue Flushing API List](#queue-flushing-api-list). A brief categorization is as follows:

- **Triggers flushing**: device/stream synchronization, implicit conversion of stream objects (`npu_stream`, `__hash__`, `__repr__`, etc.), Event time statistics, memory pool configuration, `tensor.item`, `empty_cache`, dump completion, operator timeout settings, aclgraph graph capture start, etc.
- **Does not trigger flushing**: device query and setting, stream setting, obtaining NPUStream objects via `current_stream`/`default_stream`, `Event.record`, `Event.query`, `Event.synchronize`, aclgraph graph capture end, etc.

**Solution**

For latency-sensitive critical paths, avoid frequently calling interfaces that trigger queue flushing. If you must obtain `aclrtStream`, you can use the following methods:

- Use `torch.npu._C._npu_getCurrentRawStreamNoWait`.
- Use an explicit call to stream(false) to obtain aclrtStream without triggering TaskQueue flushing:

    ```cpp
    // Obtain aclrtStream via stream(false) without triggering TaskQueue clearing
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    ```

### Queue Flushing API List

**APIs That Trigger Queue Flushing**

| API Name | API Function |
|------|------|
| `torch.npu.synchronize` | The semantics of device synchronization require that all operators on the device complete dispatch and execution |
| `torch.npu.Stream.synchronize` | The semantics of stream synchronization require that all operators on the stream complete dispatch and execution |
| `torch.npu._C._npu_getCurrentRawStream` | Obtains the aclrt stream, triggering implicit synchronization |
| `torch.npu.Stream.npu_stream` | Obtains the stream, triggering implicit queue flushing |
| `torch.npu.Stream._as_parameter_` | Obtains the stream attribute, triggering implicit queue flushing |
| `torch.npu.Stream.__hash__()` | Obtains the stream attribute, triggering implicit queue flushing |
| `torch.npu.Stream.__repr__()` | Obtains the stream attribute, triggering implicit queue flushing |
| `torch.npu.Stream.query` | Queries the stream status, triggering implicit queue flushing |
| `torch.npu.set_stream_limit` | Sets the Device resource limit of the specified Stream |
| `torch.npu.reset_stream_limit` | Resets the Device resource limit of the specified Stream and restores the default configuration |
| `torch.npu.Event.elapsed_time` | Measuring the time between two events requires event synchronization; the event is in the queue |
| `torch.npu.Event.recorded_time` | Obtains the time when the NPU Event object was recorded on the device; the event is in the queue |
| `torch.npu.ExternalEvent.reset` | Resets the event, triggering implicit queue flushing |
| `torch.npu.memory._set_allocator_settings` | Sets the allocator configuration |
| `tensor.item` | Obtains the element value, triggering device synchronization |
| `torch.npu.empty_cache` | Cache release interface |
| `torch.npu.empty_virt_addr_cache` | A lightweight empty_cache that only releases virtual memory |
| `torch.npu.finalize_dump` | Ends the dump |
| `torch.npu.set_op_timeout_ms` | Sets the execution timeout for operators on the NPU |
| `torch.npu.graphs.graph.__enter__` | Starts aclgraph graph capture |

**APIs That Do Not Trigger Queue Flushing**

| API Name | API Function |
|------|------|
| `torch.npu.set_device` | Sets the specified device |
| `torch.npu.current_device` | Obtains the current device |
| `torch.npu.device_count` | Obtains the number of available devices |
| `torch.npu.set_stream` | Sets the stream |
| `torch.npu.current_stream` | Obtains the current stream; returns an NPUStream object |
| `torch.npu._C._npu_getCurrentRawStreamNoWait` | Obtains the current aclrt stream (without waiting for queue flushing) |
| `torch.npu.default_stream` | Obtains the default stream; returns an NPUStream object |
| `torch.npu.Event.record` | Event recording |
| `torch.npu.Event.query` | Does not flush the queue; if the event record has not completed dispatch, returns false immediately; otherwise, blocks and queries the runtime event status |
| `torch.npu.Event.synchronize` | Event synchronization; does not flush the queue but blocks until the corresponding event record completes dispatch |
| `torch.npu.graphs.graph.__exit__` | Ends aclgraph graph capture |
