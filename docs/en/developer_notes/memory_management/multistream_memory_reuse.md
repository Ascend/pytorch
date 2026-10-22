# Multi-Stream Memory Reuse

## Introduction

A stream is an important mechanism in PyTorch, where each stream represents a strictly sequential execution logic. Generally, PyTorch launches multiple streams during execution to parallelize model communication and computation tasks. During execution, each stream requests memory from the device as needed, which is referred to as the memory pool of the stream. If the memory pool requested by one stream needs to be used by another stream, communication between the two streams is required. The completion of data execution corresponding to the memory block of the current stream serves as the signal that the block can be used by the other stream. When host operators are dispatched too quickly, the operators on the computation stream do not have enough time to reuse the memory on the communication stream. In particular, when operators on the communication stream have dependencies, all of them must finish before the memory is released for reuse.

The logic of multi-stream reuse is to release memory on the communication stream earlier so that the computation stream can reuse it. This is achieved by finding the right moment to erase the cross-stream dependency markers added during the `recordStream` call in memory management. The `work.wait()` call dispatches an event wait task to block the computation task. From the device execution perspective, the wait block is only lifted after the collective communication operator completes execution. At this point, the memory that the collective communication operator depends on has already been fully used by the collective communication task, and the cross-stream dependency markers are no longer needed to delay the release. Therefore, they can be erased directly. Memory blocks with erased cross-stream dependency markers are returned directly to the memory pool after the tensor is destructed. After `work.wait()`, this memory block can be allocated to other operators. The subsequent operators are guaranteed to execute after the event wait task on the device, which means they use this memory block after the collective communication operator, thus avoiding concurrent operations with the collective communication operator. This is how multi-stream reuse is implemented.

The schematic diagram without multi-stream reuse enabled is shown in [Figure 1](#disabling-multi-stream-memory-reuse-schematic-diagram). When destructing tensor1 (free tensor1), the native PyTorch framework ensures that the memory of tensor1 is released only after the HCCL operator finishes using it. Therefore, after tensor1 is destructed (free tensor1), the memory is not immediately reclaimed by the memory pool. Instead, an Event3 record task is dispatched. Before the next memory allocation, the memory is only reclaimed into the memory pool after confirming that Event3 record is complete. In the diagram, malloc tensor2 cannot reuse the memory of tensor1 because the query for Event3 record completion has not yet finished, preventing the memory pool from reclaiming the memory of tensor1 in time.

**Figure 1**  Schematic diagram without multi-stream memory reuse enabled  <a id="disabling-multi-stream-memory-reuse-schematic-diagram"></a>
![](../../figures/disabling_multi_stream_memory_reuse.png)

The schematic diagram for enabling multi-stream memory reuse is shown in [Figure 2](#enabling-multi-stream-memory-reuse-schematic-diagram). After the Event1 wait task, the cross-stream dependency marker is also erased, eliminating the need to issue an Event3 record task. After tensor1 is destructed (free tensor1), the memory of tensor1 is directly reclaimed by the memory pool.

**Figure 2**  Schematic diagram with multi-stream memory reuse enabled <a id="enabling-multi-stream-memory-reuse-schematic-diagram"></a>
![](../../figures/enabling_multi_stream_memory_reuse.png)  

## Application Scenarios

In LLM distributed training scenarios, consider using this feature to avoid out-of-bounds memory access and improve the cross-stream memory reuse rate.

## Usage Guide

Use the environment variable `MULTI_STREAM_MEMORY_REUSE` to control whether multi-stream memory reuse is enabled.

- 0: Disable memory reuse.
- 1: Enable memory reuse. Based on the `eraseStream` method, it erases previous `recordStream` marks to ensure memory reuse, holds a weak reference to the tensor, and does not extend the lifecycle of the tensor.
- 2: Enables memory reuse. Based on the method that does not perform recordStream marking, memory reuse is guaranteed. A strong reference to the tensor is held, which may extend the tensor lifecycle.
- 3: Enable memory reuse. Based on the value "1", it further optimizes reuse and can erase the `recordStream` marks in scenarios where tensors are released early.

The default value is 2.

For details on using this environment variable, refer to the "[MULTI_STREAM_MEMORY_REUSE](../../api/environment_variable/memory_management/MULTI_STREAM_MEMORY_REUSE.md)" section in *Environment Variable Reference*.

## Usage Examples

Disable memory reuse:

```bash
export MULTI_STREAM_MEMORY_REUSE=0
```

## Constraints

None
