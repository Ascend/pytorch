# MULTI\_STREAM\_MEMORY\_REUSE

## Function Description

This environment variable configures whether to enable multi-stream memory reuse. In multi-stream scenarios involving collective communication, it optimizes the multi-stream memory management of TorchNPU, preventing delayed release of collective communication input and output memory in multi-stream scenarios and reducing peak memory usage.

- 0: Disable memory reuse.
- 1: Enable memory reuse. Based on the `eraseStream` method, it erases previous `recordStream` marks to ensure memory reuse, holds a weak reference to the tensor, and does not extend the lifecycle of the tensor.
- 2: Enables memory reuse. Based on the method that does not execute the recordStream marker, memory reuse is guaranteed, and holding strong references to tensors may extend the tensor lifecycle.
- 3: Enable memory reuse. Based on the value "1", it performs further reuse optimization and can erase `recordStream` marks in scenarios where tensors are released early.

The default value is 2.

## Configuration Example

```bash
export MULTI_STREAM_MEMORY_REUSE=0
```

## Usage Constraints

MULTI_STREAM_MEMORY_REUSE=2:

1. For asynchronous collective communication that returns a Work object, you must call the wait() method of the object.
2. The memory protection mechanism of point-to-point communication falls back to recordStream.

## Supported Products

<!-- npu="910" id1 -->
- <term>Atlas training products</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 training products</term>
<!-- end id3 -->
<!-- npu="910b" id5 -->
- <term>Atlas 800I A2 inference server</term>
<!-- end id5 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT products</term>
<!-- end id4 -->
