# MULTI\_STREAM\_MEMORY\_REUSE

## Function Description

This environment variable configures whether to enable multi-stream memory reuse. In multi-stream scenarios involving collective communication, it optimizes the multi-stream memory management of TorchNPU, preventing delayed release of collective communication input and output memory in multi-stream scenarios and reducing peak memory usage.

- 0: Disable memory reuse.
- 1: Enable memory reuse. Based on the `eraseStream` method, it erases previous `recordStream` marks to ensure memory reuse, holds a weak reference to the tensor, and does not extend the lifecycle of the tensor.
- 2: Enable memory reuse. Based on a method that does not execute `recordStream` marks, it ensures memory reuse capability, holds a strong reference to the tensor, and may extend the lifecycle of the tensor. It is currently not recommended.
- 3: Enable memory reuse. Based on the value "1", it performs further reuse optimization and can erase `recordStream` marks in scenarios where tensors are released early.

The default value is 1.

## Configuration Example

```bash
export MULTI_STREAM_MEMORY_REUSE=0
```

## Usage Constraints

None

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas 800I A2 inference products</term>
- <term>Ascend 950DT</term>
