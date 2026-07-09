# MULTI\_STREAM\_MEMORY\_REUSE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:04:29.780Z pushedAt=2026-06-16T03:14:22.254Z -->

## Feature Description

This environment variable configures whether to enable multi-stream memory reuse. In multi-stream scenarios involving collective communication, it optimizes the multi-stream memory management of Ascend Extension for PyTorch, preventing delayed release of collective communication input and output memory in multi-stream scenarios and reducing peak memory usage.

- 0: Disable memory reuse.
- 1: Enable memory reuse. Based on the `eraseStream` method, it erases previous `recordStream` marks to ensure memory reuse, holds a weak reference to the tensor, and does not extend the tensor's lifecycle.
- 2: Enable memory reuse. Based on the method of not executing `recordStream` marks, it ensures memory reuse capability, holds a strong reference to the tensor, and may extend the tensor's lifecycle. Currently not recommended.
- 3: Enable memory reuse. Based on the value "1", it performs further reuse optimization, allowing the erasure of `recordStream` marks in scenarios where tensors are released early.

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
