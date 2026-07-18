# Stream-Level TaskQueue Parallel Delivery

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T07:52:19.404Z pushedAt=2026-06-15T12:00:44.106Z -->

> [!NOTICE]  
>
> This feature is currently experimental and may change in future releases.

## Introduction

Currently, when the task_queue operator delivery queue is enabled, a single-device single-TaskQueue mode is used, where all streams share the same task queue. The first-level pipeline threads (multi-threaded) submit tasks to the unified queue, and the second-level pipeline thread serially retrieves tasks from the queue for delivery. Under high-concurrency scenarios (multiple Streams submitting simultaneously), this architecture leads to queue contention, creating a serialization bottleneck in task delivery.

To address this issue, Ascend Extension for PyTorch introduces the stream-level TaskQueue parallel delivery feature. When this feature is enabled, each stream initializes an independent TaskQueue and a corresponding Dequeue thread, implementing a true second-level pipeline parallel delivery mechanism. Tasks from different streams can be delivered in parallel, effectively resolving the queue contention problem and improving task delivery efficiency in high-concurrency scenarios.

## Use Scenario

In multi-threaded multi-stream delivery scenarios, when Dequeue is blocked, it is recommended to enable this feature; it is not recommended in other scenarios.

## Usage Guide

The `PER_STREAM_QUEUE` environment variable can be used to configure whether to enable one `task_queue` operator delivery queue per stream.

- When configured as "0", the one task\_queue operator delivery queue per stream is disabled.
- When configured as "1", the one task\_queue operator delivery queue per stream is enabled.

This environment variable is configured as "0" by default.

## Usage Example

```bash
export PER_STREAM_QUEUE=1
```

## Constraints

- This feature depends on TaskQueue. It takes effect only when `TASK_QUEUE_ENABLE` is configured as "1" or "2".
- This feature does not support [Configuring Process-Level Online Recovery](https://gitcode.com/Ascend/mind-cluster/blob/branch_v26.0.0/docs/en/scheduling/usage/resumable_training/04_configuring_fault_handling_policies.md#configuring-process-level-online-recovery) scenarios.
- When this feature is enabled, OOM in the TaskQueue of a non-default stream does not immediately trigger a memory snapshot.
- When this feature is enabled, there will be multiple TaskQueues in a multi-stream scenario, corresponding to multiple threads, which may cause resource contention and affect performance.
- When this feature is enabled, if there are Event interactions between multiple streams, the first-level pipeline may incur additional overhead to ensure that Events are delivered in order in the second-level pipeline, affecting performance.
- When this feature is enabled, there may be multiple second-level pipeline threads, and fine-grained core binding is not supported.
