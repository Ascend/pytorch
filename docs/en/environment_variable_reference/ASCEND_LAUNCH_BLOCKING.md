# ASCEND\_LAUNCH\_BLOCKING

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:03:07.720Z pushedAt=2026-06-16T03:14:22.156Z -->

## Function Description

Controls whether to enable synchronous mode during operator execution.

During model training on Ascend NPUs, operators are executed asynchronously by default. As a result, when an error occurs during operator execution, the printed error stack information is not the actual call stack. When set to "1", operators are forced to run in synchronous mode, which prints the correct call stack information, making it easier to debug and locate issues in the code. When set to "0", operators are executed asynchronously.

The default value is 0.

## Configuration Example

```bash
export ASCEND_LAUNCH_BLOCKING=1
```

## Usage Constraints

- When `ASCEND_LAUNCH_BLOCKING` is set to "1", forcing operators to run in synchronous mode will cause performance degradation.
- When `ASCEND_LAUNCH_BLOCKING` is set to "1", the task_queue operator queue is disabled, and the [TASK\_QUEUE\_ENABLE](TASK_QUEUE_ENABLE.md) setting does not take effect.

- When `ASCEND_LAUNCH_BLOCKING` is set to "0", it will increase memory consumption and pose a risk of causing OOM.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
