# ASCEND\_LAUNCH\_BLOCKING

## Function Description

This environment variable controls whether to enable synchronous mode during operator execution.

During model training on Ascend NPUs, operators are executed asynchronously by default. As a result, when an error occurs during operator execution, the printed error stack information is not the actual call stack information. When set to "1", operators are forced to run in synchronous mode, which prints the correct call stack information, making it easier to debug and locate issues in the code. When set to "0", operators are executed asynchronously.

The default configuration is "0".

This environment variable corresponds to [CUDA_LAUNCH_BLOCKING](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html) in PyTorch and has the same function. Both are used to control whether operators are executed in synchronous mode.

## Configuration Example

```bash
export ASCEND_LAUNCH_BLOCKING=1
```

## Usage Constraints

- When `ASCEND_LAUNCH_BLOCKING` is set to "1", forcing operators to run in synchronous mode causes performance degradation.
- When `ASCEND_LAUNCH_BLOCKING` is set to "1", the task\_queue operator queue is disabled, and the [TASK\_QUEUE\_ENABLE](TASK_QUEUE_ENABLE.md) setting does not take effect.
- When `ASCEND_LAUNCH_BLOCKING` is set to "0", it increases memory consumption and carries a risk of OOM.

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
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>
<!-- end id4 -->
<!-- npu="950" id5 -->
- <term>Ascend 950DT products</term>
<!-- end id5 -->
