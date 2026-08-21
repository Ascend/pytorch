# TORCH\_NPU\_LAZY\_FUSION

## Feature Description

This environment variable enables DVM operator fusion in TorchNPU Eager mode. DVM merges multiple adjacent small operators into a single fused kernel, reducing the number of kernel launches and intermediate tensor data movement, thereby accelerating training and inference.

- When set to "True": DVM operator fusion is enabled.
- When not configured or set to "False": DVM operator fusion is disabled.

This environment variable is not configured by default.

## Configuration Example

```bash
export TORCH_NPU_LAZY_FUSION=True
```

## Usage Constraints

- It takes effect only when [TASK_QUEUE_ENABLE](TASK_QUEUE_ENABLE.md) is set to 1 or 2. Otherwise, operator fusion is automatically disabled.
- It takes effect only on the main thread and its backward thread. Other independent threads (for example, dataloader workers) automatically disable operator fusion.

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
