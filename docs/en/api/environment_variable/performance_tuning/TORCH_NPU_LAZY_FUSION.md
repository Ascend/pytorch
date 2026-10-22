# TORCH\_NPU\_LAZY\_FUSION

## Function Description

This environment variable enables DVM operator fusion in TorchNPU Eager mode. DVM merges multiple adjacent small operators into a single fused kernel, reducing the number of kernel launches and intermediate tensor data movement, thereby accelerating training and inference.

- When set to "True": DVM operator fusion is enabled.
- When not configured or set to "False": DVM operator fusion is disabled.

This environment variable is not configured by default.

> [!NOTE]
>
> This environment variable is used for operator fusion in Eager mode without graph compilation.

This environment variable is provided by TorchNPU. PyTorch does not have a directly corresponding environment variable.

## Configuration Example

```bash
export TORCH_NPU_LAZY_FUSION=True
```

## Usage Constraints

- It must be set before importing `torch_npu`.
- It takes effect only when [TASK_QUEUE_ENABLE](../op_execution/TASK_QUEUE_ENABLE.md) is 1 or 2; otherwise, operator fusion is automatically disabled.
- It takes effect only on the main thread and its backward thread. Other independent threads (for example, dataloader workers) automatically disable operator fusion.
- It takes effect only on chip models supported by DVM.

## Supported Products

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>
<!-- end id2 -->
