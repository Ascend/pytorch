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

### Enabling Fusion on Demand in Scripts

The environment variable must be set before `import torch_npu`. After the environment variable is enabled, you can control graph-free fusion by code range in user scripts.

For example, to temporarily disable graph-free fusion for a code segment and restore the previous state after exit:

```python
from torch_npu.npu import lazy_fusion

with lazy_fusion.disabled():
    output = model_part(input)
```

You can also disable fusion at the outer layer and enable it on demand only in the specified code segment:

```python
from torch_npu.npu import lazy_fusion

with lazy_fusion.disabled():
    x = preprocess(input)
    with lazy_fusion.enabled():
        x = model_part(x)
    output = postprocess(x)
```

You can also set the switch directly. The setting remains effective until it is modified again:

```python
from torch_npu.npu import lazy_fusion

lazy_fusion.set_disable()
x = preprocess(input)

lazy_fusion.set_enable()
output = model(x)
```

`set_enable()` enables the script-side fusion switch, and `set_disable()` disables the script-side fusion switch. Neither interface accepts parameters, and the return value is `None`. Unlike `with`, direct setting does not automatically restore the state. For code segments that need to automatically restore the state on abnormal exit, use the context interfaces.

The two methods can be mixed: when the context exits, the state before entering is restored (ignoring `set_enable()` or `set_disable()` calls inside the context).

- The environment variable is the master switch. When the environment variable is not enabled, neither `lazy_fusion.enabled()` nor `lazy_fusion.set_enable()` can enable graph-free fusion alone.
- Contexts support nesting. The previous state is restored both on normal exit and on abnormal exit within the range.
- When the switch state changes, the current pending fusion graph is submitted to prevent operators from being fused across the enable/disable boundary. The interfaces do not wait for device execution to complete.
- The switch is process-level and should be called at code range boundaries. Do not build fusion graphs concurrently when switching.

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
