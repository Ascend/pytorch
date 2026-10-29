# TORCH\_DEVICE\_BACKEND\_AUTOLOAD

## Function Description

This environment variable controls whether the out-of-tree device backend is automatically loaded when `import torch` is executed. This variable is defined and parsed by PyTorch (introduced since PyTorch 2.5.0). When TorchNPU is installed via pip, a loading entry is registered under the `torch.backends` entry point group to integrate with this mechanism.

- When not configured or set to "1": `import torch` automatically imports torch_npu and registers the NPU backend. You do not need to explicitly execute `import torch_npu`.
- When set to "0" (or any other value): automatic loading is disabled. Before using the NPU, you need to explicitly execute `import torch_npu`.

This variable corresponds to [TORCH_DEVICE_BACKEND_AUTOLOAD](https://docs.pytorch.org/docs/stable/miscellaneous_environment_variables.html) in PyTorch. The variable name, default value, and value semantics are consistent.

> [!NOTE]
>
> - During the torch_npu import process, this variable is temporarily set to "0" to avoid circular dependency with `import torch`. After the import is complete, the original value is automatically restored, and child processes inherit the restored value.
> - CANN device initialization uses a lazy loading mechanism: `import torch` only imports torch_npu and registers the NPU backend. Initialization is performed only when an NPU-related interface is called for the first time. Device-related environment variables (such as `ASCEND_RT_VISIBLE_DEVICES`) must be set before the first NPU use.

## Configuration Example

To disable automatic loading and explicitly import torch_npu instead (applicable to scenarios where you need to explicitly control the import and CANN initialization timing):

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

## Usage Constraints

- This environment variable must be set before the process starts (before the first `import torch`). Changes made while the process is running do not take effect.
- This environment variable enables automatic loading only when the value is strictly equal to "1". When not configured, automatic loading is enabled by default. Any other value (such as "true" or "on") is regarded as disabled.
- This mechanism depends on the device backend automatic loading capability of PyTorch 2.5.0 and later versions, and torch_npu must be installed via pip (a loading entry is automatically registered under the `torch.backends` group during installation). When the source code directory is directly referenced (for example, by setting PYTHONPATH), there is no installation metadata, and automatic loading does not take effect.
- When multiple accelerator backends are installed at the same time (for example, the GPU version of PyTorch and torch_npu), automatic loading reports an error at `import torch`. In this case, you can set the variable to "0" to avoid automatically loading torch_npu and use other accelerators. If you need to use the NPU, install the CPU version of PyTorch.

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
<!-- npu="950" id4 -->
- <term>Ascend 950DT products</term>
<!-- end id4 -->
