# TORCH\_NPU\_USE\_COMPATIBLE\_IMPL

## Function Description

This environment variable controls whether to enable the compatibility configuration. When enabled, the implementation of operator APIs is fully aligned with the native PyTorch community. This environment variable is only used to switch the underlying operators called by the APIs. The default value is "0", which disables the compatibility configuration.

- When set to "0", the compatibility configuration is disabled.
- When set to "1", the compatibility configuration is enabled.

This environment variable is provided by TorchNPU. PyTorch does not have a directly corresponding environment variable.

## Configuration Example

``` bash
export TORCH_NPU_USE_COMPATIBLE_IMPL=1
```

## Usage Constraints

- This environment variable must be configured before `import torch` to take effect.

- Currently, only `torch.nn.functional.gelu`, `torch.matmul`, `torch.nn.functional.max_pool1d`, and `torch.nn.functional.max_pool2d` are supported.

- Configuring `TORCH_NPU_USE_COMPATIBLE_IMPL` affects [torch_npu.npu.use_compatible_impl(is_enable)](https://gitcode.com/Ascend/op-plugin/blob/26.2.0/docs/en/custom_APIs/torch_npu-npu/torch_npu-npu-use_compatible_impl.md)
status. Setting `TORCH_NPU_USE_COMPATIBLE_IMPL=1` automatically configures `torch_npu.npu.use_compatible_impl(True)`.

- This environment variable applies to TorchNPU 26.0.0 and later versions.

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
