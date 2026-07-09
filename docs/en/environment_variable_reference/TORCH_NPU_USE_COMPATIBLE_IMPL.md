# TORCH_NPU_USE_COMPATIBLE_IMPL

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:06:38.626Z pushedAt=2026-06-16T03:14:22.400Z -->

## Feature Description

This environment variable controls whether to enable the compatibility configuration. When enabled, the implementation of operator APIs is fully aligned with the native PyTorch community. This environment variable is only used to switch the underlying operators called by the APIs.

- When set to "0", the compatibility configuration is disabled.

- When set to "1", the compatibility configuration is enabled.

## Configuration Example

``` bash
export TORCH_NPU_USE_COMPATIBLE_IMPL=1
```

## Usage Constraints

- This environment variable must be configured before `import torch` to take effect.

- Currently, only `torch.nn.functional.gelu`, `torch.matmul`, `torch.nn.functional.max_pool1d`, and `torch.nn.functional.max_pool2d` are supported.

- Configuring `TORCH_NPU_USE_COMPATIBLE_IMPL` affects [torch_npu.npu.use_compatible_impl(is_enable)](https://gitcode.com/Ascend/op-plugin/blob/26.0.0/docs/en/custom_APIs/torch_npu-npu/torch_npu-npu-use_compatible_impl.md)

status. Setting `TORCH_NPU_USE_COMPATIBLE_IMPL=1` automatically configures `torch_npu.npu.use_compatible_impl(True)`.

- This environment variable applies to Ascend Extension for PyTorch 26.0.0 and later versions.

## Supported Products

- <term>Atlas training products</term>

- <term>Atlas A2 training products</term>

- <term>Atlas A3 training products</term>

- <term>Atlas inference products</term>
