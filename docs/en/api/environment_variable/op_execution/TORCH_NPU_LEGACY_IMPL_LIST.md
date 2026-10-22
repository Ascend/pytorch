# TORCH\_NPU\_LEGACY\_IMPL\_LIST

## Function Description

This environment variable specifies features that need to use the old-version implementation (hereinafter referred to as the legacy implementation) through configuration items. It is applicable to scenarios where the original execution behavior needs to be retained after product upgrade or migration to new-generation hardware. When this environment variable is not configured, the related features use the default implementation of the current product.

The priority of this environment variable is higher than that of the `TORCH_NPU_USE_COMPATIBLE_IMPL` and `torch_npu.npu.use_compatible_impl()` global switches. After a configuration item is added to the list, the corresponding feature uses the legacy implementation. If it is not added to the list, the corresponding feature is not affected by this environment variable.

The currently supported configuration items are as follows:

| Configuration Item | Description |
| ------ | ---- |
| `randomness` | Controls the execution path on the NPU of native PyTorch random number APIs that support switching. |

`randomness` currently applies to the following native PyTorch APIs:

| API Category | API |
| ------- | --- |
| Random number generation APIs | `torch.rand`, `torch.rand_like`, `torch.randn`, `torch.randn_like`, `torch.randint`, `torch.randint_like`, `torch.normal` |
| Tensor in-place APIs | `Tensor.uniform_`, `Tensor.normal_`, `Tensor.random_(from, to)`, `Tensor.random_(to)` |
| Parameter initialization APIs | `torch.nn.init.uniform_`, `torch.nn.init.normal_`, `torch.nn.init.trunc_normal_`, `torch.nn.init.xavier_uniform_`, `torch.nn.init.xavier_normal_`, `torch.nn.init.kaiming_uniform_`, `torch.nn.init.kaiming_normal_`, `torch.nn.init.orthogonal_`, `torch.nn.init.sparse_` |

<!-- npu="950,A3,910b" id2 -->
On <term>Ascend 950DT products</term>, the aforementioned random number APIs use implementations fully aligned with the PyTorch native community by default. After `randomness` is configured, the related APIs switch to the random number implementation consistent with <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term>.
<!-- end id2 -->

This environment variable is provided by TorchNPU. PyTorch does not have a directly corresponding environment variable.

## Configuration Example

Configuring random number APIs to use the legacy implementation:

```bash
export TORCH_NPU_LEGACY_IMPL_LIST=randomness
```

## Usage Constraints

- This environment variable must be configured before the Python process starts.
- When configuring multiple configuration items, use English commas to separate them.
- Configuration items are case-sensitive. If an unsupported configuration item is configured, an exception is thrown, and the error message indicates the currently supported configuration items.
- `randomness` requires a CANN version that supports random number implementation switching. When CANN does not support this capability, the related APIs retain the default implementation of the current product.

## Supported Products

<!-- npu="950" id1 -->
<term>Ascend 950DT products</term>
<!-- end id1 -->
