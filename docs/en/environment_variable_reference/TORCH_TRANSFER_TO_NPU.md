# TORCH\_TRANSFER\_TO\_NPU

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:06:53.810Z pushedAt=2026-06-16T03:14:22.402Z -->

## Feature Description

This environment variable configures whether to automatically enable the transfer\_to\_npu feature, which automatically replaces PyTorch CUDA-related APIs with corresponding NPU APIs, facilitating migration from CUDA to NPU.

- When set to "1": Enables the transfer\_to\_npu feature, automatically replacing torch.cuda-related interfaces with corresponding torch.npu interfaces, including device creation, tensor operations, memory management, and stream management.
- When set to "0" or not configured: Does not enable the transfer\_to\_npu feature, requiring users to manually use torch.npu interfaces.
- When configured with other values: A ValueError exception is thrown, indicating that only "0" or "1" is supported.

This environment variable is configured as "0" by default.

## Configuration Example

Enable the transfer_to_npu feature:

```bash
export TORCH_TRANSFER_TO_NPU=1
```

Disable the transfer_to_npu feature:

```bash
export TORCH_TRANSFER_TO_NPU=0
```

## Usage Constraints

- This environment variable must be set before importing torch; otherwise, it will not take effect.
- For more constraints related to transfer_to_npu, refer to the "[Recommended) Automatic Migration](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/recommended_auto_migration.md)" section in the *PyTorch Model Migration and Tuning Guide*.

## Supported Products

- <term>Atlas training series</term>
- <term>Atlas A2 training series</term>
- <term>Atlas A3 training series</term>
