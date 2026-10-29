# TORCH\_TRANSFER\_TO\_NPU

## Function Description

This environment variable configures whether to automatically enable the transfer\_to\_npu feature, which automatically replaces PyTorch CUDA-related APIs with corresponding NPU APIs, facilitating migration from CUDA to NPU.

- When set to "1": Enables the transfer\_to\_npu feature, automatically replacing torch.cuda-related interfaces with corresponding torch.npu interfaces, including device creation, tensor operations, memory management, and stream management.
- When set to "0" or not configured: Does not enable the transfer\_to\_npu feature, requiring users to manually use torch.npu interfaces.
- When configured with other values: A ValueError exception is thrown, indicating that only "0" or "1" is supported.

This environment variable is configured as "0" by default.

## Configuration Example

Enable the transfer\_to\_npu feature:

```bash
export TORCH_TRANSFER_TO_NPU=1
```

Disable the transfer\_to\_npu feature:

```bash
export TORCH_TRANSFER_TO_NPU=0
```

## Usage Constraints

- This environment variable must be set before importing torch. Otherwise, it will not take effect.
- For more constraints related to transfer\_to\_npu, see [(Recommended) Automatic Migration](https://www.hiascend.com/document/detail/en/ModelZoo/traditional_model_train/PyTorch/docs/en/model_migration/auto_migration.md).

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
