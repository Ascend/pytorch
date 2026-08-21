# Precision Anomaly After Using NZ Format

## Symptom Description

During network tuning, after the user enables [`torch.npu.config.allow_internal_format = True`](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/en/custom_APIs/torch_npu-npu/%EF%BC%88beta%EF%BC%89torch_npu-npu-config-allow_internal_format.md), the model's computation results exhibit a precision anomaly.

## Cause Analysis

Ascend NPU internally supports two tensor data layout formats: ND (N-Dimensional) is the standard multi-dimensional layout format, consistent with the native PyTorch format and offering good generality; NZ (Z-order) is a fractal layout format that reorganizes data into 16×16 blocks, which can improve memory access efficiency for certain operators and is commonly used in network tuning scenarios.

User code may contain tensors in NZ format. When the `torch.npu.config.allow_internal_format` switch is not enabled (default value `False`), TorchNPU implicitly converts NZ format to ND format before computation, and users are unaware of this during usage.

After enabling the `torch.npu.config.allow_internal_format` switch, TorchNPU no longer performs implicit format conversion, and tensors in NZ format directly participate in operator computation. Since the computation paths of certain operators in NZ format differ numerically from those in ND format, the final result exhibits precision anomalies.

## Solutions

Choose one of the following methods based on your actual scenario:

1. **Recommended**: Convert the input data from NZ format to ND format before passing it to the model, for example:

   ```python
   input_tensor = input_tensor.float().npu_format_cast(2)  # 2 means ND format
   ```

2. Keep the `torch.npu.config.allow_internal_format` option disabled (retain the default value `False`).
