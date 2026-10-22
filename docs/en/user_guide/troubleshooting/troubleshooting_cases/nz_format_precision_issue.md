# Precision Issue After Using the NZ Format

## Symptom

During network tuning, after the user enables [`torch.npu.config.allow_internal_format = True`](https://gitcode.com/Ascend/op-plugin/blob/26.2.0/docs/en/custom_APIs/torch_npu-npu/%EF%BC%88beta%EF%BC%89torch_npu-npu-config-allow_internal_format.md), the computation results of the model show precision anomalies.

## Cause Analysis

The Ascend NPU internally supports two tensor data layout formats: ND (N-Dimensional), a standard multi-dimensional layout format consistent with the PyTorch native format with good generality; and NZ (Z-order), a fractal layout format that reorganizes data into 16×16 blocks, improving the memory access efficiency of some operators and commonly used in network tuning scenarios.

The user code contains NZ-format tensor input. When the `torch.npu.config.allow_internal_format` switch is not enabled (default value `False`), TorchNPU implicitly converts the NZ format to the ND format before computation, which is imperceptible to users.

After the `torch.npu.config.allow_internal_format` switch is enabled, TorchNPU no longer performs implicit format conversion, and NZ-format tensors directly participate in operator computation. Because the computation paths of some operators in the NZ format differ numerically from those in the ND format, the final results show precision anomalies.

## Solution

Choose one of the following methods based on the actual scenario:

1. **Recommended**: Convert the NZ-format input data to the ND format before passing it to the model, for example:

   ```python
   input_tensor = input_tensor.float().npu_format_cast(2)  # 2 means the ND format
   ```

2. Keep the `torch.npu.config.allow_internal_format` switch disabled (keep the default value `False`).
