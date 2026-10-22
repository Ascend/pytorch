# PYTORCH\_HAL\_BASED\_NPU\_CHECK

## Function Description

This environment variable configures the NPU availability detection method used by `torch_npu.npu.is_available()`.

- When set to "1": `torch_npu.npu.is_available()` probes the number of devices through the HAL interface (using the same path as `torch_npu.npu.device_count()`).
  - Dynamically loads `libascend_hal.so`.
  - Parses the `ASCEND_RT_VISIBLE_DEVICES` environment variable.
  - If HAL probing fails, detection falls back to the Runtime method.
- When set to other values or not configured: `torch_npu.npu.is_available()` queries the number of devices through the Runtime interface.
  - The first non-zero device count obtained by the query is cached.
  - Effectively reduces the overhead of frequently calling `is_available()` before NPU initialization.

This environment variable is not configured by default.

The naming and value semantics of this variable are designed with reference to PyTorch's [PYTORCH_NVML_BASED_CUDA_CHECK](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html), remaining consistent in functional mode and usage logic to facilitate developer migration and understanding.

## Configuration Example

Enable the HAL-based detection path:

```bash
export PYTORCH_HAL_BASED_NPU_CHECK=1
```

## Usage Constraints

- This environment variable must be set before the first call to `torch_npu.npu.is_available()`.
- This environment variable takes effect only when its value is exactly "1". If other values are configured (such as "true" or "on"), it is considered not enabled, and the corresponding feature is not triggered.

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
