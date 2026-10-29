# INF\_NAN\_MODE\_FORCE\_DISABLE

## Function Description

<!-- npu="A3,910b" id5 -->
<term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>, this environment variable can be used to forcibly disable INF_NAN mode. Disabling INF_NAN mode causes Inf and NaN generated during computation to be converted to the maximum value and 0 of the corresponding data type, leading to differences in subsequent operation results. Therefore, forced interception is implemented. If you must disable INF_NAN mode, set this environment variable to "1" to forcibly disable INF_NAN mode. After disabling it, pay attention to the changes in Inf and NaN values.
<!-- end id5 -->

<!-- npu="A3,910b" id1 -->
- 1: Forcibly disables INF_NAN mode and enables saturation mode. After forcibly disabling INF_NAN mode on <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>, pay attention to the changes in Inf and NaN values.
<!-- end id1 -->
<!-- npu="A3,910b" id2 -->
- 0: Does not process INF_NAN mode. When INF_NAN mode is disabled on <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>, an error is intercepted and reported. The default value is 0.
<!-- end id2 -->

<!-- npu="950,910,310p,310b" id6 -->
For <term>Atlas training products</term>/<term>Atlas inference products</term>/<term>Atlas 200I/500 A2 inference products</term>/<term>Ascend 950DT products</term>, this environment variable does not take effect.
<!-- end id6 -->

This environment variable is provided by TorchNPU. PyTorch does not have a directly corresponding environment variable.

## Configuration Example

```bash
export INF_NAN_MODE_FORCE_DISABLE=1
```

## Usage Constraints

- This environment variable must be configured before the Python process starts. Changes made while the process is running do not take effect.
- Only "0" or "1" is supported.
- After forcibly disabling INF_NAN mode, Inf and NaN generated during computation are converted to the maximum value and 0 of the corresponding data type, which may cause operation results to be inconsistent with expectations. You are advised not to configure it except in special circumstances.

## Supported Products

<!-- npu="910b" id3 -->
- <term>Atlas A2 training products</term>
<!-- end id3 -->
<!-- npu="A3" id4 -->
- <term>Atlas A3 training products</term>
<!-- end id4 -->
