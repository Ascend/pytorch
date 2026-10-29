# INDUCTOR\_ASCEND\_ENABLE\_COSTMODEL

## Function Description

This environment variable controls whether to enable the CostModel pre-filtering feature of Inductor-Ascend.

When enabled, Inductor-Ascend generates TTIR for each candidate config before Triton backend precompile and calls the CostModel backend of Triton-Ascend to predict the time. The prediction results are used to reorder and filter candidate configs, reducing the number of subsequent compilations and actual profiling.

- When set to "0", "false", "no", or not configured: CostModel pre-filtering is disabled (default).
- When set to "1", "true", or "yes": CostModel pre-filtering is enabled.

> [!NOTE]
>
> - CostModel is used to pre-filter configs and does not replace subsequent compilation and actual profiling. The final usable config still depends on the precompile and subsequent autotune results.
> - If the CostModel backend is unavailable, returns abnormal results, or produces no valid prediction, CostModel pre-filtering is skipped and the original config set is used.

## Configuration Example

```bash
export INDUCTOR_ASCEND_ENABLE_COSTMODEL=1
```

## Usage Constraints

- This feature only affects scenarios where the Triton backend has multiple candidate configs. CostModel is not invoked when the number of configs is less than or equal to 1.
- Enabling this feature requires that the Triton-Ascend package with the CostModel backend is installed in the current environment.

## Supported Products

<!-- npu="950" id1 -->
<term>Ascend 950DT products</term>
<!-- end id1 -->
