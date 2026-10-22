# INDUCTOR\_ASCEND\_COSTMODEL\_RATIO

## Function Description

This environment variable controls the proportion of configs retained after CostModel pre-filtering. When CostModel is enabled, Inductor-Ascend sorts candidate configs by the predicted time returned by CostModel and retains the configs with the shortest predicted time for the subsequent precompile process. The value range is (0, 1]. When the value is less than or equal to 0 or greater than 1, it falls back to the default value "0.25".

- When not configured: uses the default value "0.25", retaining the 25% of candidate configs with the shortest CostModel-predicted time for the subsequent precompile process.
- When set to a value in (0,1): retains configs with shorter predicted time in the CostModel prediction results by proportion.
- When set to "1": CostModel pre-filtering is not performed.

> [!NOTE]
>
> - The smaller the retention ratio, the fewer configs for subsequent compilation and actual profiling, and the lower the first compilation overhead is usually, but the actual optimal config may be missed.
> - If none of the configs selected by CostModel can be compiled successfully, Inductor-Ascend uses the configs filtered out by CostModel for fallback compilation.

## Configuration Example

```bash
export INDUCTOR_ASCEND_ENABLE_COSTMODEL=1
export INDUCTOR_ASCEND_COSTMODEL_RATIO=0.25
```

## Usage Constraints

This variable only takes effect when `INDUCTOR_ASCEND_ENABLE_COSTMODEL=1`.

## Supported Products

<!-- npu="950" id1 -->
<term>Ascend 950DT products</term>
<!-- end id1 -->
