# TORCHINDUCTOR\_PROFILE\_WITH\_DO\_BENCH\_USING\_PROFILING

## Function Description

This environment variable controls whether profiling is used for performance measurement during autotune. When enabled, Inductor uses profiling to obtain more accurate kernel execution time instead of estimates based on event timing.

- Default value "0": profiling is not used; performance is estimated based on event timing.
- Set to "1": uses profiling to measure kernel execution time.

> [!NOTE]
>
> - This environment variable must be set before importing `torch_npu`.
> - Using profiling produces more accurate tuning results but increases the time overhead of autotune.

This variable corresponds to PyTorch's [TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py), with the same configuration method.

## Configuration Example

Enabling profiling-based autotune:

```bash
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING=1
```

## Usage Constraints

It only takes effect when `TORCHINDUCTOR_MAX_AUTOTUNE=1`.

## Supported Products

<!-- npu="950" id1 -->
<term>Ascend 950DT products</term>
<!-- end id1 -->
