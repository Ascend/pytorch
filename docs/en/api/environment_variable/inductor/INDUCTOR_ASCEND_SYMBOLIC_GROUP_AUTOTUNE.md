# INDUCTOR\_ASCEND\_SYMBOLIC\_GROUP\_AUTOTUNE

## Function Description

This environment variable controls whether to enable grouped autotune for dynamic shapes.

When enabled, for Triton kernels with dynamic shape dimensions, Inductor-Ascend divides runtime shapes into different groups (buckets) by shape characteristics (such as dimension length). Each group performs one autotune benchmark with a representative shape, and at runtime the config is selected according to the group to which the actual shape belongs. Compared with autotuning each shape separately, this significantly reduces the repeated compilation and tuning overhead in dynamic shape scenarios, achieving "tune once, reuse across multiple shapes".

- Default value "0": disables grouped autotune.
- Set to "1", "true", or "yes": enables grouped autotune.

## Configuration Example

```bash
export INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE=1
```

## Usage Constraints

- It only takes effect for Triton kernels with dynamic shape dimensions; the behavior of static shape kernels is not affected.
- It only takes effect for pointwise, reduction, and persistent_reduction template kernels. The template types involved in tuning can be controlled by `INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES`.
- This variable is currently a grayscale switch. When the grouping plan is not supported, or the memory usage of the grouped benchmark exceeds the budget (controlled by `INDUCTOR_ASCEND_SYMBOLIC_GROUP_MAX_BENCHMARK_MEMORY_RATIO`), it automatically falls back to the normal autotune process.

## Supported Products

<!-- npu="950" id1 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id1 -->
