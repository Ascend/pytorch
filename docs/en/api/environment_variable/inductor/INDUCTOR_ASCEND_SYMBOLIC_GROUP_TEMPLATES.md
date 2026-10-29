# INDUCTOR\_ASCEND\_SYMBOLIC\_GROUP\_TEMPLATES

## Function Description

This environment variable configures the list of template types that participate in grouped autotune for dynamic shapes. Multiple templates are separated by commas, and the default value is `pointwise,reduction,persistent_reduction`.

After `INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE=1` is set to enable grouped autotune, only the template kernels in the list go through the grouped autotune process; the other template kernels still go through the normal autotune process. This is used to control the range of kernels participating in grouped tuning during grayscale release.

- When not configured: uses the default value `pointwise,reduction,persistent_reduction`, meaning all three types of template kernels participate in grouped autotune, equivalent to no range restriction.
- Set to `pointwise`: element-wise template kernels participate in grouped autotune.
- Set to `reduction`: reduction template kernels participate in grouped autotune.
- Set to `persistent_reduction`: persistent reduction template kernels participate in grouped autotune.

## Configuration Example

Enabling grouped autotune only for pointwise template kernels:

```bash
export INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE=1
export INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES=pointwise
```

## Usage Constraints

- This variable only takes effect when `INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE=1`.
- The value must be any combination of `pointwise`, `reduction`, and `persistent_reduction`; other values are ignored.
- This variable is currently a grayscale switch, used to control the rollout range of grouped autotune.

## Supported Products

<!-- npu="950" id1 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id1 -->
