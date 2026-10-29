# TORCHINDUCTOR\_MAX\_AUTOTUNE

## Function Description

This environment variable controls whether to enable the max autotune feature. When enabled, Inductor performs more comprehensive auto-tuning on candidate configs to obtain the optimal kernel configuration.

- Default value "0": disables max autotune.
- Set to "1": enables max autotune.

> [!NOTE]
>
> - This environment variable must be set before importing `torch_npu`.
> - Enabling max autotune increases compilation time but may achieve better kernel performance.
> - The tunable candidate implementations for matrix multiplication operators are controlled by `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS`.

This variable corresponds to PyTorch's [TORCHINDUCTOR_MAX_AUTOTUNE](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_inductor_profiling.html), with the same configuration method.

## Configuration Example

Enabling max autotune:

```bash
export TORCHINDUCTOR_MAX_AUTOTUNE=1
```

## Usage Constraints

None

## Supported Products

<!-- npu="910" id1 -->
- <term>Atlas training products</term>
<!-- end id1 -->
<!-- npu="950" id2 -->
- <term>Ascend 950DT products</term>
<!-- end id2 -->
