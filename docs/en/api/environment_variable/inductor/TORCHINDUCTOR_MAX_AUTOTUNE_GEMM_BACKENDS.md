# TORCHINDUCTOR\_MAX\_AUTOTUNE\_GEMM\_BACKENDS

## Function Description

This environment variable configures the list of candidate implementations for matrix multiplication (matmul) operators to participate in tuning during max autotune. Here, "implementation" refers to the generation source of the matmul kernel (such as the ATen operator library, Triton template, or Catlass template), which is different from the compilation mode (Triton, DVM, etc.) configured by `TORCHINDUCTOR_NPU_BACKEND`. It only takes effect when `TORCHINDUCTOR_MAX_AUTOTUNE=1`.

- Default value `"ATEN,TRITON,CPP"`: on NPU, the candidate implementations actually participating in tuning are the ATen operator library and the Triton template (`CPP` is only for CPU devices).
- Set to a comma-separated list of implementation names: customizes the combination of candidate implementations (options are as follows).

The candidate implementation options are as follows:

| Implementation Name | Description |
|:---|:---|
| `ATEN` | ATen operator library |
| `TRITON` | Triton template in Inductor |
| `CATLASS` | Catlass template, specific to TorchNPU; requires configuring `TORCHINDUCTOR_NPU_CATLASS_DIR` |

> [!NOTE]
>
> - This environment variable must be set before importing `torch_npu`.
> - This environment variable separates implementation names with commas; space separation is not supported, and it is case-insensitive.
> - After adding `"CATLASS"`, `TORCHINDUCTOR_NPU_CATLASS_DIR` must also be configured to specify the Catlass template library path.

This variable corresponds to PyTorch's [TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py), with the same configuration method.

## Configuration Example

Trying the CATLASS and ATen candidate implementations:

```bash
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="CATLASS,ATEN"
```

Using only the Triton template for tuning:

```bash
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="TRITON"
```

## Usage Constraints

It only takes effect when `TORCHINDUCTOR_MAX_AUTOTUNE=1`.

## Supported Products

<!-- npu="950" id1 -->
<term>Ascend 950DT products</term>
<!-- end id1 -->
