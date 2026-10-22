# TORCHINDUCTOR\_CATLASS\_ENABLED\_OPS

## Function Description

This environment variable configures the list of matrix multiplication operators supported by the Catlass template library. After it is set, Inductor attempts template tuning for matmul operators of the specified types using the Catlass template library.

- Default value `"mm,addmm,bmm"`: enables Catlass for the `mm`, `addmm`, and `bmm` operators.
- Set to `"ALL"`: enables Catlass for all supported matrix multiplication operators.
- Set to an empty string: does not enable Catlass for any operator.
- Set to a comma-separated list of operator names: enables Catlass for the matrix multiplication operators in the list (options are as follows).

The configurable operator names are as follows:

| Operator Name | Description |
|:---|:---|
| `mm` | Two-dimensional matrix multiplication. |
| `addmm` | Two-dimensional matrix multiplication with bias. |
| `bmm` | Batched matrix multiplication. |
| `grouped_mm` | Grouped matrix multiplication, specific to TorchNPU. |

> [!NOTE]
>
> - This environment variable must be set before importing `torch_npu`.
> - This environment variable separates operator names with commas; space separation is not supported, and it is case-insensitive.
> - It takes effect only when `TORCHINDUCTOR_NPU_CATLASS_DIR` (specifying the Catlass template library path) and `TORCHINDUCTOR_MAX_AUTOTUNE=1` are also configured.

This variable corresponds to PyTorch's [TORCHINDUCTOR_CUTLASS_ENABLED_OPS](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py), with the same configuration method. The default values differ: the TorchNPU default is `"mm,addmm,bmm"`, while the PyTorch default is `all`.

## Configuration Example

Extending support to grouped_mm:

```bash
export TORCHINDUCTOR_CATLASS_ENABLED_OPS="mm,addmm,bmm,grouped_mm"
```

Enabling Catlass for all supported matrix multiplication operators:

```bash
export TORCHINDUCTOR_CATLASS_ENABLED_OPS="ALL"
```

## Usage Constraints

None

## Supported Products

<!-- npu="950" id1 -->
<term>Ascend 950DT products</term>
<!-- end id1 -->
