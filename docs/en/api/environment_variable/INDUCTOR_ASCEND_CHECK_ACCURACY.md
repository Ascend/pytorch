# (Beta) INDUCTOR\_ASCEND\_CHECK\_ACCURACY

## Feature Description

INDUCTOR_ASCEND_CHECK_ACCURACY is an accuracy verification tool provided by Ascend Extension for PyTorch. It automatically detects the numerical accuracy of fused operators when the torch.compile graph compilation backend is "inductor".

This tool captures the FX subgraphs corresponding to fused operators, generates independently executable single-operator test cases, and compares the output differences between eager and the fused operator under the same input conditions. When the difference exceeds the preset threshold, it automatically outputs accuracy verification failure logs and diagnostic information, helping developers quickly locate accuracy issues.

## Configuration Example

Example 1: Enable the accuracy tool with default accuracy threshold configuration

```bash
export INDUCTOR_ASCEND_CHECK_ACCURACY=1
```

**Table 1** Default accuracy thresholds

| Data Type | Relative Error rtol | Absolute Error atol |
|:---:|:---:|:---:|
| float32 | 1.3e-6 | 1e-5 |
| float16 | 1e-3 | 1e-5 |
| bfloat16 | 1.6e-2 | 1e-5 |
| Others | 1.3e-6 | 1e-5 |

Example 2: Enable the accuracy tool and set accuracy comparison thresholds

```bash
export INDUCTOR_ASCEND_CHECK_ACCURACY=1
# Set the relative error threshold to 1e-6 and the absolute error threshold to 1e-7 for accuracy comparison
export INDUCTOR_ASCEND_CHECK_ACCURACY_RTOL_ATOL="rtol=1e-6,atol=1e-7"
```

> [!CAUTION]
>
> If you need to configure different accuracy thresholds for different data types (for example, float32, float16, and bfloat16), manually modify the `acc_comp_tol` dictionary in the source code of different backends. The configuration file paths of each backend are as follows:
>
> - Triton: [config](../../../torch_npu/_inductor/config.py)
> - MLIR and DVM: [config](../../../torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/config.py)

## Usage Constraints

- This environment variable can only be used in PyTorch 2.7.1 and PyTorch 2.9.0.

- This environment variable can be used when the torch.compile graph compilation backend is "inductor".

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Ascend 950DT</term> (only Triton mode and DVM mode of the Inductor backend compiler are supported, and MLIR mode is not supported)
