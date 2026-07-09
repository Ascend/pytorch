# (Beta) INDUCTOR\_ASCEND\_CHECK\_ACCURACY

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:03:49.560Z pushedAt=2026-06-16T03:14:22.203Z -->

## Feature Description

`INDUCTOR_ASCEND_CHECK_ACCURACY` is an accuracy verification tool provided by Ascend Extension for PyTorch. It automatically detects the numerical accuracy of fused operators when the torch.compile graph compilation backend is "inductor".

This tool captures the FX subgraph corresponding to a fused operator, generates an independently executable single-operator test case, and compares the output differences between eager mode and the fused operator under the same input conditions. When the difference exceeds a preset threshold, it automatically outputs accuracy verification failure logs and diagnostic information, helping developers quickly locate accuracy issues.

## Configuration Example

Example 1: Enable the accuracy tool with default accuracy threshold configuration.

```bash
export INDUCTOR_ASCEND_CHECK_ACCURACY=1
```

**Table 1** Default accuracy thresholds

| Data Type | Relative Error (rtol) | Absolute Error (atol) |
|:---:|:---:|:---:|
| float32 | 1.3e-6 | 1e-5 |
| float16 | 1e-3 | 1e-5 |
| bfloat16 | 1.6e-2 | 1e-5 |
| Others | 1.3e-6 | 1e-5 |

Example 2: Enable the accuracy tool and set accuracy comparison thresholds.

```bash
export INDUCTOR_ASCEND_CHECK_ACCURACY=1
# Set the relative error threshold to 1e-6 and the absolute error threshold to 1e-7 for accuracy comparison
export INDUCTOR_ASCEND_CHECK_ACCURACY_RTOL_ATOL="rtol=1e-6,atol=1e-7"
```

> [!CAUTION]
>
> If you need to configure different accuracy thresholds based on data types (such as float32, float16, bfloat16, etc.), manually modify the `acc_comp_tol` dictionary in the source code of different backends. The configuration file paths for each backend are as follows:
>
> - Triton: [config](../../../torch_npu/_inductor/config.py)
> - MLIR and DVM: [config](../../../torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/config.py)

## Usage Constraints

- This environment variable is only available in PyTorch 2.7.1 and PyTorch 2.9.0.
- This environment variable can be used when the torch.compile graph compilation backend is "inductor".

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
