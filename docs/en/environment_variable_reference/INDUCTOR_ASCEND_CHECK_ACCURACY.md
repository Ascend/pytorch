# (Beta) INDUCTOR\_ASCEND\_CHECK\_ACCURACY

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-08T10:23:46.643Z pushedAt=2026-07-08T10:47:16.881Z -->

## Feature Description

INDUCTOR_ASCEND_CHECK_ACCURACY is an accuracy verification tool provided by Ascend Extension for PyTorch. It automatically detects the numerical accuracy of fused operators only when the torch.compile graph compilation backend is "Inductor" and the mode is "Triton".

This tool captures the FX subgraphs corresponding to fused operators, generates independently executable single-operator test cases, and compares the output differences between eager and Triton under the same input conditions. When the difference exceeds the preset threshold, it automatically outputs accuracy verification failure logs and diagnostic information, helping developers quickly locate accuracy issues.

## Configuration Example

Example 1: Enable the accuracy tool with default accuracy threshold configuration

```bash
export INDUCTOR_ASCEND_CHECK_ACCURACY=1
```

**Table 1** Default accuracy thresholds

| Data Type | Relative Error rtol | Absolute Error atol |
| :---: | :---: | :---: |
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
> If you need to configure different accuracy thresholds based on different data types (such as float32, float16, bfloat16, etc.), manually modify the `acc_comp_tol` dictionary in the source code [config](../../../torch_npu/_inductor/config.py).

## Usage Constraints

- This environment variable can only be used in PyTorch 2.7.1.

- This environment variable is available when the torch.compile graph compilation backend is "Inductor" and the mode is "Triton" (the environment variable `TORCHINDUCTOR_NPU_BACKEND` is empty or set to "default").

## Supported Products

- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
