# (Beta) INDUCTOR\_ASCEND\_CHECK\_ACCURACY

## Function Description

INDUCTOR_ASCEND_CHECK_ACCURACY is an accuracy check tool provided by TorchNPU that automatically checks the numerical accuracy of fused operators in torch.compile graph compilation mode (Inductor).

This tool captures the FX subgraphs corresponding to fused operators, generates independently executable single-operator test cases, and compares the output differences between eager and the fused operator under the same input conditions. When the difference exceeds the preset threshold, it automatically outputs accuracy verification failure logs and diagnostic information, helping developers quickly locate accuracy issues.

- Default value not configured: the accuracy check is not enabled.
- Set to "1": enables the accuracy check, automatically checking the numerical accuracy of fused operators.

> [!NOTE]
>
> When this environment variable is enabled, Triton mode (`TORCHINDUCTOR_NPU_BACKEND="default"`) automatically sets `torch._inductor.config.split_reductions` to `False`.

This environment variable is provided by TorchNPU; PyTorch has no directly corresponding environment variable.

## Configuration Example

Example 1: Enable the accuracy tool with default accuracy threshold configuration

```bash
export INDUCTOR_ASCEND_CHECK_ACCURACY=1
```

The default accuracy thresholds are as follows:

| Data Type | Relative Error rtol | Absolute Error atol |
|:---|:---|:---|
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

> [!NOTE]
>
> To configure different accuracy thresholds for different data types (such as float32, float16, bfloat16, etc.), manually modify the acc_comp_tol dictionary in the source code of the corresponding mode. The configuration file paths for each mode are as follows:
>
> - Triton: [config](../../../../../torch_npu/_inductor/config.py)
> - DVM: [config](../../../../../torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/config.py)

No additional information is printed when the accuracy comparison is normal. When a discrepancy is detected, a warning in the following format is printed:

```bash
CHECK ACCURACY FAILED! Kernel: <kernel_name>, Output idx: <idx>, Mismatched: <m>/<n> (<pct>%), Greatest Rel Diff: <rel>, Greatest Abs Diff: <abs>, rtol: <rtol>, atol: <atol>, dump_path: <path> 
```

When Triton mode (`TORCHINDUCTOR_NPU_BACKEND="default"`) is set, the data and fx graph files are saved to the dump_path directory shown in the printed information.

## Usage Constraints

- This environment variable can only be used in PyTorch 2.7.1 and PyTorch 2.9.0.
- This environment variable can be used in torch.compile graph compilation mode (Inductor).
- Only the Triton and DVM modes of the Inductor backend compiler are supported.

## Supported Products

<!-- npu="910" id1 -->
- <term>Atlas training products</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 training products</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT products</term>
<!-- end id4 -->
