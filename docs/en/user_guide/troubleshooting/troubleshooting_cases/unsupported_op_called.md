# Unsupported Operator Called

## Symptom

Keyword "Warning: CAUTION: The operator 'xxx' is not currently supported on the NPU backend and will fall back to run on the CPU. This may have performance implications. \(function npu\_cpu\_fallback\)"

```text
[W compiler_depend.ts:51] Warning: CAUTION: The operator 'aten::linalg_lstsq.out' is not currently supported on the NPU backend and will fall back to run on the CPU. This may have performance implications. (function npu_cpu_fallback)
Traceback (most recent call last):
  File "temp_1.py", line 4, in <module>
    torch.linalg.lstsq(torch.randn(1, 3, 3).npu(), torch.randn(2, 3, 3).npu())
RuntimeError: _copy_from_and_resize now only support copy with same size!
[ERROR] 2024-11-28-11:37:15 (PID:6547, Device:0, RankID:-1) ERR01007 OPS feature not supported
```

## Cause Analysis

When the model runs, the system prints the error code "ERR01007".

An operator that is not yet supported on the NPU is called.

## Solution

If only an alarm is triggered without an error, you can leave it unhandled when performance improvement is not considered. In other cases, use other replaceable and supported torch interfaces. For details, see *[Custom APIs](https://gitcode.com/Ascend/op-plugin/blob/26.2.0/docs/en/custom_APIs/overview.md)* or *[Native APIs](../../../api/native_api/pytorch_2-14-0/overview.md)*.
