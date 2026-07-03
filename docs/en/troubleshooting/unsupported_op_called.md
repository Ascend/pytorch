# Unsupported Operators Called

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-12T08:25:13.524Z pushedAt=2026-06-12T11:22:41.058Z -->

## Symptom

Keyword "Warning: CAUTION: The operator 'xxx' is not currently supported on the NPU backend and will fall back to run on the CPU. This may have performance implications. \(function npu\_cpu\_fallback\)"

```ColdFusion
[W compiler_depend.ts:51] Warning: CAUTION: The operator 'aten::linalg_lstsq.out' is not currently supported on the NPU backend and will fall back to run on the CPU. This may have performance implications. (function npu_cpu_fallback)
Traceback (most recent call last):
  File "temp_1.py", line 4, in <module>
    torch.linalg.lstsq(torch.randn(1, 3, 3).npu(), torch.randn(2, 3, 3).npu())
RuntimeError: _copy_from_and_resize now only support copy with same size!
[ERROR] 2024-11-28-11:37:15 (PID:6547, Device:0, RankID:-1) ERR01007 OPS feature not supported
```

## Possible Cause

When the model is running, the system prints the error code **ERR01007**.

An operator that is not yet supported on the NPU is called.

## Solution

If only a warning is triggered without an error, you can ignore it if performance improvement is not a concern. Otherwise, use other replaceable and supported interfaces of torch. For details, see [Ascend Extension for PyTorch Custom API Reference](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/overview.md) or [Supported PyTorch Native APIs](../native_apis/pytorch_2-10-0/overview.md).
