# 调用不支持的算子

## 问题现象描述

关键字"Warning: CAUTION: The operator 'xxx' is not currently supported on the NPU backend and will fall back to run on the CPU. This may have performance implications. \(function npu\_cpu\_fallback\)"

```ColdFusion
[W compiler_depend.ts:51] Warning: CAUTION: The operator 'aten::linalg_lstsq.out' is not currently supported on the NPU backend and will fall back to run on the CPU. This may have performance implications. (function npu_cpu_fallback)
Traceback (most recent call last):
  File "temp_1.py", line 4, in <module>
    torch.linalg.lstsq(torch.randn(1, 3, 3).npu(), torch.randn(2, 3, 3).npu())
RuntimeError: _copy_from_and_resize now only support copy with same size!
[ERROR] 2024-11-28-11:37:15 (PID:6547, Device:0, RankID:-1) ERR01007 OPS feature not supported
```

## 原因分析

当模型运行时，屏显信息会打印该错误码“ERR01007”。

调用了在NPU上还不支持的算子。

## 解决措施

如果只是触发告警无报错，在不考虑改善性能的情况下可不处理；其他情况，请改用torch其他可替换并支持的接口，具体可参考《[Ascend Extension for PyTorch 自定义API参考](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/overview.md)》或《[PyTorch 原生API支持度](../native_apis/pytorch_2-9-0/overview.md)》。


