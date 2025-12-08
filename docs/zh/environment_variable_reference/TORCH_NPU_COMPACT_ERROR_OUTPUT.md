# TORCH\_NPU\_COMPACT\_ERROR\_OUTPUT

## 功能描述

通过此环境变量可精简打印错误信息，开启后会将CANN内部调用栈、Ascend Extension for PyTorch错误码等自定义报错信息转移到plog中，仅保留有效的错误说明，提高异常信息的可读性。

-   配置为0时：正常进行ERROR打印。
-   配置为1时：启用精简ERROR打印。

此环境变量默认值为0。

## 配置示例

```
export TORCH_NPU_COMPACT_ERROR_OUTPUT=1
```

## 使用约束

无

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>

## 结果示例

-   设置为默认值0，正常进行ERROR打印时：

    ```ColdFusion
    >>> torch_npu.npu.set_device(100)
    ......
    RuntimeError: Initialize:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:147 NPU function error: c10_npu::SetDevice(device_id_), error code is 107001
    [ERROR] 2025-08-06-16:15:27 (PID:2228607, Device:0, RankID:-1) ERR00100 PTA call acl api failed
    [Error]: Invalid device ID.
            Check whether the device ID is valid.
    EE1001: [PID: 2228607] 2025-08-06-16:15:27.377.593 The argument is invalid.Reason: Set device failed, invalid device, set drv device=100, valid device range is [0, 8)
            Solution: 1.Check the input parameter range of the function. 2.Check the function invocation relationship.
            TraceBack (most recent call last):
            rtSetDevice execute failed, reason=[device id error][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:53]
            open device 100 failed, runtime result = 107001.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
            ctx is NULL![FUNC:GetDevErrMsg][FILE:api_impl.cc][LINE:6146]
            The argument is invalid.Reason: rtGetDevMsg execute failed, reason=[context pointer null]
    ```

-   设置为1，开启精简打印时：

    ```ColdFusion
    >>> torch_npu.npu.set_device(100)
    ......
    RuntimeError: CANN error: The argument is invalid.Reason: Set device failed, invalid device, set drv device=100, valid device range is [0, 8) Solution: 1.Check the input parameter range of the function. 2.Check the function invocation relationship.
    ```

