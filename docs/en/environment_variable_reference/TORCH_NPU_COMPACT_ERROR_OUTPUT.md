# TORCH\_NPU\_COMPACT\_ERROR\_OUTPUT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:06:13.167Z pushedAt=2026-06-16T03:14:22.389Z -->

## Feature Description

This environment variable enables compact error message printing. When enabled, custom error information such as the internal CANN call stack and Ascend Extension for PyTorch error codes is redirected to plog, retaining only the effective error description to improve the readability of exception information. For details about plog, see [Plog Information](../troubleshooting/plog_log.md).

- When set to 0: Normal ERROR printing is performed.

- When set to 1: Compact ERROR printing is enabled.

The default value of this environment variable is 0.

## Configuration Example

```bash
export TORCH_NPU_COMPACT_ERROR_OUTPUT=1
```

## Usage Constraints

None

## Supported Products

- <term>Atlas training series</term>
- <term>Atlas A2 training series</term>
- <term>Atlas A3 training series</term>

## Output Example

- When set to the default value 0, normal ERROR printing is performed:

    ```python
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

- When set to 1, compact printing is enabled:

    ```python
    >>> torch_npu.npu.set_device(100)
    ......
    RuntimeError: CANN error: The argument is invalid.Reason: Set device failed, invalid device, set drv device=100, valid device range is [0, 8) Solution: 1.Check the input parameter range of the function. 2.Check the function invocation relationship.
    ```
