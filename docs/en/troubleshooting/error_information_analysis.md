# Error Message Analysis

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-08T10:22:43.781Z pushedAt=2026-07-08T10:47:16.872Z -->

## Analysis Process

After obtaining the error message, you can refer to the following process for self-service problem analysis, so that developers can quickly locate and resolve faults.

**Figure 1**  Error Message Analysis Process  
![figure1](../figures/error_information_analysis_process.png "Error message analysis process")

1. View the error information printed on the screen. First check the first error, then further examine the information by category based on specific details, and finally analyze the cause of the fault.
2. If the echoed information cannot definitively determine the cause of the fault, you can continue to check the plog log to assist in the analysis.

## Analysis Examples

This section uses the echoed information shown in the following figure as an example to introduce how to analyze error messages.

**Figure 2**  Echoed Information Example  
![figure2](../figures/example.png "Echoed Information Example")

1. Check the first error in the echoed information.

    ```ColdFusion
    EZ3002: 2024-11-05-22:31:29.035.909 Optype [%s] of Ops kernel [%s] is unsupported. Reason: %s.
    ```

    "EZ3002" is the CANN software error code. Users can refer to the "[Error Codes](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/maintenref/troubleshooting/troubleshooting_0225.html)" section in *CANN Troubleshooting* to perform fault analysis based on the corresponding error code information. If the source of the problem remains unclear, other echoed information can be further examined.

2. Check the Python call stack and exception information.

    **Figure 3** Python call stack  
    ![figure3](../figures/python_call_stack.png "Python call stack")

    The screen shows that `torch_npu.npu.synchronize()` was called first, followed by a failure in `torch_npu._C._npu_synchronize()`. The exception information indicates that the operator running at the time of the error was ReduceAny, based on which the corresponding faulty component can be identified. If there is no clear error indication, subsequent calls need to be examined further.

3. Check the torch_npu error code.

    ```ColdFusion
    ERR00100 PTA call acl api failed
    ```

    "ERR00100" is the torch_npu error code. If there is a clear error indication, the fault can be cleared based on the specific failure cause.

4. In addition, this indicates that torch_npu reported an error when calling the underlying interface. You can also check the plog and analyze the fault cause based on the first error in the log.

    **Figure 4** Locating the error-reporting component in the plog  
    ![figure4](../figures/locate_component_reports_error_plog.png "Locating the error-reporting component in the plog log")

    In the above printed information, the error-reporting component is ASCENDCL, and the error message is the operator DynamicGRUV2. Based on this, the corresponding abnormal component can be identified. If the faulty component still cannot be determined based on the error message, contact Huawei technical support for assistance.

> [!NOTE]
>
> If the echoed information contains a native framework error, resolve it according to the error message indication. If Ascend-related issues are involved, check other Ascend first error messages in addition to this.
