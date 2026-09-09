# Error Code Information

## ERR\*\*001

This error code indicates that an invalid parameter is passed. "\*\*" represents different modules. For example, in ERR00001, "00" after "ERR" represents the PTA module, meaning that an invalid parameter exists on the PTA side.

**Symptom**

\[%s\] ERR00001 PTA invalid parameter.

**Possible Cause**

The input parameters are invalid.

**Solution**

Check if the input parameters are as expected.

## ERR\*\*002

This error code indicates a parameter type error. "\*\*" represents different modules. ERR01002 indicates an invalid parameter type on the OPS side, and ERR04002 indicates an invalid parameter type in the profiling module.

**Symptom**

1. \[%s\] ERR01002 OPS invalid type.
2. \[%s\] ERR04002 PROF invalid type.

**Possible Cause**

The dtype of the parameter is not as expected.

**Solution**

Find the error line and check the parameter's dtype.

## ERR\*\*003

This error code indicates a value error. "\*\*" represents different modules. For example, ERR01003 indicates errors in parameter values, shapes, or dimensions on the OPS operator side.

**Symptom**

\[%s\] ERR01003 OPS invalid value.

**Possible Cause**

The value of input parameters is not as expected.

**Solution**

Check the input's value, shape, dimension and so on.

## ERR\*\*004

This error code indicates that an invalid pointer exists. "\*\*" represents different modules. For example, ERR01004 indicates that an invalid pointer parameter exists on the OPS operator side.

**Symptom**

\[%s\] ERR01004 OPS invalid pointer.

**Possible Cause**

Some pointer variables are not expected.

**Solution**

Submit an issue to request for support at  [https://gitcode.com/Ascend/pytorch/issues](https://gitcode.com/Ascend/pytorch/issues).

## ERR\*\*005

This error code indicates an internal exception. "\*\*" represents different modules. For example, ERR02005 indicates an exception in the DIST module.

**Symptom**

1. \[%s\] ERR02005 DIST internal error.
2. \[%s\] ERR00005 PTA internal error.

**Possible Cause**

There are some errors in the corresponding module (for example, DIST, PTA).

**Solution**

Submit an issue to request for support at  [https://gitcode.com/Ascend/pytorch/issues](https://gitcode.com/Ascend/pytorch/issues).

## ERR\*\*006

This error code indicates a memory error. "\*\*" represents different modules. For example, ERR00006 indicates a memory error on the PTA framework side.

**Symptom**

\[%s\] ERR00006 memory error.

**Possible Cause**

There are exceptions in memory usage.

**Solution**

According to the error information, fix the error, or submit an issue to request for support at  [https://gitcode.com/Ascend/pytorch/issues](https://gitcode.com/Ascend/pytorch/issues).

## ERR\*\*007

This error code indicates that a feature is not supported. "\*\*" represents different modules. For example, ERR00007 indicates that an unsupported interface of the framework is called.

**Symptom**

\[%s\] does not support \[%s\]. ERR00007 PTA feature not supported.

**Possible Cause**

Some features are not supported.

**Solution**

According to the error information, use other similar functions instead, or submit an issue to request for support at  [https://gitcode.com/Ascend/pytorch/issues](https://gitcode.com/Ascend/pytorch/issues).

## ERR\*\*008

This error code indicates that the related resource cannot be found. "\*\*" represents different modules. For example, ERR00008 indicates that the resource cannot be found on the PTA framework side.

**Symptom**

\[%s\] ERR00008 PTA resource not found.

**Possible Cause**

Some files are not found.

**Solution**

1. Check if the execution environment is configured correctly.
2. Check if the required files exist.

## ERR\*\*009

This error code indicates that a resource is unavailable. "\*\*" represents different modules. For example, ERR04009 indicates that the PROF module has a resource unavailable issue.

**Symptom**

1. \[%s\] ERR04009 PROF resource unavailable.
2. \[%s\] ERR00009 PTA resource unavailable.

**Possible Cause**

Some required resources are unavailable.

**Solution**

According to the error information, fix the resource problem.

## ERR\*\*010

This error code indicates a system call error. "\*\*" represents different modules. For example, ERR00010 indicates a system call error on the PTA framework side.

**Symptom**

\[%s\] ERR00010 PTA system call failed.

**Possible Cause**

System call returns some errors.

**Solution**

Submit an issue to request for support at  [https://gitcode.com/Ascend/pytorch/issues](https://gitcode.com/Ascend/pytorch/issues).

## ERR\*\*011

This error code indicates a timeout access. "\*\*" represents different modules. For example, ERR02011 indicates a timeout access in the DIST module.

**Symptom**

\[%s\] ERR02011 DIST timeout error.

**Possible Cause**

1. Unexpected situations occur during the communication link.

2. The operations on different nodes are inconsistent.

**Solution**

1. Find the error line from the log, and fix it.

2. Check if the timeout variable settings are reasonable.

## ERR\*\*012

This error code indicates a permission error. "\*\*" represents different modules. For example, ERR00012 indicates a permission error on the PTA side.

**Symptom**

\[%s\]. ERR00012 PTA permission error.

**Possible Cause**

The permissions of the file or directory have some problems.

**Solution**

According to the error information, modify the permissions of the file or directory specified in the error message.

## ERR\*\*100

This error code indicates an ACL API call error. "\*\*" represents different modules. For example, ERR00100 indicates an error when calling the ACL API on the PTA side.

**Symptom**

\[%s\] ERR00100 PTA call acl api failed.

**Possible Cause**

Calling the ACL API returns some errors.

**Solution**

Check CANN-related errors in logs and find the CANN Error Code.

> [!NOTE]  
> For details about CANN software-related errors, see the CANN Troubleshooting.
<!-- see the [CANN Troubleshooting](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/maintenref/troubleshooting/troubleshooting_0001.html) Troubleshooting. -->

## ERR\*\*200

This error code indicates an HCCL API call error. "\*\*" represents different modules. For example, ERR02200 indicates that the distributed DIST module encountered an error when calling the HCCL API.

**Symptom**

\[%s\] ERR02200 DIST call hccl api failed.

**Possible Cause**

Calling the HCCL API returns some errors.

**Solution**

Check CANN-related errors in logs and find the CANN Error Code.

> [!NOTE]  
> For details about CANN software-related errors, see the CANN Troubleshooting.
<!-- see the [CANN Troubleshooting](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/maintenref/troubleshooting/troubleshooting_0001.html) Troubleshooting. -->

## ERR\*\*300

This error code indicates a GE API call error. "\*\*" represents different modules. For example, ERR00300 indicates an error when calling the GE API on the PTA side.

**Symptom**

\[%s\] ERR00300 PTA call ge api failed.

**Possible Cause**

Calling the GE API returns some errors.

**Solution**

Check CANN-related errors in logs and find the CANN Error Code.

> [!NOTE]  
> For details about CANN software-related errors, see the CANN Troubleshooting.
<!-- see the [CANN Troubleshooting](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/maintenref/troubleshooting/troubleshooting_0001.html)Troubleshooting. -->

## ERR\*\*999

This error code indicates an application exception. "\*\*" represents different modules. For example, ERR99999 indicates an application exception in an unknown module.

**Symptom**

\[%s\] ERR99999 \[%s\] application exception.

**Possible Cause**

The code of the application or a third-party library throws an exception.

**Solution**

Check the application logs or stack trace information to locate the specific exception code line. Verify whether the relevant dependency library versions are compatible, and fix the business logic or configuration issues that caused the exception.
