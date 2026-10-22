# Error Code Information

## ERR\*\*001

This error code indicates that invalid parameters are transferred. "\*\*" indicates different modules. For example, in ERR00001, "00" after "ERR" indicates the PTA module, and ERR00001 indicates that invalid parameters exist on the PTA side.

**Symptom**

\[%s\] ERR00001 PTA invalid parameter.

**Possible Cause**

The input parameters are invalid.

**Solution**

Check if the input parameters are as expected.

## ERR\*\*002

This error code indicates a parameter type error. "\*\*" indicates different modules. ERR01002 indicates an incorrect parameter type on the OPS operator side, and ERR04002 indicates an incorrect parameter type in the profiling module.

**Symptom**

1. \[%s\] ERR01002 OPS invalid type.
2. \[%s\] ERR04002 PROF invalid type.

**Possible Cause**

The dtype of the parameter is not as expected.

**Solution**

Find the error line and check the parameter's dtype.

## ERR\*\*003

This error code indicates a value error. "\*\*" indicates different modules. For example, ERR01003 indicates an error in the value, shape, or dimension of a parameter on the OPS operator side.

**Symptom**

\[%s\] ERR01003 OPS invalid value.

**Possible Cause**

The value of input parameters is not as expected.

**Solution**

Check the input's value, shape, dimension and so on.

## ERR\*\*004

This error code indicates that an invalid pointer exists. "\*\*" indicates different modules. For example, ERR01004 indicates an invalid pointer parameter on the OPS operator side.

**Symptom**

\[%s\] ERR01004 OPS invalid pointer.

**Possible Cause**

Some pointer variables are not expected.

**Solution**

Submit an issue to request for support at  [https://gitcode.com/Ascend/pytorch/issues](https://gitcode.com/Ascend/pytorch/issues).

## ERR\*\*005

This error code indicates an internal exception. "\*\*" indicates different modules. For example, ERR02005 indicates an exception in the distributed DIST module.

**Symptom**

1. \[%s\] ERR02005 DIST internal error.
2. \[%s\] ERR00005 PTA internal error.

**Possible Cause**

There are some errors in the corresponding module (e.g., DIST, PTA).

**Solution**

Submit an issue to request for support at  [https://gitcode.com/Ascend/pytorch/issues](https://gitcode.com/Ascend/pytorch/issues).

## ERR\*\*006

This error code indicates a memory error. "\*\*" indicates different modules. For example, ERR00006 indicates a memory error on the PTA framework side.

**Symptom**

\[%s\] ERR00006 memory error.

**Possible Cause**

There are exceptions in memory usage.

**Solution**

According to the error information, fix the error, or submit an issue to request for support at  [https://gitcode.com/Ascend/pytorch/issues](https://gitcode.com/Ascend/pytorch/issues).

## ERR\*\*007

This error code indicates that the feature is not supported. "\*\*" indicates different modules. For example, ERR00007 indicates that an interface not supported by the framework is called.

**Symptom**

\[%s\] does not support \[%s\]. ERR00007 PTA feature not supported.

**Possible Cause**

Some features are not supported.

**Solution**

According to the error information, use other similar functions instead, or submit an issue to request for support at  [https://gitcode.com/Ascend/pytorch/issues](https://gitcode.com/Ascend/pytorch/issues).

## ERR\*\*008

This error code indicates that the related resource cannot be found. "\*\*" indicates different modules. For example, ERR00008 indicates that the related resource cannot be found on the PTA framework side.

**Symptom**

\[%s\] ERR00008 PTA resource not found.

**Possible Cause**

Some files are not found.

**Solution**

1. Check if the execution environment is configured correctly.
2. Check if the required files exist.

## ERR\*\*009

This error code indicates that the resource is unavailable. "\*\*" indicates different modules. For example, ERR04009 indicates that a resource is unavailable in the PROF module.

**Symptom**

1. \[%s\] ERR04009 PROF resource unavailable.
2. \[%s\] ERR00009 PTA resource unavailable.

**Possible Cause**

Some required resources are unavailable.

**Solution**

According to the error information, fix the resource problem.

## ERR\*\*010

This error code indicates a system call error. "\*\*" indicates different modules. For example, ERR00010 indicates a system call error on the PTA framework side.

**Symptom**

\[%s\] ERR00010 PTA system call failed.

**Possible Cause**

System call returns some errors.

**Solution**

Submit an issue to request for support at  [https://gitcode.com/Ascend/pytorch/issues](https://gitcode.com/Ascend/pytorch/issues).

## ERR\*\*011

This error code indicates a timeout access. "\*\*" indicates different modules. For example, ERR02011 indicates a timeout access in the distributed DIST module.

**Symptom**

\[%s\] ERR02011 DIST timeout error.

**Possible Cause**

1. Unexpected situations occur during the communication link.

2. The operations on different nodes are inconsistent.

**Solution**

1. Find the error line from the log, and fix it.

2. Check if the timeout variable settings are reasonable.

## ERR\*\*012

This error code indicates a permission error. "\*\*" indicates different modules. For example, ERR00012 indicates a permission error on the PTA side.

**Symptom**

\[%s\]. ERR00012 PTA permission error.

**Possible Cause**

The permissions of the file or directory have some problems.

**Solution**

According to the error information, modify the permissions of the file or directory specified in the error message.

## ERR\*\*100

This error code indicates an ACL interface invocation error. "\*\*" indicates different modules. For example, ERR00100 indicates an error when the PTA side calls the ACL interface.

**Symptom**

\[%s\] ERR00100 PTA call acl api failed.

**Possible Cause**

Calling the ACL API returns some errors.

**Solution**

Check CANN-related errors in logs and find the CANN Error Code.

> [!NOTE]  
> For details about the errors related to the CANN software, see *[CANN Fault Handling](https://gitcode.com/cann/docs/blob/9.2.0/docs/en/troubleshooting/troubleshooting_intro.md)*.

## ERR\*\*200

This error code indicates an HCCL interface invocation error. "\*\*" indicates different modules. For example, ERR02200 indicates an error when the distributed DIST module calls the HCCL interface.

**Symptom**

\[%s\] ERR02200 DIST call hccl api failed.

**Possible Cause**

Calling the HCCL API returns some errors.

**Solution**

Check CANN-related errors in logs and find the CANN Error Code.

> [!NOTE]  
> For details about the errors related to the CANN software, see *[CANN Fault Handling](https://gitcode.com/cann/docs/blob/9.2.0/docs/en/troubleshooting/troubleshooting_intro.md)*.

## ERR\*\*300

This error code indicates a GE interface invocation error. "\*\*" indicates different modules. For example, ERR00300 indicates an error when the PTA side calls the GE interface.

**Symptom**

\[%s\] ERR00300 PTA call ge api failed.

**Possible Cause**

Calling the GE API returns some errors.

**Solution**

Check CANN-related errors in logs and find the CANN Error Code.

> [!NOTE]  
> For details about the errors related to the CANN software, see *[CANN Fault Handling](https://gitcode.com/cann/docs/blob/9.2.0/docs/en/troubleshooting/troubleshooting_intro.md)*.

## ERR\*\*999

This error code indicates an application exception. "\*\*" indicates different modules. For example, ERR99999 indicates an application exception in an unknown module.

**Symptom**

\[%s\] ERR99999 \[%s\] application exception.

**Possible Cause**

The code of the application or a third-party library throws an exception.

**Solution**

Check the application logs or stack trace information to locate the specific exception code line. Verify whether the relevant dependency library versions are compatible, and fix the business logic or configuration issues that caused the exception.
