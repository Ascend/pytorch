# 错误码信息

## ERR\*\*001

该错误码代表存在无效传参。“\*\*”表示不同模块，例如ERR00001，“ERR”后面的“00”代表PTA模块，ERR00001代表在PTA侧存在无效参数。

**Symptom**

\[%s\] ERR00001 PTA invalid parameter.

**Possible Cause**

The input parameters are invalid.

**Solution**

Check if the input parameters are as expected.

## ERR\*\*002

该错误码代表参数类型错误。“\*\*”表示不同模块，ERR01002代表在OPS算子侧存在错误的参数类型，ERR04002代表在profiling模块中存在错误的参数类型。

**Symptom**

1. \[%s\] ERR01002 OPS invalid type.
2. \[%s\] ERR04002 PROF invalid type.

**Possible Cause**

The dtype of the parameter is not as expected.

**Solution**

Find the error line and check the parameter's dtype.

## ERR\*\*003

该错误码代表数值错误。“\*\*”表示不同模块，例如ERR01003代表在OPS算子侧参数的值、shape或维度等存在错误。

**Symptom**

\[%s\] ERR01003 OPS invalid value.

**Possible Cause**

The value of input parameters is not as expected.

**Solution**

Check the input's value, shape, dimension and so on.

## ERR\*\*004

该错误码表示存在无效指针。“\*\*”表示不同模块，例如ERR01004代表OPS算子侧存在无效指针参数。

**Symptom**

\[%s\] ERR01004 OPS invalid pointer.

**Possible Cause**

Some pointer variables are not expected.

**Solution**

Submit an issue to request for support at  [https://gitcode.com/Ascend/pytorch/issues](https://gitcode.com/Ascend/pytorch/issues).

## ERR\*\*005

该错误码表示存在内部异常。“\*\*”表示不同模块，例如ERR02005代表在分布式DIST模块存在异常。

**Symptom**

1. \[%s\] ERR02005 DIST internal error.
2. \[%s\] ERR00005 PTA internal error.

**Possible Cause**

There are some errors in the corresponding module (e.g., DIST, PTA).

**Solution**

Submit an issue to request for support at  [https://gitcode.com/Ascend/pytorch/issues](https://gitcode.com/Ascend/pytorch/issues).

## ERR\*\*006

该错误码表示存在内存错误。“\*\*”表示不同模块，例如ERR00006代表PTA框架侧存在内存错误。

**Symptom**

\[%s\] ERR00006 memory error.

**Possible Cause**

There are exceptions in memory usage.

**Solution**

According to the error information, fix the error, or submit an issue to request for support at  [https://gitcode.com/Ascend/pytorch/issues](https://gitcode.com/Ascend/pytorch/issues).

## ERR\*\*007

该错误码表示特性不支持。“\*\*”表示不同模块，例如ERR00007代表调用了框架不支持的接口。

**Symptom**

\[%s\] does not support \[%s\]. ERR00007 PTA feature not supported.

**Possible Cause**

Some features are not supported.

**Solution**

According to the error information, use other similar functions instead, or submit an issue to request for support at  [https://gitcode.com/Ascend/pytorch/issues](https://gitcode.com/Ascend/pytorch/issues).

## ERR\*\*008

该错误码表示找不到相关资源。“\*\*”表示不同模块，例如ERR00008代表PTA框架侧找不到相关资源。

**Symptom**

\[%s\] ERR00008 PTA resource not found.

**Possible Cause**

Some files are not found.

**Solution**

1. Check if the execution environment is configured correctly.
2. Check if the required files exist.

## ERR\*\*009

该错误码表示资源不可用。“\*\*”表示不同模块，例如ERR04009代表PROF模块存在资源不可用的问题。

**Symptom**

1. \[%s\] ERR04009 PROF resource unavailable.
2. \[%s\] ERR00009 PTA resource unavailable.

**Possible Cause**

Some required resources are unavailable.

**Solution**

According to the error information, fix the resource problem.

## ERR\*\*010

该错误码表示系统调用错误。“\*\*”表示不同模块，例如ERR00010代表PTA框架侧存在系统调用错误。

**Symptom**

\[%s\] ERR00010 PTA system call failed.

**Possible Cause**

System call returns some errors.

**Solution**

Submit an issue to request for support at  [https://gitcode.com/Ascend/pytorch/issues](https://gitcode.com/Ascend/pytorch/issues).

## ERR\*\*011

该错误码表示超时访问。“\*\*”表示不同模块，例如ERR02011代表分布式DIST模块存在超时访问。

**Symptom**

\[%s\] ERR02011 DIST timeout error.

**Possible Cause**

1. Unexpected situations occur during the communication link.

2. The operations on different nodes are inconsistent.

**Solution**

1. Find the error line from the log, and fix it.

2. Check if the timeout variable settings are reasonable.

## ERR\*\*012

该错误码表示存在权限错误。“\*\*”表示不同模块，例如ERR00012代表PTA侧存在权限错误。

**Symptom**

\[%s\]. ERR00012 PTA permission error.

**Possible Cause**

The permissions of the file or directory have some problems.

**Solution**

According to the error information, modify the permissions of the file or directory specified in the error message.

## ERR\*\*100

该错误码表示ACL接口调用错误。“\*\*”表示不同模块，例如ERR00100代表PTA侧调用ACL接口出现错误。

**Symptom**

\[%s\] ERR00100 PTA call acl api failed.

**Possible Cause**

Calling the ACL API returns some errors.

**Solution**

Check CANN-related errors in logs and find the CANN Error Code.

> [!NOTE]  
> CANN软件相关报错的详细介绍请参见《[CANN 故障处理](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/troubleshooting/troubleshooting_0001.html)》。

## ERR\*\*200

该错误码表示HCCL接口调用错误。“\*\*”表示不同模块，例如ERR02200代表分布式DIST模块调用HCCL接口出现错误。

**Symptom**

\[%s\] ERR02200 DIST call hccl api failed.

**Possible Cause**

Calling the HCCL API returns some errors.

**Solution**

Check CANN-related errors in logs and find the CANN Error Code.

> [!NOTE]  
> CANN软件相关报错的详细介绍请参见《[CANN 故障处理](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/troubleshooting/troubleshooting_0001.html)》。

## ERR\*\*300

该错误码表示GE接口调用错误。“\*\*”表示不同模块，例如ERR00300代表PTA侧调用GE接口出现错误。

**Symptom**

\[%s\] ERR00300 PTA call ge api failed.

**Possible Cause**

Calling the GE API returns some errors.

**Solution**

Check CANN-related errors in logs and find the CANN Error Code.

> [!NOTE]  
> CANN软件相关报错的详细介绍请参见《[CANN 故障处理](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/troubleshooting/troubleshooting_0001.html)》。

## ERR\*\*999

该错误码表示应用异常。“\*\*”表示不同模块，例如ERR99999代表未知模块存在应用异常。

**Symptom**

\[%s\] ERR99999 \[%s\] application exception.

**Possible Cause**

The code of the application or a third-party library throws an exception.

**Solution**

Check the application logs or stack trace information to locate the specific exception code line. Verify whether the relevant dependency library versions are compatible, and fix the business logic or configuration issues that caused the exception.
