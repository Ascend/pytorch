# HCCL\_EVENT\_TIMEOUT

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可设置等待Event完成的超时时间。

一个进程内，调用acl.init接口初始化pyACL后，调用acl.rt.set\_op\_wait\_timeout接口设置超时时间，本进程内后续调用acl.rt.stream\_wait\_event接口下发的任务支持在所设置的超时时间内等待，若等待的时间超过所设置的超时时间，则pyACL会返回报错。

单位为s，取值范围为\[0, 2147483647\]，默认值为1868，当配置为0时代表永不超时。

> [!NOTE]  
> -   acl.init接口详情具体请参见《CANN  应用开发接口》中“[函数：init](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/appdevgapi/aclpythondevg_01_0005.html)”章节。
> -   acl.rt.set\_op\_wait\_timeout接口详情具体请参见《CANN   应用开发接口》中“[函数：set\_op\_wait\_timeout](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/appdevgapi/aclpythondevg_01_0102.html)”章节。
> -   acl.rt.stream\_wait\_event接口详情具体请参见《CANN   应用开发接口》中“[函数：stream\_wait\_event](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/appdevgapi/aclpythondevg_01_0101.html)”章节。

## 配置示例

```
export HCCL_EVENT_TIMEOUT=1800
```

## 使用约束

该环境变量被配置时，配置值需要大于HCCL\_EXEC\_TIMEOUT的配置值，HCCL\_EXEC\_TIMEOUT具体可参考《CANN 环境变量参考》中的“[HCCL\_EXEC\_TIMEOUT](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0078.html)”章节。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>

