# HCCL\_ASYNC\_ERROR\_HANDLING

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可控制是否开启异步错误处理。

-   0：不开启异步错误处理。
-   1：开启异步错误处理。

当PyTorch版本为1.11.0时，默认值为0；当PyTorch版本大于等于2.1.0时，默认值为1。

> [!NOTE]  
> 当前版本，开启异步处理时，若出现ERROR CQE错误，进程会终止；其他错误信息，仅屏显信息提示，不会终止进程。

## 配置示例

```
export HCCL_ASYNC_ERROR_HANDLING=1
```

## 使用约束

通过此环境变量开启异步错误处理时，为了更好地明确HCCL超时原因，建议new\_group和init\_process\_group传参的timeout时间大于HCCL\_CONNECT\_TIMEOUT和HCCL\_EXEC\_TIMEOUT环境变量配置的时间，HCCL\_CONNECT\_TIMEOUT具体参考《CANN 环境变量参考》中的“HCCL\_CONNECT\_TIMEOUT”章节，HCCL\_EXEC\_TIMEOUT具体请参考《CANN 环境变量参考》中的“HCCL\_EXEC\_TIMEOUT”章节。

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>
-   <term>Atlas 推理系列产品</term>

