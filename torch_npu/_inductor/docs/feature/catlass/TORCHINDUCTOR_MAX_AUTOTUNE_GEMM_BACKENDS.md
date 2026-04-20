## TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS （同社区）

## 功能描述

确认max autotune可尝试的后端有哪些，该环境变量与社区max autotune设置可尝试后端的环境变量一致。

若想尝试Catlass的后端，请在该环境变量中配置上"CATLASS"。

默认配置为TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="ATEN,TRITON,CPP", 此默认值为社区的默认配置。

## 配置示例

尝试在max autotune中使用ATEN和CATLASS的后端。

```shell
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="CATLASS,ATEN"
```

## 使用约束

无

## 支持的型号

-   <term>Atlas A5 系列产品</term>