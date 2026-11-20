# TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS （同社区）

## 功能描述

确认max autotune可尝试的后端有哪些，该环境变量与社区max autotune设置可尝试后端的环境变量一致。

默认配置为TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="ATEN,TRITON,CPP", 此默认值为社区的默认配置。

## 配置示例

在max autotune中仅尝试TRITON后端。

```shell
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="TRITON"
```

## 使用约束

无

## 支持的型号

<term>Ascend 950DT系列产品</term>
