## TORCHINDUCTOR_CATLASS_ENABLED_OPS （同社区TORCHINDUCTOR_CUTLASS_ENABLED_OPS）

## 功能描述

catlass可作用于的矩阵乘类的算子，该环境变量与社区保持一致，社区环境变量为TORCHINDUCTOR_CUTLASS_ENABLED_OPS。

目前可支持的算子类型有：
mm,addmm,bmm,grouped_mm

默认配置为TORCHINDUCTOR_CATLASS_ENABLED_OPS="mm,addmm,bmm"

## 配置示例

对mm,addmm,bmm,grouped_mm均尝试开启catlass后端功能：

```shell
export TORCHINDUCTOR_CATLASS_ENABLED_OPS="mm,addmm,bmm,grouped_mm"
```

## 使用约束

无

## 支持的型号

-   <term>Atlas A5 系列产品</term>