# STREAMS\_PER\_DEVICE

## 功能描述

通过此环境变量可配置stream pool的最大流数。

stream pool采用Round Robin策略。

-   配置为32时：stream pool有32条流。
-   配置为8时：stream pool有8条流。
-   配置为其他值时：打印Warning级别日志预警，并配置为默认值32。

此环境变量默认值为32。

## 配置示例

```
export STREAMS_PER_DEVICE=8
```

## 使用约束

无

## 支持的型号

-   <term>Atlas 训练系列产品</term>
-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>

