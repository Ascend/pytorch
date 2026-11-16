# STREAMS\_PER\_DEVICE

## 功能描述

通过此环境变量可配置stream pool的最大流数。

其中，stream pool内部采用Round Robin（轮询）策略分配流，本环境变量控制该池可创建的最大流数。

- 配置为“32”时：stream pool有32条流。
- 配置为“8”时：stream pool有8条流。
- 配置为其他值时：打印Warning级别日志预警，并配置为默认值32。

此环境变量默认值为“32”。

## 配置示例

```bash
export STREAMS_PER_DEVICE=8
```

## 使用约束

无

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3训练系列产品</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT系列产品</term>
<!-- end id4 -->
