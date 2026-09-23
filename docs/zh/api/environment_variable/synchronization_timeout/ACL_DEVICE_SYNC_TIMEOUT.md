# ACL\_DEVICE\_SYNC\_TIMEOUT

## 功能描述

通过此环境变量可配置设备同步的超时时间。

单位为s，配置范围为[1,2147483]，若不配置则为设备默认同步超时时间。

## 配置示例

```shell
export ACL_DEVICE_SYNC_TIMEOUT=300
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
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>
<!-- end id4 -->
<!-- npu="950" id5 -->
- <term>Ascend 950DT系列产品</term>
<!-- end id5 -->
