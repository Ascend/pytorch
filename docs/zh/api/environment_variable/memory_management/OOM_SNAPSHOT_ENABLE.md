# OOM\_SNAPSHOT\_ENABLE

## 功能描述

通过此环境变量可配置在内存不足报错时是否保存内存数据，以供分析内存不足原因。

- 配置为“2”，当发生内存不足报错时，仅保存当前内存使用信息，包含申请和释放的内存信息。
- 配置为“1”，当发生内存不足报错时，将保存当前和历史内存使用信息，包含申请和释放的内存信息。
- 配置为“0”，关闭此功能，不保存内存数据。

此环境变量默认为0。

## 配置示例

```bash
export OOM_SNAPSHOT_ENABLE=1
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
<!-- npu="910b" id6 -->
- <term>Atlas 800I A2训练服务器</term>
<!-- end id6 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>
<!-- end id4 -->
<!-- npu="950" id5 -->
- <term>Ascend 950DT系列产品</term>
<!-- end id5 -->
