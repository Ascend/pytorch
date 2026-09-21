# OOM\_SNAPSHOT\_PATH

## 功能描述

通过此环境变量可配置在内存不足报错时内存数据的保存路径。

- 未配置时，内存数据默认保存至当前路径。
- 配置时，内存数据保存至指定路径，并确保该路径已存在且对运行进程具有写入权限。

此环境变量默认为未配置。

## 配置示例

```bash
export OOM_SNAPSHOT_PATH="/home/usr/"
```

## 使用约束

必须与[OOM\_SNAPSHOT\_ENABLE](OOM_SNAPSHOT_ENABLE.md)配套使用。

## 支持的型号

<!-- npu="910" id1 -->
- <term>Atlas 训练系列产品</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 训练系列产品</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 训练系列产品</term>
<!-- end id3 -->
<!-- npu="910b" id6 -->
- <term>Atlas 800I A2 推理产品</term>
<!-- end id6 -->
<!-- npu="310p" id4 -->
- <term>Atlas 推理系列产品</term>
<!-- end id4 -->
<!-- npu="950" id5 -->
- <term>Ascend 950DT</term>
<!-- end id5 -->
