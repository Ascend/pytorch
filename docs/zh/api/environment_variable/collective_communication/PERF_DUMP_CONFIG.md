# PERF\_DUMP\_CONFIG

## 功能描述

通过此环境变量可配置HCCL操作的性能数据记录功能。格式为分号分隔的`key:value`键值对。只有`enable:true`精确匹配时才会启用性能记录，该环境变量默认值未设置。

- 配置为`enable:true`：启用collective性能记录。
- 其他值或未配置：不启用collective性能记录。

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export PERF_DUMP_CONFIG=enable:true
```

## 使用约束

配置为`PERF_DUMP_CONFIG=enable:true`后，需同时设置`PERF_DUMP_PATH`为可通过`realpath`解析的目录路径（如`/data/perf_logs`），否则collective热路径会抛出错误。若`PERF_DUMP_PATH`为空或无效路径，请先设置该环境变量。

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
