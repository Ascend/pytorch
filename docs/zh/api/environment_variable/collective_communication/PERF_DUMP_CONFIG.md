# PERF\_DUMP\_CONFIG

## 功能描述

通过此环境变量可配置HCCL操作的性能数据记录功能。格式为逗号分隔的`key:value`键值对。只有`enable:true`精确匹配时才会启用性能记录。

- `enable:true`：启用collective性能记录。
- 其他配置：不启用。

默认值：未设置（不启用）。

> [!CAUTION]
>
> `PERF_DUMP_CONFIG=enable:true`后若`PERF_DUMP_PATH`为空或不是可解析的real path，collective热路径会抛出错误。该路径必须能通过`realpath`解析。
>

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export PERF_DUMP_CONFIG=enable:true
```

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
