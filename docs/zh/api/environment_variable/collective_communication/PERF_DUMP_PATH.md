# PERF\_DUMP\_PATH

## 功能描述

通过此环境变量可配置HCCL性能dump文件的输出目录路径。启用性能记录后，collective性能记录会输出到该目录下的`perf_pt_<pid>_<device>.log`文件中。

默认值：空。

> [!CAUTION]
>
> `PERF_DUMP_PATH`必须能通过`realpath`解析，否则在`PERF_DUMP_CONFIG=enable:true`时collective热路径会抛出错误。
>

该环境变量由TorchNPU提供，PyTorch没有直接对应的环境变量。

## 配置示例

```bash
export PERF_DUMP_PATH=/data/perf_logs
```

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Ascend 950DT</term>
