# TORCH\_FR\_BUFFER\_SIZE

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可配置飞行记录器（Flight Recorder，简称FR）环形缓冲区最多保存的通信记录条数。

同一进程中的多个HCCL进程组共享FR缓冲区，条数限制适用于整个缓冲区。增大缓冲区会增加主机内存占用和转储数据量，应结合通信频率和故障前需要保留的记录范围配置。

- 0：关闭通信记录。
- 正整数：开启通信记录，并保存最近指定条数的记录。缓冲区写满后，新记录覆盖旧记录。

默认值为0，记录默认关闭。

> [!NOTE]
>
> - 配置在当前进程首次访问FR时读取，之后修改环境变量不会重新生效。初始化进程组、导出记录或重置记录均可能触发首次访问，建议在启动训练进程前设置。
> - 原生HCCL记录器不读取社区NCCL配置`TORCH_NCCL_TRACE_BUFFER_SIZE`。

## 配置示例

保存最近2000条通信记录：

```bash
export TORCH_FR_BUFFER_SIZE=2000
```

关闭通信记录：

```bash
export TORCH_FR_BUFFER_SIZE=0
```

## 使用约束

- 开启记录不会自动开启心跳监控或定期生成文件。自动故障转储需配合`TORCH_HCCL_ENABLE_MONITORING`和`TORCH_HCCL_DUMP_ON_TIMEOUT`使用。
- 取值为0时，本地诊断路径不生成FR数据文件；Python线程调用栈等其他诊断信息仍可能被采集。
- 当前版本同时兼容旧名称`TORCH_HCCL_TRACE_BUFFER_SIZE`。两个变量同时配置时，优先使用`TORCH_FR_BUFFER_SIZE`，包括配置为0的情况。

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
