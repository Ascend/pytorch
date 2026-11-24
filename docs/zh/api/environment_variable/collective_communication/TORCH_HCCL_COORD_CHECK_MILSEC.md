# TORCH\_HCCL\_COORD\_CHECK\_MILSEC

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量配置全局进程组心跳监控线程检查共享诊断请求和有效命名管道的轮询间隔。单位为毫秒，默认值为1000，即1秒。建议配置为正整数。

该间隔的实际生效方式取决于监控场景：

- 自动转储开启时，全局进程组monitor按此间隔检查本地和远端故障请求及提前采集请求，但Store查询还受队列最近进度、经过时间等条件限制，无法确保每轮都执行；

- 自动转储关闭时，若已配置有效管道且FR缓冲区大于0，仍按此间隔轮询手动管道请求，若无有效管道，则可能改按心跳超时间隔等待。

为避免同一进程中的多个进程组重复轮询，共享诊断信号和管道由全局进程组monitor统一处理，各进程组的Work检查队列仍保持独立。因此按默认值1000配置时，仅表示满足条件的monitor每秒醒来检查一次，并非每秒转储一次。转储请求与心跳检查的完整关系参见[WatchDog](../../../developer_notes/fault_diagnosis/watchdog.md)。

> [!NOTE]
>
> - 该变量不启动监控线程。需先开启[TORCH_HCCL_ENABLE_MONITORING](TORCH_HCCL_ENABLE_MONITORING.md)，否则没有对应的轮询。
> - 应在进程组创建前设置，建议在各Rank启动前统一配置。
> - 缩短间隔可能加快请求响应，但也会增加线程唤醒和满足条件时的Store查询频率。

## 配置示例

按默认的1秒间隔检查协调转储请求：

```bash
export TORCH_HCCL_BLOCKING_WAIT=0
export TORCH_HCCL_ENABLE_MONITORING=1
export TORCH_HCCL_DUMP_ON_TIMEOUT=1
export TORCH_FR_BUFFER_SIZE=2000
export TORCH_HCCL_COORD_CHECK_MILSEC=1000
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
