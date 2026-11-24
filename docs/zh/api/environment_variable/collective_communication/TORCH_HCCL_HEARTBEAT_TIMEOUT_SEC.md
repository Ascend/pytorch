# TORCH\_HCCL\_HEARTBEAT\_TIMEOUT\_SEC

## 功能描述

当使用HCCL作为通信后端并开启心跳监控时，通过此环境变量配置检查WatchDog心跳进展的时间间隔。监控线程在相邻检查中没有观察到心跳计数变化时，会判断WatchDog可能卡住，进入诊断和退出处理。单位为秒，默认值为600，即10分钟。建议配置为正整数。

它检查的是WatchDog是否仍有进展，不是单次通信必须在多少秒内完成。正常运行的WatchDog即使Work队列为空也会更新心跳。

发生故障后，心跳监控线程完成诊断还会继续等待，最长等待该变量指定的秒数，给Desync报告生成或进程组关闭留出时间。等待期间，如果收到停止心跳监控线程的请求，会提前结束等待。

> [!NOTE]
>
> - 只有心跳监控线程启动后，此配置才参与心跳检查。`TORCH_HCCL_ENABLE_MONITORING`默认0，仅设置心跳超时不会启动监控线程。
> - 应在进程组创建前设置，建议在各Rank启动前统一配置。
> - 该数值不要设置得太小，WatchDog处理底层接口调用或诊断任务时可能耗时较长，如果在设定时间内没有更新心跳，监控线程就可能误以为WatchDog卡死。

## 配置示例

按默认的600秒间隔检查WatchDog心跳：

```bash
export TORCH_HCCL_BLOCKING_WAIT=0
export TORCH_HCCL_ENABLE_MONITORING=1
export TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC=600
```

如需在心跳故障诊断时尝试生成FR主文件，还需开启[TORCH_HCCL_DUMP_ON_TIMEOUT](TORCH_HCCL_DUMP_ON_TIMEOUT.md)，并将[TORCH_FR_BUFFER_SIZE](TORCH_FR_BUFFER_SIZE.md)设为正整数。

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
