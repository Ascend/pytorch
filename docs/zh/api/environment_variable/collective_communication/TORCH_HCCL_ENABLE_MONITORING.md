# TORCH\_HCCL\_ENABLE\_MONITORING

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量控制是否启动WatchDog的心跳监控线程（HeartbeatMonitor）。该线程检查WatchDog心跳是否持续更新，并处理符合配置条件的FR转储请求。默认值为0。

- 配置为“0”：不启动心跳监控线程。

- 配置为“1”：启动心跳监控线程。

该变量不控制WatchDog线程是否启动。设为0时，WatchDog仍可按既有启动条件检查通信任务的错误、超时和完成状态。

开启监控后，如果连续检查未观察到WatchDog心跳进展，会进入诊断流程，并在未收到停止请求时终止进程；不是只输出告警后继续训练。

心跳检查间隔由`TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC`控制，默认600秒。它监测WatchDog线程是否仍有进展，不是单次通信操作的执行超时，也不是故障到进程退出的严格总时限。

即使Work队列为空，正常运行的WatchDog仍会更新心跳。

监控线程从WatchDog运行入口启动。开启[TORCH_HCCL_BLOCKING_WAIT](TORCH_HCCL_BLOCKING_WAIT.md)后，不创建WatchDog，也不会通过该入口启动心跳监控。

> [!NOTE]
>
> - 设置为0不会关闭FR内存记录；但由于不创建监控线程，不会由该线程检查心跳或处理命名管道请求。
> - 应在进程组创建前设置，建议在启动各Rank进程前统一配置。训练中修改环境变量不会切换已有监控线程。
> - 关闭监控后，WatchDog自身卡住时不再由这个线程兜底终止进程。开启监控后，诊断处理、底层接口和线程调度可能影响退出时间。

## 配置示例

只开启WatchDog心跳监控，不开启故障自动dump：

```bash
export TORCH_HCCL_BLOCKING_WAIT=0
export TORCH_HCCL_ENABLE_MONITORING=1
export TORCH_HCCL_DUMP_ON_TIMEOUT=0
export TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC=600
```

如需开启故障自动dump，还需开启[TORCH_HCCL_DUMP_ON_TIMEOUT](TORCH_HCCL_DUMP_ON_TIMEOUT.md)，并将[TORCH_FR_BUFFER_SIZE](TORCH_FR_BUFFER_SIZE.md)设置为正整数。

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
