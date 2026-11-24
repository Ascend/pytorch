# TORCH\_HCCL\_WAIT\_TIMEOUT\_DUMP\_MILSEC

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量配置常规自动故障转储的等待时间。单位为毫秒，TorchNPU 2.15版本默认值为15000，即15秒。建议配置为正整数。

WatchDog发出Work故障转储请求后，按配置值的4倍等待，默认等待60秒，再继续后续错误处理。心跳监控线程发起异步转储后，每轮按配置值等待转储结果，默认最多等待15秒。

WatchDog仅在监控与常规自动转储均开启时，为这条故障转储路径广播请求并额外等待。

首轮异步转储未成功完成时，monitor会尝试第二轮不包含FR条目调用栈的转储，并再次使用该等待时间。

转储前提和离线分析请参见[飞行记录器](../../../developer_notes/fault_diagnosis/flight_recorder.md)。

> [!NOTE]
>
> - 等待超时不会取消前一次转储任务。该值不限制采集线程的全部执行时间，也不是诊断、资源清理和进程退出的总时间预算。
> - 应在进程组创建前设置，建议在各Rank启动前统一配置。
> - 数值过小可能使转储尚未完成就进入后续错误处理；数值较大也不能保证底层接口或I/O卡住时最终能完成转储。
> - 关闭常规自动转储应使用[TORCH_HCCL_DUMP_ON_TIMEOUT](TORCH_HCCL_DUMP_ON_TIMEOUT.md)=0，不应把该变量设为0来代替开关。
> - 如诊断数据较多，可结合实际采集耗时调整，例如设为60000时，监控线程每轮最多等待60秒，WatchDog预留240秒。增大该值也可能延后后续故障处理。

## 配置示例

将常规自动故障转储的等待时间设为默认的15000毫秒：

```bash
export TORCH_HCCL_BLOCKING_WAIT=0
export TORCH_HCCL_ENABLE_MONITORING=1
export TORCH_HCCL_DUMP_ON_TIMEOUT=1
export TORCH_FR_BUFFER_SIZE=2000
export TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC=15000
```

## 使用约束

该环境变量在TorchNPU 2.15.0之前版本默认值为60000，即60秒。

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
