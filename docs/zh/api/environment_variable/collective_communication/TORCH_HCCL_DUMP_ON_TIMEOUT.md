# TORCH\_HCCL\_DUMP\_ON\_TIMEOUT

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量控制是否在Work异常、通信超时或WatchDog心跳停滞的诊断流程中，尝试进行常规自动FR转储。默认值为0。

- 配置为“0”：不启用常规自动FR转储。

- 配置为“1”：启用常规自动FR转储，需要同时配置心跳监控和有效的FR缓冲区。

同时开启该变量和心跳监控后，Work异常或超时会触发故障转储请求。全局进程组的心跳监控线程处理本地或远端故障请求后，若未收到停止请求，会终止当前进程。异步错误处理模式为0或2时，这条路径也会生效。WatchDog心跳停滞同样会进入故障诊断。

Work长时间未完成时也可能触发提前采集。监控线程异步处理提前采集请求，随后继续监控。

`TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC`默认15000毫秒，即15秒，监控线程每轮按该值等待异步转储结果。WatchDog广播故障转储信号后，按该值的4倍等待，默认等待60秒。

首轮异步转储未成功完成时，监控线程会再尝试一轮不包含FR条目调用栈的转储；前一轮任务不一定已经取消。

> [!NOTE]
>
> - 设置为1不会开启FR记录。常规自动转储需配合[TORCH_HCCL_ENABLE_MONITORING](TORCH_HCCL_ENABLE_MONITORING.md)=1和[TORCH_FR_BUFFER_SIZE](TORCH_FR_BUFFER_SIZE.md)的配置值为正整数。
> - 管道手动转储不依赖此变量开启，但仍需有效的监控、FR缓冲区和管道配置。[TORCH_HCCL_EXTRA_DUMP_ON_EXEC](TORCH_HCCL_EXTRA_DUMP_ON_EXEC.md)控制的本地异常出口额外采集也与此变量独立。
> - 应在进程组创建前设置，建议在各Rank启动前统一配置；远端能否响应请求还取决于对应rank的配置和执行状态。
> - 异步错误查询接口自身失败、通信器预检查异常等路径不一定经过常规自动转储请求；如需在WatchDog标准异常出口额外尝试采集，可单独配置`TORCH_HCCL_EXTRA_DUMP_ON_EXEC=1`。
> - Store不可用、设备接口或I/O阻塞、进程提前退出等情况可能导致文件缺失，不能保证所有rank均生成转储文件。

## 配置示例

在各rank启动前设置以下变量，按原有方式启动HCCL训练：

```bash
umask 077
mkdir -p ./trace_dir
export TORCH_HCCL_BLOCKING_WAIT=0
export TORCH_HCCL_ENABLE_MONITORING=1
export TORCH_HCCL_DUMP_ON_TIMEOUT=1
export TORCH_FR_BUFFER_SIZE=2000
export TORCH_FR_DUMP_TEMP_FILE="$PWD/trace_dir/comm_lib_trace_rank_"
```

默认静态文件名模式下，FR主文件以指定前缀追加rank，例如`comm_lib_trace_rank_0`。输出路径和分析方法请参见[飞行记录器](../../../developer_notes/fault_diagnosis/flight_recorder.md)。

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
