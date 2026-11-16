# TORCH\_HCCL\_EXTRA\_DUMP\_ON\_EXEC

## 功能描述

当使用HCCL作为通信后端时，通过此环境变量可控制WatchDog在标准异常出口额外尝试采集一次诊断快照。该路径可覆盖communicator预检查发现异步错误、异步错误查询接口本身失败等可能未经过常规转储请求的情况。

- 0：不增加异常出口额外采集。
- 1：在WatchDog捕获标准异常后，尝试广播诊断请求并在当前rank采集快照。

默认值：0。

> [!NOTE]
>
> - 本地额外采集不依赖`TORCH_HCCL_ENABLE_MONITORING`或`TORCH_HCCL_DUMP_ON_TIMEOUT`开启。
> - 即使本地FR缓冲区有效值为0，仍尝试广播请求；本地不生成FR数据文件，但可能采集Python线程调用栈。
> - 其他rank能否响应请求取决于各自的监控、常规转储配置以及执行时序。

## 配置示例

```bash
export TORCH_HCCL_EXTRA_DUMP_ON_EXEC=1
export TORCH_FR_BUFFER_SIZE=2000
```

如需其他rank响应广播请求，还需在这些rank上配置：

```bash
export TORCH_HCCL_ENABLE_MONITORING=1
export TORCH_HCCL_DUMP_ON_TIMEOUT=1
```

## 使用约束

- 该变量在WatchDog异常出口读取，建议在启动训练进程前统一配置。未进入该异常出口的故障不能依赖该变量采集。
- 已有诊断请求可能导致额外采集被跳过，且其他采集任务可能并发执行，因此当前rank或全部rank无法确保均生成文件。
- 额外快照不包含FR条目的调用栈，但仍会尝试采集独立的Python线程调用栈文件。
- 额外采集为同步尽力执行，不受`TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC`的单次异步等待时间限制。采集或底层I/O阻塞可能延迟异常退出。
- 该路径保留原通信错误和已有状态保存流程；无法确保采集一定成功，且不会修复同步提交失败或`NoHandling`模式下的event生命周期限制，具体请参见[飞行记录器](../../../developer_notes/fault_diagnosis/flight_recorder.md)。

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
