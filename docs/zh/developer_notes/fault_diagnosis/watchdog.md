# WatchDog

## 简介

WatchDog是用于异步检查通信进展的后台线程。它跟踪已提交的通信任务（Work），检查HCCL异步错误、通信超时和执行状态，并按配置记录诊断信息、清理通信器或终止当前训练进程。

WatchDog与训练代码运行在同一个进程中，不是独立的监控进程。每个满足启动条件的`ProcessGroupHCCL`有自己的WatchDog线程；同一训练进程创建多个HCCL进程组时，可以存在多个WatchDog线程。

心跳监控线程（HeartbeatMonitor）负责检查WatchDog本身是否仍有进展，并处理诊断转储请求。飞行记录器（Flight Recorder，以下简称FR）负责保存通信记录。三者用途不同：

| 组件 | 主要职责 | 当前默认行为 |
| --- | --- | --- |
| WatchDog | 轮询Work队列，检查通信错误和超时，通知通信结果。 | 在构建启用HCCL错误检查且未开启阻塞等待时启动。 |
| 心跳监控线程 | 检查WatchDog心跳，响应自动诊断和手动管道请求。 | 不启动，需设置`TORCH_HCCL_ENABLE_MONITORING=1`。 |
| FR | 在内存环形缓冲区中保存通信记录，按请求转储。 | 记录默认关闭，需将`TORCH_FR_BUFFER_SIZE`设为正整数。 |

本文描述WatchDog以及HeartbeatMonitor的实现。FR字段、输出文件和离线分析方法请参见[飞行记录器](flight_recorder.md)。

## 使用场景

- **正常训练中的通信监测**：WatchDog持续检查已入队通信任务的执行状态、异步错误和超时，任务正常完成后更新进度并进行相关收尾。
- **WatchDog运行状态监测**：开启心跳监控后，监控线程持续检查WatchDog是否仍有进展；即使暂时没有通信任务，正常运行的WatchDog也会更新心跳。
- **通信异常与超时处理**：某个rank未参加通信，或HCCL、设备、链路出现异常时，记录错误并按配置执行诊断、清理或退出。
- **多rank通信一致性排查**：通信调用顺序等出现不一致时，通过Desync诊断和FR记录比较各rank进度，辅助定位原因。
- **通信记录导出**：通过FR保存最近的通信操作、执行状态和调用栈等记录，用于分析各rank的通信进度及异常位置。开启相应配置后，可在通信异常、超时或WatchDog心跳停滞时自动尝试导出，也可在训练运行期间通过命名管道手动触发；WatchDog异常出口还支持独立配置额外导出。
- **通信状态保存**：开启`TORCH_HCCL_STATUS_SAVE_ENABLE`后，在正常训练过程中按配置的间隔保存HCCL进程组状态，并在部分异常或退出路径尝试保存现场信息，辅助分析通信进度和故障。保存目录由`TORCH_HCCL_STATUS_SAVE_PATH`指定，周期保存间隔由`TORCH_HCCL_STATUS_SAVE_INTERVAL`控制。

## 使用指导

### 基本配置与默认值

请在启动训练进程、初始化HCCL进程组前配置环境变量。

**表 1**  WatchDog与心跳监控基本配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| [TORCH_HCCL_ASYNC_ERROR_HANDLING](../../api/environment_variable/collective_communication/TORCH_HCCL_ASYNC_ERROR_HANDLING.md) | 3 | Work异常处理策略，支持0、1、2、3，详见表2。 |
| [TORCH_HCCL_BLOCKING_WAIT](../../api/environment_variable/collective_communication/TORCH_HCCL_BLOCKING_WAIT.md) | 0 | 设为1时在用户调用Work等待接口时进行主机侧阻塞等待，不创建WatchDog线程，也不启动心跳监控。 |
| [TORCH_HCCL_ENABLE_MONITORING](../../api/environment_variable/collective_communication/TORCH_HCCL_ENABLE_MONITORING.md) | 0 | 是否启动心跳监控线程。设为0时不启动线程。 |
| [TORCH_HCCL_DUMP_ON_TIMEOUT](../../api/environment_variable/collective_communication/TORCH_HCCL_DUMP_ON_TIMEOUT.md) | 0 | 是否启用Work异常、通信超时及心跳停滞诊断中的常规自动FR转储，默认不开启。 |
| [TORCH_HCCL_DESYNC_DEBUG](../../api/environment_variable/collective_communication/TORCH_HCCL_DESYNC_DEBUG.md) | 0 | 记录各rank通信进度，并在Work超时时尝试生成Desync报告。`TORCH_DISTRIBUTED_DEBUG=DETAIL`也会启用相关诊断。 默认不开启。|
| [TORCH_HCCL_EXTRA_DUMP_ON_EXEC](../../api/environment_variable/collective_communication/TORCH_HCCL_EXTRA_DUMP_ON_EXEC.md) | 0 | 在WatchDog捕获标准异常后，额外尝试广播诊断请求并在本地采集快照。默认不开启。|
| [TORCH_HCCL_STATUS_SAVE_ENABLE](../../api/environment_variable/collective_communication/TORCH_HCCL_STATUS_SAVE_ENABLE.md) | 0 | 是否保存HCCL进程组状态。WatchDog周期检查和故障处理中会使用，输出不是FR文件。 |
| [TORCH_HCCL_STATUS_SAVE_PATH](../../api/environment_variable/collective_communication/TORCH_HCCL_STATUS_SAVE_PATH.md) | `/tmp` | 状态文件保存目录，仅在开启状态保存时生效。 |
| [TORCH_HCCL_STATUS_SAVE_INTERVAL](../../api/environment_variable/collective_communication/TORCH_HCCL_STATUS_SAVE_INTERVAL.md) | 2秒 | 周期保存的时间阈值，实际写盘时机还受WatchDog轮询影响；配置小于等于0时回退为2秒，仅在开启状态保存时生效。 |

> [!NOTE]
> 当前仍兼容旧名称`HCCL_ASYNC_ERROR_HANDLING`、`HCCL_BLOCKING_WAIT`和`HCCL_DESYNC_DEBUG`。新旧名称同时设置时，优先使用对应的`TORCH_HCCL_*`名称。建议新脚本统一使用新名称。

### 选择异步错误处理策略

**表 2**  Work异常分支的错误处理模式

| 取值 | 模式 | 清理通信器 | 在该分支抛出异常 |
| --- | --- | --- | --- |
| 0 | `NoHandling` | 否 | 否，但仍跟踪符合条件的Work，记录错误和结果 |
| 1 | `TearDown` | 是，尝试清理出错Work及该进程组相关通信器。 | 是 |
| 2 | `CleanUpOnly` | 是，尝试清理相关通信器。 | 否 |
| 3（默认） | `SkipCleanUp` | 跳过该分支的通信器清理 | 是 |

表中的抛出行为还要经过WatchDog外层重抛策略处理。模式1和3在默认外层配置下会终止进程；模式0和2不代表任何故障下都不会退出，心跳监控和其他异常路径仍可能终止进程。

在非阻塞等待模式下，如果Desync诊断已启用，而异步错误处理模式为0，进程组在初始化时会将自身的异步错误处理策略改为模式3（`SkipCleanUp`），并打印`Enabling TORCH_HCCL_ASYNC_ERROR_HANDLING.`。此时WatchDog处理Work异常时会跳过该分支的通信器清理并抛出异常。

### 区分各类超时

**表 3**  超时与轮询配置

| 配置 | 单位与默认行为 | 控制范围 |
| --- | --- | --- |
| `init_process_group(timeout=...)`、`new_group(timeout=...)` | 使用`datetime.timedelta`设置。使用HCCL后端且未指定`timeout`时，进程组默认超时时间为30分钟（1800秒）。WatchDog超时时间的调整条件见下文。 | 设置通信任务允许等待的时间。任务从创建开始计时，超过有效超时时间仍未完成时，WatchDog报告超时。 |
| [HCCL_CONNECT_TIMEOUT](https://gitcode.com/cann/hccl/blob/9.2.0/docs/zh/user_guide/hccl_env/HCCL_CONNECT_TIMEOUT.md) | 秒，默认120。 | HCCL建链等待时间。 |
| [HCCL_EXEC_TIMEOUT](https://gitcode.com/cann/hccl/blob/9.2.0/docs/zh/user_guide/hccl_env/HCCL_EXEC_TIMEOUT.md) | 秒，默认1836。 | 底层HCCL执行超时。 |
| [TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC](../../api/environment_variable/collective_communication/TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC.md) | 秒，默认600。 | 心跳监控检查WatchDog进展的时间间隔，以及诊断后的额外等待时间。|
| [TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC](../../api/environment_variable/collective_communication/TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC.md) | 毫秒，默认 15000（15 秒）。 | 监控线程每次等待异步转储结果的超时时间。WatchDog广播故障转储信号后，按该值的 4 倍等待，默认为 60 秒。 |
| [TORCH_HCCL_COORD_CHECK_MILSEC](../../api/environment_variable/collective_communication/TORCH_HCCL_COORD_CHECK_MILSEC.md) | 毫秒，默认1000。 | 全局进程组监控线程轮询共享诊断请求和有效管道的间隔。 |

异步错误处理模式为1或3，且进程组的timeout恰好是 1800秒时，后端会调整watchdog超时时间。

- 如果HCCL_EXEC_TIMEOUT大于0，就将WatchDog超时设为：HCCL执行超时＋1800秒。
- 如果HCCL_EXEC_TIMEOUT=0，表示HCCL执行不设超时，WatchDog超时也会被设为一个很大的值。
- 如果进程组的timeout不是1800秒，例如300秒，就保留用户设置，不做上述调整。

### 高级诊断与错误传播

**表 4**  高级配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| [TORCH_HCCL_RETHROW_HCCL_ERRORS](../../api/environment_variable/collective_communication/TORCH_HCCL_RETHROW_HCCL_ERRORS.md) | 1 | WatchDog捕获标准异常后，是否继续重抛。设为0可抑制对应标准异常的重抛并结束WatchDog线程，但其他路径仍可能终止进程。|
| [TORCH_HCCL_TEARDOWN_ON_TIMEOUT](../../api/environment_variable/collective_communication/TORCH_HCCL_TEARDOWN_ON_TIMEOUT.md) | 0 | 对异步错误处理模式1或3中经过Work超时处理的异常，允许在关闭一般重抛后仍重抛。 |
| [TORCH_HCCL_PROPAGATE_ERROR](../../api/environment_variable/collective_communication/TORCH_HCCL_PROPAGATE_ERROR.md) | 0 | 通过进程组Store传播错误来源，并在远端设置`REMOTE_ERROR`状态。 |
| [TORCH_HCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN](../../api/environment_variable/collective_communication/TORCH_HCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN.md) | 1 | monitor进入故障诊断时，在C++栈采集器已注册的条件下尝试输出当前C++线程栈。|

通常保持默认重抛行为更便于错误及时暴露。关闭重抛用于有明确错误接管机制的排障或恢复流程，不应作为让故障训练继续运行的通用方法。

### 查看日志与定位异常

排查时先保留所有rank的完整日志，找到最早的设备、HCCL或用户线程错误，再按进程组和通信序列对齐。其他rank后续报超时，可能是首个故障导致的结果。

**表 5**  常见日志及含义

| 日志片段 | 含义与排查方向 |
| --- | --- |
| `Process group watchdog thread started!` | 对应进程组的WatchDog已进入运行入口。 |
| `Heartbeat monitor thread started!` | 对应心跳监控线程已运行。结合配置日志核对开关。 |
| `Watchdog caught collective operation timeout` | 某次通信超过设定时间仍未完成。检查各rank是否都调用了对应通信，操作顺序和张量大小是否一致，以及是否有rank提前报错或退出。 |
| `remote error detected from rank` | 通过Store获知远端错误，优先检查该来源rank的原始日志。 |
| `watchdog got stuck` | 心跳未观察到进展。检查WatchDog是否阻塞在NPU/HCCL API、锁或其他调用。 |
| `Process group work ... dispatch timeout` | HCCL下发进度检查的告警，不等同于Work通信超时；之后的`dispatch success`表示已观察到下发恢复。 |
| `Could not acquire GIL within 300 ms` | 诊断线程未在规定时间内完成GIL获取检查，是可能存在GIL相关阻塞的线索。 |
| `Flight Recorder trace successfully dumped.` | 转储函数已返回。使用默认写入器时，请结合`Finished writing Flight Recorder debug info to`日志和目标文件确认写盘结果。 |
| `Process group watchdog thread terminated with exception` | WatchDog外层捕获异常。|

启动时的`PGHCCL Watchdog environments`和`HeartbeatMonitor environments`日志可辅助核对配置是否符合预期。当前实现只由全局进程组打印这组配置。

FR文件的位置、内容和跨rank分析命令请参见[飞行记录器](flight_recorder.md)。没有文件时，应分别检查记录是否开启、监控是否启动、是否进入转储路径以及写盘是否成功。

## 使用样例

### 使用默认异步错误处理

在启动训练前配置：

```bash
export TORCH_HCCL_ASYNC_ERROR_HANDLING=3
export TORCH_HCCL_BLOCKING_WAIT=0
export TORCH_HCCL_ENABLE_MONITORING=0
export TORCH_HCCL_DUMP_ON_TIMEOUT=0
```

这组配置保留WatchDog通信检查，不启动心跳监控，也不增加常规自动FR转储。变量未设置时，当前版本采用相同默认值。

### 开启心跳保护和自动故障转储

```bash
umask 077
mkdir -p ./trace_dir
export TORCH_HCCL_BLOCKING_WAIT=0
export TORCH_HCCL_ASYNC_ERROR_HANDLING=1
export TORCH_HCCL_ENABLE_MONITORING=1
export TORCH_HCCL_DUMP_ON_TIMEOUT=1
export TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC=600
export TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC=15000
export TORCH_FR_BUFFER_SIZE=2000
export TORCH_FR_DUMP_TEMP_FILE="$PWD/trace_dir/comm_lib_trace_rank_"
```

此处显式选择模式1，表示在Work异常分支先尝试清理再抛出；若需要保留默认的跳过清理策略，可改为3。以上配置不会主动制造故障。多机训练需在各rank所在环境配置可写路径，并收集各机对应文件。

如需记录各rank通信进度，可增加：

```bash
export TORCH_HCCL_DESYNC_DEBUG=1
```

如需覆盖WatchDog标准异常出口的额外快照，可增加：

```bash
export TORCH_HCCL_EXTRA_DUMP_ON_EXEC=1
```

如果只需要WatchDog心跳保护，不需要自动FR转储，可保留`TORCH_HCCL_ENABLE_MONITORING=1`，并设置`TORCH_HCCL_DUMP_ON_TIMEOUT=0`。

### 在脚本中等待通信结果

以下示例启动两个训练进程，分别使用两张NPU执行一次all-reduce，并通过`get_future_result()`等待WatchDog通知通信结果。

先将下面代码保存为`watchdog_example.py`：

```python
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu

local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank)
dist.init_process_group("hccl", timeout=timedelta(seconds=300))

tensor = torch.ones(16, device=f"npu:{local_rank}")
work = dist.all_reduce(tensor, async_op=True)
result = work.get_future_result().wait()
if result != 0:
    raise RuntimeError(f"HCCL all-reduce failed with WorkResult={result}")
work.wait()
torch.npu.synchronize()
print(f"rank={dist.get_rank()}, result={result}, tensor={tensor.cpu()}", flush=True)

dist.destroy_process_group()
```

然后在有两张可用NPU的单机环境中执行：

```bash
TORCH_HCCL_BLOCKING_WAIT=0 torchrun --standalone --nproc_per_node=2 watchdog_example.py
```

正常运行时，两个rank的输出均为`result=0`，张量的16个元素均为`2`。每个rank最初提供全为1的张量，all-reduce默认执行求和，所以结果为`1 + 1 = 2`。两个进程独立运行，打印顺序不固定。

`get_future_result().wait()`返回的是通信状态对应的整数。`WorkResult`枚举约定了这些数值，HCCL将相应数值写入future，因此直接打印时显示数字，而不是`SUCCESS`等名称：

| 返回值 | 状态名称 | 含义 |
| --- | --- | --- |
| 0 | `WorkResult.SUCCESS` | 这次通信成功完成|
| 1 | `WorkResult.TIMEOUT` | 这次通信超时|
| 2 | `WorkResult.COMM_ERROR` | 这次通信发生错误|
| 100 | `WorkResult.UNKNOWN` | 枚举保留的未知结果状态，当前HCCL路径不保证会返回该值 |

## 约束说明

- 本文所述的WatchDog故障检测与诊断机制仅适用于TorchNPU 2.15及以上版本。
- 发生故障后，程序会尝试导出FR记录和调用栈。但如果设备接口、网络、Python线程或文件写入卡住，或者部分rank已经退出，可能无法保存完整信息，也可能延迟进程退出，因此不保证每个rank都有诊断文件。
- 当`TORCH_HCCL_BLOCKING_WAIT=0`时，`Work.wait()`主要让当前NPU流上的后续操作等待通信完成，Python代码可能在通信完成前继续执行。如果需要明确等待设备完成，可调用`torch.npu.synchronize()`。
- 请在启动训练前，为各rank配置好所需环境变量，不要依赖训练中途修改变量来改变已有进程组的行为。不同设备和CANN版本的支持情况，请查看对应版本文档。
- 本文所述的WatchDog故障检测与诊断机制仅适用于TorchNPU 2.15及以上版本。
