# 主要进程与线程说明

本文档帮助用户在运行torch及torch_npu训练或推理任务时，快速识别和理解系统中出现的各类进程与线程。当用户需要排查性能瓶颈、诊断异常进程/线程、了解系统资源占用情况或评估程序行为是否符合预期时，可通过本文档查询各进程/线程的归属、触发条件与生命周期。

基于CANN构建AI应用过程中CANN自身启动的进程/线程请参考《[CANN 环境变量参考](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/envvar/envref_07_0001.html)》。

## 主要进程与线程

进程与线程分为两类：

- **必然启动的进程与线程**：只要运行torch及torch_npu程序即会出现，无需额外特性开关或调用特定API。
- **条件触发的进程与线程**：仅在调用特定API或启用特定功能时才会出现。

**表 1** 必然启动的进程/线程

| 进程/线程类型 | 进程/线程名称 | 归属 | 简介 |
|---|---|---|---|
| Python主进程 | python3 | 用户/torch | 用户Python脚本的主进程，承载所有PyTorch/torch_npu运行时。程序启动时创建。 |
| intra-op线程池 | python3（与主线程同名） | torch | 算子内部并行计算线程池（矩阵乘法、卷积等CPU后端操作），可通过 `torch.set_num_threads(n)` 控制。import torch时创建，默认64线程。torchrun场景下自动设置 `OMP_NUM_THREADS=1`，线程池缩减为1。 |
| CANN初始化线程 | AtraceMonitor | CANN | Ascend Trace监控线程，管理ATrace数据采集生命周期。因torch_npu通过 `TORCH_DEVICE_BACKEND_AUTOLOAD` 自动加载，import torch即触发CANN初始化。 |
| CANN初始化线程 | WatchdogMonitor | CANN | ACL Runtime看门狗线程，检测运行时异常（超时、hang）。import torch时创建。 |
| CANN初始化线程 | PlogFlush | CANN | Plog日志异步刷盘线程，import torch时创建。可通过 `ASCEND_GLOBAL_LOG_LEVEL` 控制日志级别。 |
| ACL Runtime线程 | adx_data_dump_t | CANN | 调试数据dump线程。set_device时创建，每个设备一个。 |
| ACL Runtime线程 | PlogReportRecv | CANN | Plog日志上报接收线程，收集并分发各组件日志。set_device时创建。 |
| ACL Runtime线程 | TraceClientRecv | CANN | Trace客户端接收线程，接收并处理设备侧trace数据。set_device时创建。 |
| ACL Runtime线程 | MONITOR_0 | CANN | 设备监控线程，监控设备健康状态、温度、功耗。线程名中数字对应设备编号，每个NPU一个。set_device时创建。 |
| ACL Runtime线程 | REPORT_RAS | CANN | RAS硬件可靠性事件上报线程。set_device时创建。 |
| ACL Runtime线程 | RT_RECYCLE | CANN | Runtime资源异步回收线程。set_device时创建。 |
| TaskQueue线程 | acl_thread | torch_npu | 二级流水TaskQueue的consumer线程，从host侧取task下发到NPU，每个设备一个。首次计算时创建。与环境变量 `TASK_QUEUE_ENABLE` 配合使用。 |
| TaskQueue线程 | release_thread | torch_npu | 与acl_thread配对，异步释放acl_thread中存放的event/tensor资源。首次计算时创建。 |
| 算子编译线程 | AOE_RTKB | CANN | AOE Runtime Knowledge Base线程池（8线程），算子tiling策略搜索与缓存。非标量算子（如fp16 addmm）经aclop路径首次编译时创建；简单标量算子（如标量加法）不触发。 |

**表 2** 条件触发的进程/线程

| 功能场景 | 进程/线程名称 | 归属 | 简介 |
|---|---|---|---|
| DataLoader | pt_data_worker（子进程） | torch | worker子进程，数据预加载与预处理。OS进程名 `pt_data_worker`，`DataLoader(num_workers>0)` 时创建。 |
| DataLoader | pt_data_pin（线程） | torch (torch_npu适配) | pin_memory线程，锁定数据页到物理内存，加速CPU→NPU传输。OS线程名 `pt_data_pin`，Python线程名 `_pin_memory_loop`。`DataLoader(pin_memory=True)` 时创建。 |
| 多卡训练 | torchrun子进程 | torch | 每个NPU一个独立训练子进程（OS名 `python3`）。torchrun自动设 `OMP_NUM_THREADS=1`，intra-op线程池缩减。 |
| 多卡训练 | hccl_watchdog_t | torch_npu | HCCL通信看门狗，检测通信超时或进程挂起。`init_process_group(backend='hccl')` 时创建，每rank一个。 |
| 多卡训练 | pt_tcpstore_uv | torch | TCPStore后台通信线程（libuv），TCP事件循环和进程间键值同步。分布式init时创建。 |
| 多卡训练 | Hccl_HeartBeat | torch_npu | HCCL心跳监控线程，独立于watchdog。首次集合通信时创建，每rank一个。 |
| 多卡训练 | HcclIntra_0 | CANN | HCCL节点内通信线程。首次集合通信时创建。 |
| 多卡训练 | hccp_epoll | CANN | HCCL通信epoll事件循环线程。首次集合通信时创建。 |
| 多卡训练 | hccp_connect | CANN | HCCL通信连接建立线程。首次集合通信时创建，连接建立后即刻销毁（短生命周期）。 |
| 多卡训练 | Hccl_TopoDetect | CANN | HCCL拓扑检测线程。HCCL通信超时期间创建，用于检测集群拓扑连接状态。短生命周期。 |
| 多卡训练 | CaffeTaskThread（×5） | torch_npu | NPUEventManager线程池（5线程），NPU Event异步销毁和管理。大量Event操作或首次集合通信时触发。 |
| 反向传播 | pt_autograd_0~7（×8） | torch | Autograd反向传播引擎线程池（8线程）。首次 `loss.backward()` 或 `torch.compile()` 时懒初始化，创建后永久驻留。 |
| torch.compile | ThreadPoolExecutor（×128） | torch | Inductor异步编译线程池（Python `ThreadPoolExecutor`，128线程），并行处理kernel编译。首次编译时创建，不回收。线程名 `ThreadPoolExecutor-0_*`。 |
| torch.compile | AsyncCompile子进程（×32） | torch | Inductor编译worker子进程池（32进程），OS进程名 `python3`，命令行 `torch._inductor.compile_worker`。fork自父进程，需 `ps aux` 查看命令行确认。 |
| torch.compile | TuningProcess子进程 | torch | Kernel参数自动调优子进程。max-autotune模式下触发，编译完成后不回收。 |
| 算子编译 | AOE_RTKB（×8） | CANN | aclop路径TBE算子编译复用此线程池，不启动独立TBE编译子进程。 |
| inter-op线程池 | python3（线程数增加） | torch | 算子间并行执行线程池（懒初始化）。ONNX导出、torch.compile等首次触发inter-op并行时创建，创建后不回收。默认256线程。 |
| Profiling | NPUProfiler | torch_npu | Profiler主控线程。profiling期间创建，结束后回收。 |
| Profiling | MSVP_ProfTask / MSVP_Dev_\* / MSVP_Upld_\* / MSVP_ChanPool_0~7 | CANN (msprof) | msprof数据采集线程组：设备数据采集、上传、通道管理。profiling期间创建，结束后全部回收。 |
| Profiling | Profiling解析子进程 | torch_npu | profiling结束后通过 `subprocess.run` 调用 `msprof --export` / `msprof --analyze` 解析数据。 |
| Profiling | DynamicProfilerMonitor子进程 | torch_npu | 动态Profiler后台监控daemon子进程。设置 `DynamicProfilerUtils.CFG_CONFIG_PATH` 后启动。 |
| 静默故障检测 | _async_detect / _tcp_comm_checksum_state | torch_npu | ASD检测线程（daemon）：`_async_detect`（梯度异常检测）、`_tcp_comm_checksum_state`（分布式checksum）。OS级显示为 `python3`，需Python `threading.enumerate()` 查看。 |
| 多进程 | mp.spawn子进程 | torch | PyTorch多进程，每子进程为独立Python进程，拥有完整torch/CANN线程结构（77线程：`python3×66` + ACL Runtime×9 + `acl_thread` + `release_thread`）。`mp.spawn()` 时创建。 |
| 图模式 | NPUGraph | torch_npu / CANN | 图模式capture/replay。GE不创建独立命名线程，复用ACL Runtime线程。`make_graphed_callables` 会触发 `CaffeTaskThread`（×5）和 `pt_autograd_0~7`（×8），但非GE专属。 |
| 多设备并行 | _worker（Python线程） | torch_npu | 多设备并行推理worker线程。Python `threading.enumerate()` 中名为 `_worker`，OS级为 `python3` 无区分。`_parallel_apply` 执行期间存在，执行完毕即join销毁。需 `threading.enumerate()` 高频采样捕获。 |

## 进程与线程查询方式汇总

### OS级线程查询

```bash
# 查看指定进程的线程总数
ls /proc/<PID>/task/ | wc -l

# 列出所有线程名及数量（推荐，可直接看到各线程归属）
for tid in /proc/<PID>/task/*; do cat $tid/comm 2>/dev/null; done | sort | uniq -c | sort -rn

# 查看指定线程的线程名
cat /proc/<PID>/task/<TID>/comm
```

### 子进程查询

```bash
# 查看进程树（推荐，可直观看到父子关系）
pstree -p <PID>

# 查看指定进程的直接子进程
ps -ef --ppid <PID>

# 查看 torchrun 启动的所有训练子进程
pstree -p $(pgrep torchrun)

# 查看Inductor编译worker子进程（OS名python3，需命令行区分）
ps aux | grep compile_worker
```

### Python级线程查询

```python
# 列出所有 Python 可见线程
import threading
for t in threading.enumerate():
    print(f"  {t.name} (daemon={t.daemon})")
```

> [!NOTE]
>
> 部分C++层创建的线程（如 acl_thread、AOE_RTKB、CaffeTaskThread 等）在Python `threading.enumerate()` 中不可见，需通过 `/proc/<PID>/task/<TID>/comm` 查看。

### top工具查看线程

```bash
# 按线程模式查看指定进程的所有线程（按H键切换线程视图）
top -H -p <PID>
```

### npu-smi查看NPU上的进程

```bash
# 查看所有NPU上运行的进程
npu-smi info

# 查看指定设备上的进程详情
npu-smi info -t process -i 0
```

### PyTorch内置查询

```python
import torch

# 查看 intra-op 线程池大小
torch.get_num_threads()

# 查看 inter-op 线程池大小
torch.get_num_interop_threads()
```

### 快速查询线程名称

**表 3** OS线程名速查表

| OS线程名 | 归属 | 生命周期 |
|---|---|---|
| python3 | torch | intra-op/inter-op并行触发，创建后驻留 |
| AtraceMonitor | CANN | import torch触发，永久驻留 |
| WatchdogMonitor | CANN | import torch触发，永久驻留 |
| PlogFlush | CANN | import torch触发，永久驻留 |
| adx_data_dump_t | CANN | set_device触发，永久驻留 |
| PlogReportRecv | CANN | set_device触发，永久驻留 |
| TraceClientRecv | CANN | set_device触发，永久驻留 |
| MONITOR_0 | CANN | set_device触发，永久驻留 |
| REPORT_RAS | CANN | set_device触发，永久驻留 |
| RT_RECYCLE | CANN | set_device触发，永久驻留 |
| acl_thread | torch_npu | 首次NPU计算触发，永久驻留 |
| release_thread | torch_npu | 首次NPU计算触发，永久驻留 |
| AOE_RTKB | CANN | 首次算子编译触发，永久驻留 |
| pt_data_worker | torch | DataLoader(num_workers>0)触发，随DataLoader生命周期 |
| pt_data_pin | torch | DataLoader(pin_memory=True)触发，随DataLoader生命周期 |
| hccl_watchdog_t | torch_npu | HCCL init触发，永久驻留 |
| Hccl_HeartBeat | torch_npu | 首次集合通信触发，永久驻留 |
| pt_tcpstore_uv | torch | 分布式init触发，永久驻留 |
| hccp_connect | CANN | HCCL通信建立时触发，连接建立后即刻销毁 |
| HcclIntra_0 | CANN | HCCL节点内通信触发，永久驻留 |
| hccp_epoll | CANN | HCCL通信epoll触发，永久驻留 |
| Hccl_TopoDetect | CANN | HCCL超时期间触发，检测结束后销毁 |
| CaffeTaskThread | torch_npu | Event大量操作或集合通信触发，永久驻留 |
| pt_autograd_0~7 | torch | 反向传播或torch.compile触发，永久驻留 |
| NPUProfiler | torch_npu | Profiling期间存在，结束后回收 |
| MSVP_\* | CANN (msprof) | Profiling期间存在，结束后回收 |
