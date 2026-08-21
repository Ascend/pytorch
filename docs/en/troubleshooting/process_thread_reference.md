# Main Processes and Threads

This document helps users quickly identify and understand the various processes and threads that appear in the system when running torch and TorchNPU training or inference tasks. When users need to troubleshoot performance bottlenecks, diagnose abnormal processes/threads, understand system resource usage, or evaluate whether program behavior meets expectations, they can refer to this document to look up the affiliation, trigger conditions, and lifecycle of each process/thread.

For processes and threads started by CANN itself during the process of building AI apps based on CANN, see the [CANN Environment Variable Reference](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/maintenref/envvar/envref_07_0001.html).

## Main Process and Thread Description

Processes and threads fall into two categories:

- **Always-started processes and threads**: These appear whenever a torch or TorchNPU program runs, without requiring any additional feature switches or specific API calls.
- **Conditionally triggered processes and threads**: These appear only when specific APIs are called or specific features are enabled.

**Table 1** Always-started processes/threads

| Process/Thread Type | Process/Thread Name | Affiliation | Description |
| --- | --- | --- | --- |
| Python main process | python3 | User/torch | The main process of the user Python script, hosting all PyTorch/TorchNPU runtime. Created at program startup. |
| intra-op thread pool | python3 (same name as the main thread) | torch | A thread pool for intra-operator parallel computation (CPU backend operations such as matrix multiplication and convolution). Controllable via `torch.set_num_threads(n)`. Created upon `import torch`, with 64 threads by default. In torchrun scenarios, `OMP_NUM_THREADS=1` is automatically set, reducing the thread pool to 1. |
| CANN initialization thread | AtraceMonitor | CANN | Ascend Trace monitoring thread, managing the ATrace data collection lifecycle. Since TorchNPU is auto-loaded via `TORCH_DEVICE_BACKEND_AUTOLOAD`, `import torch` triggers CANN initialization. |
| CANN initialization thread | WatchdogMonitor | CANN | ACL Runtime watchdog thread, detecting runtime anomalies (timeouts, hangs). Created upon `import torch`. |
| CANN initialization thread | PlogFlush | CANN | Plog log asynchronous flush thread. Created upon `import torch`. The log level is controllable via `ASCEND_GLOBAL_LOG_LEVEL`. |
| ACL Runtime thread | adx_data_dump_t | CANN | Debug data dump thread. Created upon `set_device`, one per device. |
| ACL Runtime thread | PlogReportRecv | CANN | Plog log report receiving thread, collecting and distributing logs from various components. Created upon `set_device`. |
| ACL Runtime thread | TraceClientRecv | CANN | Trace client receiving thread, receiving and processing device-side trace data. Created upon `set_device`. |
| ACL Runtime thread | MONITOR_0 | CANN | Device monitoring thread, monitoring device health status, temperature, and power consumption. The number in the thread name corresponds to the device index, one per NPU. Created upon `set_device`. |
| ACL Runtime thread | REPORT_RAS | CANN | RAS hardware reliability event reporting thread. Created upon `set_device`. |
| ACL Runtime thread | RT_RECYCLE | CANN | Runtime resource asynchronous recycling thread. Created upon `set_device`. |
| TaskQueue thread | acl_thread | TorchNPU | The consumer thread of the two-level pipeline TaskQueue, fetching tasks from the host side and dispatching them to the NPU, one per device. Created upon the first computation. Used in conjunction with the environment variable `TASK_QUEUE_ENABLE`. |
| TaskQueue thread | release_thread | TorchNPU | Paired with acl_thread, asynchronously releasing event/tensor resources stored in acl_thread. Created upon the first computation. |
| Operator compilation thread | AOE_RTKB | CANN | AOE Runtime Knowledge Base thread pool (8 threads), for operator tiling strategy search and caching. Created upon the first compilation of non-scalar operators (for example, fp16 addmm) via the aclop path. Simple scalar operators (for example, scalar addition) do not trigger creation. |

**Table 2** Conditionally triggered processes/threads

| Feature Scenario | Process/Thread Name | Affiliation | Description |
| --- | --- | --- | --- |
| DataLoader | pt_data_worker (subprocess) | torch | Worker subprocess for data preloading and preprocessing. OS process name `pt_data_worker`. Created when `DataLoader(num_workers>0)`. |
| DataLoader | pt_data_pin (thread) | torch (TorchNPU adapted) | pin_memory thread, locking data pages to physical memory to accelerate CPU→NPU transfers. OS thread name `pt_data_pin`, Python thread name `_pin_memory_loop`. Created when `DataLoader(pin_memory=True)`. |
| Multi-device training | torchrun subprocess | torch | One independent training subprocess per NPU (OS name `python3`). torchrun automatically sets `OMP_NUM_THREADS=1`, reducing the intra-op thread pool. |
| Multi-device training | hccl_watchdog_t | TorchNPU | HCCL communication watchdog, detecting communication timeouts or process hangs. Created upon `init_process_group(backend='hccl')`, one per rank. |
| Multi-device training | pt_tcpstore_uv | torch | TCPStore backend communication thread (libuv), handling TCP event loop and inter-process key-value synchronization. Created upon distributed initialization. |
| Multi-device training | Hccl_HeartBeat | TorchNPU | HCCL heartbeat monitoring thread, independent of the watchdog. Created upon the initial collective communication, one per rank. |
| Multi-device training | HcclIntra_0 | CANN | HCCL intra-node communication thread. Created upon the initial collective communication. |
| Multi-device training | hccp_epoll | CANN | HCCL communication epoll event loop thread. Created upon the initial collective communication. |
| Multi-device training | hccp_connect | CANN | HCCL communication connection establishment thread. Created upon the initial collective communication and destroyed immediately after the connection is established (short lifecycle). |
| Multi-device training | Hccl_TopoDetect | CANN | HCCL topology detection thread. Created during HCCL communication timeout periods to detect cluster topology connection status. Short lifecycle. |
| Multi-device training | CaffeTaskThread (×5) | TorchNPU | NPUEventManager thread pool (5 threads), for asynchronous NPU Event destruction and management. Triggered by a large number of Event operations or the initial collective communication. |
| Backpropagation | pt_autograd_0~7 (×8) | torch | Autograd backpropagation engine thread pool (8 threads). Lazily initialized upon the first `loss.backward()` or `torch.compile()`, with permanent residency after creation. |
| torch.compile | ThreadPoolExecutor (×128) | torch | Inductor asynchronous compilation thread pool (Python `ThreadPoolExecutor`, 128 threads), processing kernel compilation in parallel. Created upon the first compilation and not recycled. Thread name `ThreadPoolExecutor-0_*`. |
| torch.compile | AsyncCompile subprocess (×32) | torch | Inductor compilation worker subprocess pool (32 processes). OS process name `python3`, command line `torch._inductor.compile_worker`. Forked from the parent process. Use `ps aux` to check the command line for confirmation. |
| torch.compile | TuningProcess subprocess | torch | Kernel parameter auto-tuning subprocess. Triggered in max-autotune mode and not recycled after compilation completes. |
| Operator compilation | AOE_RTKB (×8) | CANN | TBE operator compilation via the aclop path reuses this thread pool, without launching a separate TBE compilation subprocess. |
| inter-op thread pool | python3 (increased thread count) | torch | Inter-operator parallel execution thread pool (lazily initialized). Created when inter-op parallelism is first triggered, for example, by ONNX export or torch.compile, and not recycled after creation. Default 256 threads. |
| Profiling | NPUProfiler | TorchNPU | Profiler main control thread. Created during profiling and recycled after completion. |
| Profiling | MSVP_ProfTask / MSVP_Dev_\* / MSVP_Upld_\* / MSVP_ChanPool_0~7 | CANN (msprof) | msprof data collection thread group: device data collection, upload, and channel management. Created during profiling and fully recycled after completion. |
| Profiling | Profiling parsing subprocess | TorchNPU | Invoked via `subprocess.run` after profiling completes, calling `msprof --export` / `msprof --analyze` to parse data. |
| Profiling | DynamicProfilerMonitor subprocess | TorchNPU | Dynamic Profiler background monitoring daemon subprocess. Started after setting `DynamicProfilerUtils.CFG_CONFIG_PATH`. |
| Silent fault detection | _async_detect / _tcp_comm_checksum_state | TorchNPU | ASD detection threads (daemon): `_async_detect` (gradient anomaly detection), `_tcp_comm_checksum_state` (distributed checksum). Displayed as `python3` at the OS level. Use Python `threading.enumerate()` to inspect. |
| Multi-processing | mp.spawn subprocess | torch | PyTorch multi-processing, where each subprocess is an independent Python process with a complete torch/CANN thread structure (77 threads: `python3×66` + ACL Runtime×9 + `acl_thread` + `release_thread`). Created upon `mp.spawn()`. |
| Graph mode | NPUGraph | TorchNPU / CANN | Graph mode capture/replay. GE does not create independently named threads and reuses ACL Runtime threads. `make_graphed_callables` triggers `CaffeTaskThread` (×5) and `pt_autograd_0~7` (×8), but these are not GE-specific. |
| Multi-device parallelism | _worker (Python thread) | TorchNPU | Multi-device parallel inference worker thread. Named `_worker` in Python `threading.enumerate()`, indistinguishable as `python3` at the OS level. Exists during `_parallel_apply` execution and is joined/destroyed upon completion. Requires high-frequency sampling via `threading.enumerate()` to capture. |

## Process and Thread Query Methods Summary

### OS-Level Thread Query

```bash
# View the total number of threads for a specified process
ls /proc/<PID>/task/ | wc -l

# List all thread names and their counts (recommended, allows direct viewing of thread affiliations)
for tid in /proc/<PID>/task/*; do cat $tid/comm 2>/dev/null; done | sort | uniq -c | sort -rn

# View the thread name of a specified thread
cat /proc/<PID>/task/<TID>/comm
```

### Subprocess Query

```bash
# View the process tree (recommended, visually shows parent-child relationships)
pstree -p <PID>

# View direct child processes of a specified process
ps -ef --ppid <PID>

# View all training subprocesses started by torchrun
pstree -p $(pgrep torchrun)

# View Inductor compile worker subprocesses (OS name python3, requires command-line differentiation)
ps aux | grep compile_worker
```

### Python-Level Thread Query

```python
# List all Python-visible threads
import threading
for t in threading.enumerate():
    print(f"  {t.name} (daemon={t.daemon})")
```

> [!NOTE]
>
> Some threads created at the C++ layer (such as acl_thread, AOE_RTKB, CaffeTaskThread, and so on) are not visible in Python `threading.enumerate()` and must be viewed via `/proc/<PID>/task/<TID>/comm`.

### Viewing Threads with the top Tool

```bash
# View all threads of a specified process in thread mode (press H to switch to thread view)
top -H -p <PID>
```

### View Processes on NPUs Using npu-smi

```bash
# View processes running on all NPUs
npu-smi info

# View process details on a specified device
npu-smi info -t process -i 0
```

### PyTorch Built-in Query

```python
import torch

# View the intra-op thread pool size
torch.get_num_threads()

# View the inter-op thread pool size
torch.get_num_interop_threads()
```

### Quick Thread Name Lookup

**Table 3** OS thread name quick reference

| OS Thread Name | Affiliation | Lifecycle |
| --- | --- | --- |
| python3 | torch | Triggered by intra-op/inter-op parallelism, permanent residency after creation |
| AtraceMonitor | CANN | Triggered by import torch, permanent residency |
| WatchdogMonitor | CANN | Triggered by import torch, permanent residency |
| PlogFlush | CANN | Triggered by import torch, permanent residency |
| adx_data_dump_t | CANN | Triggered by set_device, permanent residency |
| PlogReportRecv | CANN | Triggered by set_device, permanent residency |
| TraceClientRecv | CANN | Triggered by set_device, permanent residency |
| MONITOR_0 | CANN | Triggered by set_device, permanent residency |
| REPORT_RAS | CANN | Triggered by set_device, permanent residency |
| RT_RECYCLE | CANN | Triggered by set_device, permanent residency |
| acl_thread | TorchNPU | Triggered by initial NPU computation, permanent residency |
| release_thread | TorchNPU | Triggered by initial NPU computation, permanent residency |
| AOE_RTKB | CANN | Triggered by initial operator compilation, permanent residency |
| pt_data_worker | torch | Triggered by DataLoader(num_workers>0), follows DataLoader lifecycle |
| pt_data_pin | torch | Triggered by DataLoader(pin_memory=True), follows DataLoader lifecycle |
| hccl_watchdog_t | TorchNPU | Triggered by HCCL init, permanent residency |
| Hccl_HeartBeat | TorchNPU | Triggered by initial collective communication, permanent residency |
| pt_tcpstore_uv | torch | Triggered by distributed init, permanent residency |
| hccp_connect | CANN | Triggered during HCCL communication establishment, destroyed immediately after connection establishment |
| HcclIntra_0 | CANN | Triggered by HCCL intra-node communication, permanent residency |
| hccp_epoll | CANN | Triggered by HCCL communication epoll, permanent residency |
| Hccl_TopoDetect | CANN | Triggered during HCCL timeout, destroyed after detection completes |
| CaffeTaskThread | TorchNPU | Triggered by extensive Event operations or collective communication, permanent residency |
| pt_autograd_0~7 | torch | Triggered by backpropagation or torch.compile, permanent residency |
| NPUProfiler | TorchNPU | Exists during profiling, reclaimed after completion |
| MSVP_\* | CANN (msprof) | Exists during profiling, reclaimed after completion |
