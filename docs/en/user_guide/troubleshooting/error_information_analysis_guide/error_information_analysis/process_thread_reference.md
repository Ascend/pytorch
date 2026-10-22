# Main Processes and Threads

This document helps users quickly identify and understand the various processes and threads in the system when running torch and TorchNPU training or inference tasks. When users need to troubleshoot performance bottlenecks, diagnose abnormal processes/threads, understand system resource usage, or evaluate whether program behavior meets expectations, they can query the ownership, trigger conditions, and lifecycles of each process/thread through this document.

For the processes/threads started by CANN itself during the construction of AI applications based on CANN, see *[CANN Environment Variable Reference](https://gitcode.com/cann/docs/blob/9.2.0/docs/en/env-vars/README.md)*.

## Main Processes and Threads

Processes and threads are classified into two categories:

- **Processes and threads started inevitably**: they appear as long as torch and TorchNPU programs are running, without the need for additional feature switches or specific API calls.
- **Conditionally triggered processes and threads**: they appear only when specific APIs are called or specific features are enabled.

**Table 1** Processes/threads started inevitably

| Process/Thread Type | Process/Thread Name | Ownership | Description |
|---|---|---|---|
| Python main process | python3 | User/torch | Main process of the user Python script, hosting all PyTorch/TorchNPU runtimes. Created when the program starts. |
| intra-op thread pool | python3 (same name as the main thread) | torch | Thread pool for intra-operator parallel computation (CPU backend operations such as matrix multiplication and convolution), controllable through `torch.set_num_threads(n)`. Created when torch is imported, with 64 threads by default. In the torchrun scenario, `OMP_NUM_THREADS=1` is automatically set, and the thread pool is reduced to 1. |
| CANN initialization thread | AtraceMonitor | CANN | Ascend Trace monitoring thread, managing the ATrace data collection lifecycle. Because TorchNPU is automatically loaded through `TORCH_DEVICE_BACKEND_AUTOLOAD`, importing torch triggers CANN initialization. |
| CANN initialization thread | WatchdogMonitor | CANN | ACL Runtime watchdog thread, detecting runtime exceptions (timeout, hang). Created when torch is imported. |
| CANN initialization thread | PlogFlush | CANN | Asynchronous thread for flushing Plog logs to disk, created when torch is imported. The log level can be controlled through `ASCEND_GLOBAL_LOG_LEVEL`. |
| ACL Runtime thread | adx_data_dump_t | CANN | Thread for dumping debugging data. Created during set_device, one per device. |
| ACL Runtime thread | PlogReportRecv | CANN | Thread for receiving Plog log reports, collecting and distributing logs of each component. Created during set_device. |
| ACL Runtime thread | TraceClientRecv | CANN | Trace client receiving thread, receiving and processing device-side trace data. Created during set_device. |
| ACL Runtime thread | MONITOR_0 | CANN | Device monitoring thread, monitoring device health status, temperature, and power consumption. The number in the thread name corresponds to the device number, one per NPU. Created during set_device. |
| ACL Runtime thread | REPORT_RAS | CANN | Thread for reporting RAS hardware reliability events. Created during set_device. |
| ACL Runtime thread | RT_RECYCLE | CANN | Thread for asynchronously reclaiming Runtime resources. Created during set_device. |
| TaskQueue thread | acl_thread | TorchNPU | Consumer thread of the two-level pipeline TaskQueue, fetching tasks from the host side and delivering them to the NPU, one per device. Created at the first computation. Used together with the environment variable `TASK_QUEUE_ENABLE`. |
| TaskQueue thread | release_thread | TorchNPU | Paired with acl_thread, asynchronously releasing the event/tensor resources stored in acl_thread. Created at the first computation. |
| Operator compilation thread | AOE_RTKB | CANN | AOE Runtime Knowledge Base thread pool (8 threads), for operator tiling strategy search and caching. Created when non-scalar operators (such as fp16 addmm) are compiled for the first time through the aclop path; simple scalar operators (such as scalar addition) do not trigger it. |

**Table 2** Conditionally triggered processes/threads

| Function Scenario | Process/Thread Name | Ownership | Description |
|---|---|---|---|
| DataLoader | pt_data_worker (child process) | torch | Worker child process for data preloading and preprocessing. The OS process name is `pt_data_worker`, created when `DataLoader(num_workers>0)`. |
| DataLoader | pt_data_pin (thread) | torch (TorchNPU adaptation) | pin_memory thread, locking data pages to physical memory to accelerate CPU-to-NPU transfer. The OS thread name is `pt_data_pin`, and the Python thread name is `_pin_memory_loop`. Created when `DataLoader(pin_memory=True)`. |
| Multi-card training | torchrun child process | torch | One independent training child process per NPU (OS name `python3`). torchrun automatically sets `OMP_NUM_THREADS=1`, and the intra-op thread pool is reduced. |
| Multi-card training | hccl_watchdog_t | TorchNPU | HCCL communication watchdog, which detects communication timeouts or process hangs. Created when `init_process_group(backend='hccl')` is called, one per rank. |
| Multi-card training | pt_tcpstore_uv | torch | TCPStore background communication thread (libuv), TCP event loop and inter-process key-value synchronization. Created during distributed init. |
| Multi-card training | Hccl_HeartBeat | TorchNPU | HCCL heartbeat monitoring thread, independent of the watchdog. Created at the first collective communication, one per rank. |
| Multi-card training | HcclIntra_0 | CANN | HCCL intra-node communication thread. Created at the first collective communication. |
| Multi-card training | hccp_epoll | CANN | HCCL communication epoll event loop thread. Created at the first collective communication. |
| Multi-card training | hccp_connect | CANN | HCCL communication connection establishment thread. Created at the first collective communication and destroyed immediately after the connection is established (short lifecycle). |
| Multi-card training | Hccl_TopoDetect | CANN | HCCL topology detection thread. Created during an HCCL communication timeout to detect the cluster topology connection status. Short lifecycle. |
| Multi-card training | CaffeTaskThread (×5) | TorchNPU | NPUEventManager thread pool (5 threads), for asynchronous destruction and management of NPU Events. Triggered by a large number of Event operations or at the first collective communication. |
| Backpropagation | pt_autograd_0~7 (×8) | torch | Thread pool of the Autograd backpropagation engine (8 threads). Lazily initialized at the first `loss.backward()` or `torch.compile()`, and permanently resident after creation. |
| torch.compile | ThreadPoolExecutor (×128) | torch | Inductor asynchronous compilation thread pool (Python `ThreadPoolExecutor`, 128 threads), processing kernel compilation in parallel. Created at the first compilation and never reclaimed. Thread name: `ThreadPoolExecutor-0_*`. |
| torch.compile | AsyncCompile child process (×32) | torch | Inductor compilation worker child process pool (32 processes), with OS process name `python3` and command line `torch._inductor.compile_worker`. Forked from the parent process; use `ps aux` to confirm the command line. |
| torch.compile | TuningProcess child process | torch | Child process for automatic kernel parameter tuning. Triggered in max-autotune mode and not reclaimed after compilation. |
| Operator compilation | AOE_RTKB (×8) | CANN | The TBE operator compilation on the aclop path reuses this thread pool, and no independent TBE compilation child process is started. |
| inter-op thread pool | python3 (increased thread count) | torch | Thread pool for parallel execution between operators (lazy initialization). Created when ONNX export, torch.compile, and so on first trigger inter-op parallelism, and never reclaimed after creation. 256 threads by default. |
| Profiling | NPUProfiler | TorchNPU | Profiler master control thread. Created during profiling and reclaimed after profiling ends. |
| Profiling | MSVP_ProfTask / MSVP_Dev_\\\* / MSVP_Upld_\\\* / MSVP_ChanPool_0~7 | CANN (msprof) | msprof data collection thread group: device data collection, upload, and channel management. Created during profiling and all reclaimed after profiling ends. |
| Profiling | Profiling parsing child process | TorchNPU | After profiling ends, `subprocess.run` is used to call `msprof --export` / `msprof --analyze` to parse data. |
| Profiling | DynamicProfilerMonitor child process | TorchNPU | Background monitoring daemon child process of the dynamic Profiler. Started after `DynamicProfilerUtils.CFG_CONFIG_PATH` is set. |
| Silent fault detection | _async_detect / _tcp_comm_checksum_state | TorchNPU | ASD detection thread (daemon): `_async_detect` (gradient exception detection), `_tcp_comm_checksum_state` (distributed checksum). Displayed as `python3` at the OS level; use Python `threading.enumerate()` to view. |
| Multiprocessing | mp.spawn child process | torch | PyTorch multiprocessing, where each child process is an independent Python process with the complete torch/CANN thread structure (77 threads: `python3×66` + ACL Runtime×9 + `acl_thread` + `release_thread`). Created when `mp.spawn()` is called. |
| Graph mode | NPUGraph | TorchNPU / CANN | Graph mode capture/replay. GE does not create independently named threads and reuses the ACL Runtime threads. `make_graphed_callables` triggers `CaffeTaskThread` (×5) and `pt_autograd_0~7` (×8), but they are not GE-specific. |
| Multi-device parallel | _worker (Python thread) | TorchNPU | Worker thread for multi-device parallel inference. In Python `threading.enumerate()`, it is named `_worker`, and it is displayed as `python3` at the OS level without distinction. It exists during `_parallel_apply` execution and is destroyed by join after execution. It requires high-frequency sampling with `threading.enumerate()` to capture. |

## Summary of Process and Thread Query Methods

### OS-Level Thread Query

```bash
# View the total number of threads of the specified process
ls /proc/<PID>/task/ | wc -l

# List all thread names and counts (recommended; you can directly see which process each thread belongs to)
for tid in /proc/<PID>/task/*; do cat $tid/comm 2>/dev/null; done | sort | uniq -c | sort -rn

# View the thread name of the specified thread
cat /proc/<PID>/task/<TID>/comm
```

### Child Process Query

```bash
# View the process tree (recommended; directly shows parent-child relationships)
pstree -p <PID>

# View the direct child processes of the specified process
ps -ef --ppid <PID>

# View all training child processes started by torchrun
pstree -p $(pgrep torchrun)

# View Inductor compilation worker child processes (OS name python3; distinguish them by the command line)
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
> Some threads created at the C++ layer (such as acl_thread, AOE_RTKB, and CaffeTaskThread) are not visible in Python `threading.enumerate()`, and need to be viewed through `/proc/<PID>/task/<TID>/comm`.

### Viewing Threads with the top Tool

```bash
# View all threads of the specified process in thread mode (press H to switch the thread view)
top -H -p <PID>
```

### Viewing Processes on NPUs with npu-smi

```bash
# View processes running on all NPUs
npu-smi info

# View process details on the specified device
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

### Quickly Querying Thread Names

**Table 3** OS thread name quick reference

| OS Thread Name | Ownership | Lifecycle |
|---|---|---|
| python3 | torch | Triggered by intra-op/inter-op parallelism; resident after creation |
| AtraceMonitor | CANN | Triggered by importing torch; permanently resident |
| WatchdogMonitor | CANN | Triggered by importing torch; permanently resident |
| PlogFlush | CANN | Triggered by importing torch; permanently resident |
| adx_data_dump_t | CANN | Triggered by set_device; permanently resident |
| PlogReportRecv | CANN | Triggered by set_device; permanently resident |
| TraceClientRecv | CANN | Triggered by set_device; permanently resident |
| MONITOR_0 | CANN | Triggered by set_device; permanently resident |
| REPORT_RAS | CANN | Triggered by set_device; permanently resident |
| RT_RECYCLE | CANN | Triggered by set_device; permanently resident |
| acl_thread | TorchNPU | Triggered by the first NPU computation; permanently resident |
| release_thread | TorchNPU | Triggered by the first NPU computation; permanently resident |
| AOE_RTKB | CANN | Triggered by the first operator compilation; permanently resident |
| pt_data_worker | torch | Triggered by DataLoader(num_workers>0); follows the DataLoader lifecycle |
| pt_data_pin | torch | Triggered by DataLoader(pin_memory=True); follows the DataLoader lifecycle |
| hccl_watchdog_t | TorchNPU | Triggered by HCCL init; permanently resident |
| Hccl_HeartBeat | TorchNPU | Triggered by the first collective communication; permanently resident |
| pt_tcpstore_uv | torch | Triggered by distributed init; permanently resident |
| hccp_connect | CANN | Triggered when the HCCL communication is established; destroyed immediately after the connection is established |
| HcclIntra_0 | CANN | Triggered by HCCL intra-node communication; permanently resident |
| hccp_epoll | CANN | Triggered by HCCL communication epoll; permanently resident |
| Hccl_TopoDetect | CANN | Triggered during HCCL timeout; destroyed after detection ends |
| CaffeTaskThread | TorchNPU | Triggered by a large number of Event operations or collective communication; permanently resident |
| pt_autograd_0~7 | torch | Triggered by backpropagation or torch.compile; permanently resident |
| NPUProfiler | TorchNPU | Exists during profiling; reclaimed after profiling ends |
| MSVP_\* | CANN (msprof) | Exists during profiling; reclaimed after profiling ends |
