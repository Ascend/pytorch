# Environment Variable List

This manual describes the environment variables that you can use during TorchNPU training and online inference. For environment variables used in building AI apps and services based on CANN, refer to [CANN Environment Variable Reference](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/maintenref/envvar/envref_07_0001.html).

**Table 1**  Operator execution environment variables

|Environment Variable Name|Description|
|--|--|
|[INF_NAN_MODE_ENABLE](INF_NAN_MODE_ENABLE.md)|Controls how the AI processor handles input data that is Inf/NaN through this environment variable, that is, whether the AI processor uses saturation mode or INF_NAN mode.|
|[INF_NAN_MODE_FORCE_DISABLE](INF_NAN_MODE_FORCE_DISABLE.md)|For Atlas A2 training series/Atlas A3 training series, this environment variable forcibly disables INF_NAN mode.|
|[COMBINED_ENABLE](COMBINED_ENABLE.md)|Sets the combined flag through this environment variable.|
|[ASCEND_LAUNCH_BLOCKING](ASCEND_LAUNCH_BLOCKING.md)|Controls whether synchronous mode is enabled during operator execution through this environment variable.|
|[TASK_QUEUE_ENABLE](TASK_QUEUE_ENABLE.md)|Configures whether the `task_queue` operator dispatch queue is enabled and its optimization level through this environment variable.|
|[PER_STREAM_QUEUE](PER_STREAM_QUEUE.md)|Configures whether to enable one `task_queue` operator dispatch queue per stream through this environment variable.|
|[TORCH_NPU_USE_COMPATIBLE_IMPL](TORCH_NPU_USE_COMPATIBLE_IMPL.md)|Controls whether the API implementation is fully aligned with the PyTorch native community through this environment variable.|

**Table 2**  Operator compilation environment variables

|Environment Variable Name|Description|
|--|--|
|[ACL_OP_COMPILER_CACHE_DIR](ACL_OP_COMPILER_CACHE_DIR.md)|Configures the directory of the operator compilation disk cache through this environment variable.|
|[ACL_OP_COMPILER_CACHE_MODE](ACL_OP_COMPILER_CACHE_MODE.md)|Configures the operator compilation disk cache mode through this environment variable.|

**Table 3**  Memory management environment variables

|Environment Variable Name|Description|
|--|--|
|[PYTORCH_NPU_ALLOC_CONF](PYTORCH_NPU_ALLOC_CONF.md)|Controls the behavior of the cache allocator through this environment variable. Configuring this environment variable changes memory usage and may cause performance fluctuations.|
|[PYTORCH_NO_NPU_MEMORY_CACHING](PYTORCH_NO_NPU_MEMORY_CACHING.md)|Configures whether to disable the memory reuse mechanism through this environment variable.|
|[OOM_SNAPSHOT_ENABLE](OOM_SNAPSHOT_ENABLE.md)|Configures whether to save memory data when an out-of-memory error is reported through this environment variable, for analyzing the cause of the memory shortage.|
|[OOM_SNAPSHOT_PATH](OOM_SNAPSHOT_PATH.md)|Configures the path for saving memory data when an out-of-memory error is reported through this environment variable.|
|[MULTI_STREAM_MEMORY_REUSE](MULTI_STREAM_MEMORY_REUSE.md)|Configures whether multi-stream memory reuse is enabled through this environment variable.|
|[TORCH_NPUGRAPH_GC](TORCH_NPUGRAPH_GC.md)|Controls whether Python GC (Garbage Collection) is actively triggered during graph capture mode (NPUGraph Capture) through this environment variable.|

**Table 4**  Collective communication environment variables

|Environment Variable Name|Description|
|--|--|
|[TORCH_HCCL_BLOCKING_WAIT](TORCH_HCCL_BLOCKING_WAIT.md)|When HCCL is used as the communication backend, controls the synchronous mode (blocking or non-blocking) of `wait()` and `synchronize()` in `ProcessGroupHCCL` through this environment variable.|
|[HCCL_ASYNC_ERROR_HANDLING](HCCL_ASYNC_ERROR_HANDLING.md)|When HCCL is used as the communication backend, controls whether asynchronous error handling is enabled through this environment variable.|
|[HCCL_DESYNC_DEBUG](HCCL_DESYNC_DEBUG.md)|When HCCL is used as the communication backend, controls whether communication timeout analysis is performed through this environment variable.|
|[HCCL_EVENT_TIMEOUT](HCCL_EVENT_TIMEOUT.md)|When HCCL is used as the communication backend, sets the timeout duration for waiting for an event to complete through this environment variable.|
|[P2P_HCCL_BUFFSIZE](P2P_HCCL_BUFFSIZE.md)|Configures whether to enable point-to-point communication (`torch.distributed.isend`, `torch.distributed.irecv`, and `torch.distributed.batch_isend_irecv`) and use the independent communication domain feature through this environment variable.|
|[RANK_TABLE_FILE](RANK_TABLE_FILE.md)|Configures the path of the RANK_TABLE_FILE file through this environment variable, which is used for collective communication domain link establishment.|
|[(beta) TORCH_HCCL_ZERO_COPY](beta_TORCH_HCCL_ZERO_COPY.md)|In training or online inference scenarios, this environment variable enables the collective communication intra-chip zero-copy feature, reducing the number of intra-chip copies of communication operators during communication, improving collective communication efficiency, and reducing communication latency. In compute-communication parallel scenarios, it also reduces the contention for memory bandwidth during communication.|

**Table 5**  Alarm message printing environment variables

|Environment Variable Name|Description|
|--|--|
|[TORCH_NPU_DISABLED_WARNING](TORCH_NPU_DISABLED_WARNING.md)|Configures whether to print TorchNPU alarm information through this environment variable.|
|[TORCH_NPU_COMPACT_ERROR_OUTPUT](TORCH_NPU_COMPACT_ERROR_OUTPUT.md)|Streamlines error message printing through this environment variable. When enabled, custom error information such as the CANN internal call stack and TorchNPU error codes is moved to plog, retaining only valid error descriptions and improving the readability of exception information.|
|[TORCH_NPU_LOGS](TORCH_NPU_LOGS.md)|This environment variable is used to configure the log printing function of new TorchNPU modules, providing you with precise debugging and locating capabilities in debugging scenarios.|
|[TORCH_NPU_LOGS_FILTER](TORCH_NPU_LOGS_FILTER.md)|This environment variable is used to filter TorchNPU log output content. It selects the log information to be displayed through a blacklist and whitelist mechanism, helping you quickly locate key information in a large number of logs.|

**Table 6**  Synchronization timeout environment variables

|Environment Variable Name|Description|
|--|--|
|[ACL_DEVICE_SYNC_TIMEOUT](ACL_DEVICE_SYNC_TIMEOUT.md)|Configures the timeout duration for device synchronization through this environment variable.|

**Table 7**  Eigenvalue detection environment variables

|Environment Variable Name|Description|
|--|--|
|[NPU_ASD_ENABLE](NPU_ASD_ENABLE.md)|For TorchNPU 7.0.0 and earlier versions, controls whether the eigenvalue detection feature is enabled through this environment variable.|
|[NPU_ASD_UPPER_THRESH](NPU_ASD_UPPER_THRESH.md)|For TorchNPU 7.0.0 and earlier versions, configures the absolute threshold of the eigenvalue detection feature through this environment variable.|
|[NPU_ASD_SIGMA_THRESH](NPU_ASD_SIGMA_THRESH.md)|For TorchNPU 7.0.0 and earlier versions, configures the relative threshold of the eigenvalue detection feature through this environment variable.|
|[NPU_ASD_CONFIG](NPU_ASD_CONFIG.md)|For TorchNPU 7.1.0 and later versions, controls whether the TorchNPU eigenvalue detection feature is enabled through this environment variable.|

**Table 8**  Performance tuning environment variables

|Environment Variable Name|Description|
|--|--|
|[CPU_AFFINITY_CONF](CPU_AFFINITY_CONF.md)|TorchNPU can enable coarse-grained or fine-grained core binding by setting the environment variable `CPU_AFFINITY_CONF`. This configuration prevents inter-thread preemption, improves cache hits, avoids memory access across NUMA (Non-Uniform Memory Access) nodes, reduces task scheduling overhead, and optimizes task execution efficiency.|
|[PROF_CONFIG_PATH](PROF_CONFIG_PATH.md)|In TorchNPU training scenarios, specifies the path of the profiler_config.json configuration file for the `dynamic_profile` collection feature of the TorchNPU Profiler interface through this environment variable.|
|[KINETO_USE_DAEMON](KINETO_USE_DAEMON.md)|Sets whether to enable the `dynamic_profile` collection feature through the msMonitor nputrace method in training scenarios.|
|[TORCH_NPU_LAZY_FUSION](TORCH_NPU_LAZY_FUSION.md)|Enables DVM operator fusion through this environment variable, performing cross-operator fusion on elementwise and activation function operators to reduce kernel launches and HBM transfers, accelerating training and inference.|

**Table 9**  Device management environment variables

|Environment Variable Name|Description|
|--|--|
|[STREAMS_PER_DEVICE](STREAMS_PER_DEVICE.md)|Configures the maximum number of streams in the stream pool through this environment variable.|
|[TORCH_NPU_DEVICE_CAPABILITY](TORCH_NPU_DEVICE_CAPABILITY.md)|Configures the return value of `torch_npu.npu.get_device_capability()` through this environment variable.|
|[TORCH_TRANSFER_TO_NPU](TORCH_TRANSFER_TO_NPU.md)|Configures whether to automatically enable the transfer_to_npu feature through this environment variable, which automatically replaces PyTorch CUDA-related APIs with the corresponding NPU APIs.|

**Table 10**  Graph mode environment variables

|Environment Variable Name|Description|
|--|--|
|[TORCHINDUCTOR_NPU_BACKEND](TORCHINDUCTOR_NPU_BACKEND.md)|Configures the optimization mode in graph mode through this environment variable, supporting Triton, MLIR, DVM, and other optimization modes.|
|[（beta）INDUCTOR_ASCEND_CHECK_ACCURACY](INDUCTOR_ASCEND_CHECK_ACCURACY.md)|INDUCTOR_ASCEND_CHECK_ACCURACY is an accuracy verification tool provided by TorchNPU. It automatically detects the numerical accuracy of fused operators only when the torch.compile graph compilation backend is "Inductor" and the mode is "Triton".|
|[NPU_INDUCTOR_FALLBACK_LIST](NPU_INDUCTOR_FALLBACK_LIST.md)|Specifies the list of operators that need to fall back to native PyTorch through this environment variable.|
|[（beta）TORCHINDUCTOR_ENABLE_MFUSION](TORCHINDUCTOR_ENABLE_MFUSION.md)|Controls whether the MFusion fusion optimization feature is enabled through this environment variable. It takes effect only when the torch.compile graph compilation backend is "Inductor".|
|[TORCHINDUCTOR_USE_AKG](TORCHINDUCTOR_USE_AKG.md)|Configures whether AKG (Auto Kernel Generator) backend optimization is enabled in MLIR (Multi-Level Intermediate Representation) mode under the torch.compile graph mode (Inductor) through this environment variable.|
