# Environment Variable List

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T12:03:58.232Z pushedAt=2026-06-16T03:14:22.217Z -->

This manual describes the environment variables that developers can use during training and online inference with Ascend Extension for PyTorch. For environment variables used in building AI apps and services based on CANN, refer to the *[CANN Environment Variable Reference](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/envvar/envref_07_0001.html)*.

**Table 1**  Environment variable list

|Environment Variable Type|Environment Variable Name|Introduction|
|--|--|--|
|Operator Execution|[INF_NAN_MODE_ENABLE](INF_NAN_MODE_ENABLE.md)|Controls the AI processor's handling capability for input data that is Inf/NaN, i.e., controls whether the AI processor uses saturation mode or INF_NAN mode.|
|Operator Execution|[INF_NAN_MODE_FORCE_DISABLE](INF_NAN_MODE_FORCE_DISABLE.md)|Applicable to <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>, forces the INF_NAN mode to be disabled.|
|Operator Execution|[COMBINED_ENABLE](COMBINED_ENABLE.md)|Sets the combined flag.|
|Operator Execution|[ASCEND_LAUNCH_BLOCKING](ASCEND_LAUNCH_BLOCKING.md)|Controls whether to enable synchronous mode during operator execution.|
|Operator Execution|[TASK_QUEUE_ENABLE](TASK_QUEUE_ENABLE.md)|Configures whether to enable the task_queue operator dispatch queue and its optimization level.|
|Operator Execution|[PER_STREAM_QUEUE](PER_STREAM_QUEUE.md)|Configures whether to enable one task_queue operator dispatch queue per stream.|
|Operator Execution|[TORCH_NPU_USE_COMPATIBLE_IMPL](TORCH_NPU_USE_COMPATIBLE_IMPL.md)|Controls whether the API implementation is fully aligned with the native PyTorch community.|
|Operator Compilation|[ACL_OP_COMPILER_CACHE_DIR](ACL_OP_COMPILER_CACHE_DIR.md)|Configures the directory for the operator compilation disk cache.|
|Operator Compilation|[ACL_OP_COMPILER_CACHE_MODE](ACL_OP_COMPILER_CACHE_MODE.md)|Configures the operator compilation disk cache mode.|
|Memory Management|[PYTORCH_NPU_ALLOC_CONF](PYTORCH_NPU_ALLOC_CONF.md)|Controls the behavior of the cache allocator. Configuring this environment variable changes memory usage and may cause performance fluctuations.|
|Memory Management|[PYTORCH_NO_NPU_MEMORY_CACHING](PYTORCH_NO_NPU_MEMORY_CACHING.md)|Configures whether to disable the memory reuse mechanism.|
|Memory Management|[OOM_SNAPSHOT_ENABLE](OOM_SNAPSHOT_ENABLE.md)|Configures whether to save memory data when an out-of-memory error occurs, for analyzing the cause of the memory shortage.|
|Memory Management|[OOM_SNAPSHOT_PATH](OOM_SNAPSHOT_PATH.md)|Configures the save path for memory data when an out-of-memory error occurs.|
|Memory Management|[MULTI_STREAM_MEMORY_REUSE](MULTI_STREAM_MEMORY_REUSE.md)|Configures whether to enable multi-stream memory reuse.|
|Memory Management|[TORCH_NPUGRAPH_GC](TORCH_NPUGRAPH_GC.md)|Controls whether to actively trigger Python GC (Garbage Collection) during the graph capture mode (NPUGraph Capture) process.|
|Collective Communication|[HCCL_ASYNC_ERROR_HANDLING](HCCL_ASYNC_ERROR_HANDLING.md)|When using HCCL as the communication backend, controls whether to enable asynchronous error handling.|
|Collective Communication|[HCCL_DESYNC_DEBUG](HCCL_DESYNC_DEBUG.md)|When using HCCL as the communication backend, controls whether to perform communication timeout analysis.|
|Collective Communication|[HCCL_EVENT_TIMEOUT](HCCL_EVENT_TIMEOUT.md)|When using HCCL as the communication backend, sets the timeout duration for waiting for an event to complete.|
|Collective Communication|[P2P_HCCL_BUFFSIZE](P2P_HCCL_BUFFSIZE.md)|Configures whether to enable point-to-point communication (`torch.distributed.isend`, `torch.distributed.irecv`, and `torch.distributed.batch_isend_irecv`) and use the independent communication domain function.|
|Collective Communication|[RANK_TABLE_FILE](RANK_TABLE_FILE.md)|Configures the path of the RANK_TABLE_FILE file, which is used for collective communication domain link establishment.|
|Collective Communication|[(beta) TORCH_HCCL_ZERO_COPY](（beta）TORCH_HCCL_ZERO_COPY.md)|In training or online inference scenarios, enables the collective communication intra-chip zero-copy function, reducing the number of intra-chip copies for communication operators during the communication process, improving collective communication efficiency, and reducing communication latency. It also reduces contention for memory bandwidth during computation-communication parallel scenarios.|
|Alarm Information Printing|[TORCH_NPU_DISABLED_WARNING](TORCH_NPU_DISABLED_WARNING.md)|Configures whether to print alarm information for Ascend Extension for PyTorch.|
|Alarm Information Printing|[TORCH_NPU_COMPACT_ERROR_OUTPUT](TORCH_NPU_COMPACT_ERROR_OUTPUT.md)|Streamlines the printing of error information. When enabled, custom error information such as the CANN internal call stack and Ascend Extension for PyTorch error codes are transferred to plog, retaining only valid error descriptions to improve the readability of exception information.|
|Alarm Information Printing|[TORCH_NPU_LOGS](TORCH_NPU_LOGS.md)|Configures the log printing function for newly added modules in Ascend Extension for PyTorch, providing developers with precise debugging and locating capabilities in debugging scenarios.|
|Alarm Information Printing|[TORCH_NPU_LOGS_FILTER](TORCH_NPU_LOGS_FILTER.md)|Filters the log output content of Ascend Extension for PyTorch, using a blacklist and whitelist mechanism to select the log information to be displayed, helping developers quickly locate key information among a large volume of logs.|
|Synchronization Timeout|[ACL_DEVICE_SYNC_TIMEOUT](ACL_DEVICE_SYNC_TIMEOUT.md)|Configures the timeout duration for device synchronization.|
|Feature Value Detection|[NPU_ASD_ENABLE](NPU_ASD_ENABLE.md)|For Ascend Extension for PyTorch 7.0.0 and earlier versions, controls whether to enable the eigenvalue detection function of Ascend Extension for PyTorch.|
|Feature Value Detection|[NPU_ASD_UPPER_THRESH](NPU_ASD_UPPER_THRESH.md)|For Ascend Extension for PyTorch 7.0.0 and earlier versions, configures the absolute threshold for the eigenvalue detection function.|
|Feature Value Detection|[NPU_ASD_SIGMA_THRESH](NPU_ASD_SIGMA_THRESH.md)|For Ascend Extension for PyTorch 7.0.0 and earlier versions, configures the relative threshold for the eigenvalue detection function.|
|Feature Value Detection|[NPU_ASD_CONFIG](NPU_ASD_CONFIG.md)|For Ascend Extension for PyTorch 7.1.0 and later versions, controls whether to enable the eigenvalue detection function of Ascend Extension for PyTorch.|
|Performance Optimization|[CPU_AFFINITY_CONF](CPU_AFFINITY_CONF.md)|Ascend Extension for PyTorch can enable coarse/fine-grained core binding by setting the environment variable CPU_AFFINITY_CONF. This configuration helps avoid thread preemption, improve cache hit rates, avoid memory access across NUMA (Non-Uniform Memory Access) nodes, reduce task scheduling overhead, and optimize task execution efficiency.|
|Performance Optimization|[PROF_CONFIG_PATH](PROF_CONFIG_PATH.md)|In PyTorch training scenarios, specifies the path of the profiler_config.json configuration file for the dynamic_profile collection function of the Ascend PyTorch Profiler interface.|
|Performance Optimization|[KINETO_USE_DAEMON](KINETO_USE_DAEMON.md)|Sets whether to enable the dynamic_profile collection function via the msMonitor nputrace method in training scenarios.|
|Device Management|[STREAMS_PER_DEVICE](STREAMS_PER_DEVICE.md)|Configures the maximum number of streams in the stream pool.|
|Device Management|[TORCH_NPU_DEVICE_CAPABILITY](TORCH_NPU_DEVICE_CAPABILITY.md)|Configures the return value of `torch_npu.npu.get_device_capability()`.|
|Device Management|[TORCH_TRANSFER_TO_NPU](TORCH_TRANSFER_TO_NPU.md)|Configures whether to automatically enable the transfer_to_npu function, which automatically replaces PyTorch CUDA-related APIs with corresponding NPU APIs.|
|Graph Mode|[TORCHINDUCTOR_NPU_BACKEND](TORCHINDUCTOR_NPU_BACKEND.md)|Configures the optimization mode in graph mode, supporting optimization modes such as Triton, MLIR, and DVM.|
|Graph Mode|[(beta) INDUCTOR_ASCEND_CHECK_ACCURACY](INDUCTOR_ASCEND_CHECK_ACCURACY.md)|INDUCTOR_ASCEND_CHECK_ACCURACY is an accuracy verification tool provided by Ascend Extension for PyTorch. It automatically detects the numerical accuracy of fused operators only when the torch.compile graph compilation backend is "Inductor" and the mode is "Triton".|
