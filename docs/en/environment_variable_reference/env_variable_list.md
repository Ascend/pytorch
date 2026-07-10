# Environment Variable List

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-08T10:23:52.483Z pushedAt=2026-07-08T10:47:16.883Z -->

This manual describes the environment variables that developers can use during training and online inference with Ascend Extension for PyTorch. For environment variables used in building AI apps and services based on CANN, refer to [CANN Environment Variable Reference](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/envvar/envref_07_0001.html).

**Table 1**  Environment variables

|Environment Variable Type|Environment Variable Name|Description|
|--|--|--|
|Operator Execution|[INF_NAN_MODE_ENABLE](INF_NAN_MODE_ENABLE.md)|Controls the AI processor's handling of Inf/NaN input data, i.e., whether the AI processor uses saturation mode or INF_NAN mode.|
|Operator Execution|[INF_NAN_MODE_FORCE_DISABLE](INF_NAN_MODE_FORCE_DISABLE.md)|For Atlas A2 training products/Atlas A3 training products, forces the INF_NAN mode to be disabled.|
|Operator Execution|[COMBINED_ENABLE](COMBINED_ENABLE.md)|Sets the combined flag.|
|Operator Execution|[ASCEND_LAUNCH_BLOCKING](ASCEND_LAUNCH_BLOCKING.md)|Controls whether to enable synchronous mode during operator execution.|
|Operator Execution|[TASK_QUEUE_ENABLE](TASK_QUEUE_ENABLE.md)|Configures whether the task_queue operator dispatch queue is enabled and its optimization level.|
|Operator Execution|[PER_STREAM_QUEUE](PER_STREAM_QUEUE.md)|Configures whether to enable one task_queue operator dispatch queue per stream.|
|Operator Execution|[TORCH_NPU_USE_COMPATIBLE_IMPL](TORCH_NPU_USE_COMPATIBLE_IMPL.md)|Controls whether the API implementation is fully aligned with the native PyTorch community.|
|Operator Compilation|[ACL_OP_COMPILER_CACHE_DIR](ACL_OP_COMPILER_CACHE_DIR.md)|Configures the directory for the operator compilation disk cache.|
|Operator Compilation|[ACL_OP_COMPILER_CACHE_MODE](ACL_OP_COMPILER_CACHE_MODE.md)|Configures the operator compilation disk cache mode.|
|Memory Management|[PYTORCH_NPU_ALLOC_CONF](PYTORCH_NPU_ALLOC_CONF.md)|Controls the behavior of the cache allocator. Configuring this environment variable changes memory usage and may cause performance fluctuations.|
|Memory Management|[PYTORCH_NO_NPU_MEMORY_CACHING](PYTORCH_NO_NPU_MEMORY_CACHING.md)|Configures whether to disable the memory reuse mechanism.|
|Memory Management|[OOM_SNAPSHOT_ENABLE](OOM_SNAPSHOT_ENABLE.md)|Configures whether to save memory data when an out-of-memory error occurs, for analyzing the cause of the memory shortage.|
|Memory Management|[OOM_SNAPSHOT_PATH](OOM_SNAPSHOT_PATH.md)|Configures the save path for memory data when an out-of-memory error occurs.|
|Memory Management|[MULTI_STREAM_MEMORY_REUSE](MULTI_STREAM_MEMORY_REUSE.md)|Configures whether multi-stream memory reuse is enabled.|
|Memory Management|[TORCH_NPUGRAPH_GC](TORCH_NPUGRAPH_GC.md)|Controls whether Python GC (Garbage Collection) is actively triggered during the graph capture mode (NPUGraph Capture) process.|
|Collective Communication|[HCCL_ASYNC_ERROR_HANDLING](HCCL_ASYNC_ERROR_HANDLING.md)|When HCCL is used as the communication backend, controls whether asynchronous error handling is enabled.|
|Collective Communication|[HCCL_DESYNC_DEBUG](HCCL_DESYNC_DEBUG.md)|When HCCL is used as the communication backend, controls whether communication timeout analysis is performed.|
|Collective Communication|[HCCL_EVENT_TIMEOUT](HCCL_EVENT_TIMEOUT.md)|When HCCL is used as the communication backend, sets the timeout duration for waiting for event completion.|
|Collective Communication|[P2P_HCCL_BUFFSIZE](P2P_HCCL_BUFFSIZE.md)|Configures whether to enable point-to-point communication (`torch.distributed.isend`, `torch.distributed.irecv`, and `torch.distributed.batch_isend_irecv`) and use the independent communication domain feature.|
|Collective Communication|[RANK_TABLE_FILE](RANK_TABLE_FILE.md)|Configures the path to the RANK_TABLE_FILE file, which is used for collective communication domain link establishment.|
|Collective Communication|[(beta) TORCH_HCCL_ZERO_COPY](（beta）TORCH_HCCL_ZERO_COPY.md)|In training or online inference scenarios, enables the collective communication intra-chip zero-copy feature, reducing the number of intra-chip copies during communication operator execution, improving collective communication efficiency, and reducing communication latency. In compute-communication parallel scenarios, it also reduces contention for memory bandwidth during communication.|
|Alarm Information|[TORCH_NPU_DISABLED_WARNING](TORCH_NPU_DISABLED_WARNING.md)|Configures whether to print alarm information for Ascend Extension for PyTorch.|
|Alarm Information|[TORCH_NPU_COMPACT_ERROR_OUTPUT](TORCH_NPU_COMPACT_ERROR_OUTPUT.md)|Streamlines error message output. When enabled, custom error information such as the CANN internal call stack and Ascend Extension for PyTorch error codes is redirected to plog, retaining only the effective error description and improving the readability of exception information.|
|Alarm Information|[TORCH_NPU_LOGS](TORCH_NPU_LOGS.md)|Configures the log printing functionality for newly added modules in Ascend Extension for PyTorch, providing developers with precise debugging and localization capabilities in debugging scenarios.|
|Alarm Information|[TORCH_NPU_LOGS_FILTER](TORCH_NPU_LOGS_FILTER.md)|Filters the log output content of Ascend Extension for PyTorch. It uses a blacklist and whitelist mechanism to select the log information to be displayed, helping developers quickly locate key information among a large volume of logs.|
|Synchronization Timeout|[ACL_DEVICE_SYNC_TIMEOUT](ACL_DEVICE_SYNC_TIMEOUT.md)|Configures the timeout duration for device synchronization.|
|Eigenvalue Detection|[NPU_ASD_ENABLE](NPU_ASD_ENABLE.md)|For Ascend Extension for PyTorch 7.0.0 and earlier versions, controls whether to enable the eigenvalue detection feature of Ascend Extension for PyTorch.|
|Eigenvalue Detection|[NPU_ASD_UPPER_THRESH](NPU_ASD_UPPER_THRESH.md)|For Ascend Extension for PyTorch 7.0.0 and earlier versions, configures the absolute threshold for the eigenvalue detection feature.|
|Eigenvalue Detection|[NPU_ASD_SIGMA_THRESH](NPU_ASD_SIGMA_THRESH.md)|For Ascend Extension for PyTorch 7.0.0 and earlier versions, configures the relative threshold for the eigenvalue detection feature.|
|Eigenvalue Detection|[NPU_ASD_CONFIG](NPU_ASD_CONFIG.md)|For Ascend Extension for PyTorch 7.1.0 and later versions, controls whether to enable the eigenvalue detection feature of Ascend Extension for PyTorch.|
|Performance Optimization|[CPU_AFFINITY_CONF](CPU_AFFINITY_CONF.md)|Ascend Extension for PyTorch can enable coarse-grained or fine-grained CPU core binding by setting the environment variable CPU_AFFINITY_CONF. This configuration prevents inter-thread preemption, improves cache hit rates, avoids cross-NUMA (Non-Uniform Memory Access) node memory access, reduces task scheduling overhead, and optimizes task execution efficiency.|
|Performance Optimization|[PROF_CONFIG_PATH](PROF_CONFIG_PATH.md)|In PyTorch training scenarios, specifies the path to the profiler_config.json configuration file for the dynamic_profile collection feature of the Ascend PyTorch Profiler interface.|
|Performance Optimization|[KINETO_USE_DAEMON](KINETO_USE_DAEMON.md)|Sets whether to enable the dynamic_profile collection feature via the msMonitor nputrace method in training scenarios.|
|Device Management|[STREAMS_PER_DEVICE](STREAMS_PER_DEVICE.md)|Configures the maximum number of streams in the stream pool.|
|Device Management|[TORCH_NPU_DEVICE_CAPABILITY](TORCH_NPU_DEVICE_CAPABILITY.md)|Configures the return value of `torch_npu.npu.get_device_capability()`.|
|Device Management|[TORCH_TRANSFER_TO_NPU](TORCH_TRANSFER_TO_NPU.md)|Configures whether to automatically enable the transfer_to_npu feature, which replaces PyTorch CUDA-related APIs with the corresponding NPU APIs.|
|Graph Mode|[TORCHINDUCTOR_NPU_BACKEND](TORCHINDUCTOR_NPU_BACKEND.md)|Configures the optimization mode in graph mode, supporting Triton, MLIR, DVM, and other optimization modes.|
|Graph Mode|[(beta) INDUCTOR_ASCEND_CHECK_ACCURACY](INDUCTOR_ASCEND_CHECK_ACCURACY.md)|INDUCTOR_ASCEND_CHECK_ACCURACY is an accuracy verification tool provided by Ascend Extension for PyTorch. It automatically detects the numerical accuracy of fused operators only when the torch.compile graph compilation backend is "Inductor" and the mode is "Triton".|
