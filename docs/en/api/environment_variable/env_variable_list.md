# Environment Variable List

This manual describes the environment variables that you can use during TorchNPU training and online inference. For environment variables used in building AI apps and services based on CANN, refer to [CANN Environment Variable Reference](https://gitcode.com/cann/docs/blob/9.2.0/docs/en/env-vars/README.md).

Environment variables are categorized by usage scenario. Each item is assigned to a major category in the list; refer to the corresponding detail page for the value range, default behavior, when to set, and supported models.

For the correspondence and differences with PyTorch, see [PyTorch Environment Variable Comparison Table](pytorch_comparison.md).

## Category Navigation

| Category | Main Purpose |
| --- | --- |
| [Device Initialization and Management](device_management/_menu_device_management.md) | Configure device initialization, availability detection, stream resources, and device synchronization waiting. |
| [Operator Execution and Compatibility](op_execution/_menu_op_execution.md) | Control operator dispatch, synchronous execution, numerical modes, overflow detection, implementation switching, and CPU fallback behavior. |
| [Single-Operator Compilation and Caching](op_compilation/_menu_op_compilation.md) | Configure the compilation cache for single-operator mode; see the Graph Compilation category for torch.compile configurations. |
| [Memory Management](memory_management/_menu_memory_management.md) | Configure the caching allocator, memory reuse, OOM snapshots, and symmetric memory. |
| [Distributed Communication and Diagnosis](collective_communication/_menu_collective_communication.md) | Configure the HCCL process group, communication synchronization, timeout diagnosis, and status collection. |
| [Logs and Alarms](alarm_message_printing/_menu_alarm_message_printing.md) | Configure TorchNPU general logs, alarms, and error output; dedicated diagnosis configurations for communication and graph compilation are retained in the corresponding categories. |
| [Feature Value Detection](eigenvalue_detection/_menu_eigenvalue_detection.md) | Configure the eigenvalue detection feature and its detection thresholds. |
| [Performance Optimization and Profiling](performance_tuning/_menu_performance_tuning.md) | Configure CPU core binding, Eager operator fusion, and dynamic Profiler collection. |
| [Graph Compilation (torch.compile / Inductor)](inductor/_menu_inductor.md) | Look up backend selection, caching, fusion tuning, and debugging diagnosis configurations by compilation workflow. |

## Device Initialization and Management

Configure device initialization, availability detection, stream resources, and device synchronization waiting.

| Environment Variable Name | Description |
| --- | --- |
|[TORCH_ACL_INIT_CONFIG_PATH](device_management/TORCH_ACL_INIT_CONFIG_PATH.md)|Use this environment variable to specify the path of the aclInit initialization configuration file, used to pass custom ACL JSON configurations during NPU initialization.|
|[PYTORCH_HAL_BASED_NPU_CHECK](device_management/PYTORCH_HAL_BASED_NPU_CHECK.md)|Select the NPU availability detection path. When set to 1, the number of devices is queried through Ascend HAL first and falls back to Runtime if the query fails.|
|[STREAMS_PER_DEVICE](device_management/STREAMS_PER_DEVICE.md)|Use this environment variable to configure the maximum number of streams in the stream pool.|
|[ACL_DEVICE_SYNC_TIMEOUT](device_management/ACL_DEVICE_SYNC_TIMEOUT.md)|Use this environment variable to configure the timeout for device synchronization.|
|[TORCH_NPU_DEVICE_CAPABILITY](device_management/TORCH_NPU_DEVICE_CAPABILITY.md)|Use this environment variable to configure the return value of `torch_npu.npu.get_device_capability()`.|
|[TORCH_TRANSFER_TO_NPU](device_management/TORCH_TRANSFER_TO_NPU.md)|Use this environment variable to configure whether to automatically enable the transfer_to_npu feature, which automatically replaces PyTorch CUDA-related APIs with their NPU counterparts.|

## Operator Execution and Compatibility

Control operator dispatch, synchronous execution, numerical modes, overflow detection, implementation switching, and CPU fallback behavior.

| Environment Variable Name | Description |
| --- | --- |
|[INF_NAN_MODE_ENABLE](op_execution/INF_NAN_MODE_ENABLE.md)|Use this environment variable to control how the AI processor handles input data that is Inf/NaN, that is, whether the AI processor uses saturation mode or INF_NAN mode.|
|[INF_NAN_MODE_FORCE_DISABLE](op_execution/INF_NAN_MODE_FORCE_DISABLE.md)|<term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>: use this environment variable to forcibly disable INF_NAN mode.|
|[FORCE_OVERFLOW_CHECK](op_execution/FORCE_OVERFLOW_CHECK.md)|Use this environment variable to enable the overflow detection switch in non-saturation mode (INF_NAN mode) for asynchronous location of Inf/NaN problems.|
|[COMBINED_ENABLE](op_execution/COMBINED_ENABLE.md)|Use this environment variable to control whether the combined contiguity optimization is enabled, which optimizes the contiguity conversion of non-contiguous tensors generated by multiple view operations.|
|[ASCEND_LAUNCH_BLOCKING](op_execution/ASCEND_LAUNCH_BLOCKING.md)|Use this environment variable to control whether synchronous mode is enabled during operator execution.|
|[TASK_QUEUE_ENABLE](op_execution/TASK_QUEUE_ENABLE.md)|Use this environment variable to configure whether the task_queue operator dispatch queue is enabled and its optimization level.|
|[PER_STREAM_QUEUE](op_execution/PER_STREAM_QUEUE.md)|Use this environment variable to configure whether to enable one task_queue operator dispatch queue per stream.|
|[TORCH_NPU_FALLBACK_CPU_DISABLE](op_execution/TORCH_NPU_FALLBACK_CPU_DISABLE.md)|Use this environment variable to control whether the implicit CPU fallback path managed by TorchNPU is allowed to execute.|
|[TORCH_NPU_USE_COMPATIBLE_IMPL](op_execution/TORCH_NPU_USE_COMPATIBLE_IMPL.md)|This environment variable controls whether the API implementation is fully aligned with the native PyTorch community.|
|[TORCH_NPU_LEGACY_IMPL_LIST](op_execution/TORCH_NPU_LEGACY_IMPL_LIST.md)|Use this environment variable to specify configuration items that need to use the legacy implementation.|

## Single-Operator Compilation and Caching

Configure the compilation cache for single-operator mode; see the Graph Compilation category for torch.compile configurations.

| Environment Variable Name | Description |
| --- | --- |
|[ACL_OP_COMPILER_CACHE_DIR](op_compilation/ACL_OP_COMPILER_CACHE_DIR.md)|Use this environment variable to configure the directory of the operator compilation disk cache.|
|[ACL_OP_COMPILER_CACHE_MODE](op_compilation/ACL_OP_COMPILER_CACHE_MODE.md)|Use this environment variable to configure the mode of the operator compilation disk cache.|

## Memory Management

Configure the caching allocator, memory reuse, OOM snapshots, and symmetric memory.

| Environment Variable Name | Description |
| --- | --- |
|[PYTORCH_ALLOC_CONF](memory_management/PYTORCH_ALLOC_CONF.md)|Unified entry for configuring the caching allocator, with the same function as PYTORCH_NPU_ALLOC_CONF. The two cannot be set at the same time.|
|[PYTORCH_NPU_ALLOC_CONF](memory_management/PYTORCH_NPU_ALLOC_CONF.md)|Use this environment variable to control the caching allocator behavior. Configuring this environment variable changes memory usage and may cause performance fluctuations.|
|[PYTORCH_NO_NPU_MEMORY_CACHING](memory_management/PYTORCH_NO_NPU_MEMORY_CACHING.md)|Use this environment variable to configure whether to disable the memory reuse mechanism.|
|[OOM_SNAPSHOT_ENABLE](memory_management/OOM_SNAPSHOT_ENABLE.md)|Use this environment variable to configure whether to save memory data when an out-of-memory error is reported, for analyzing the cause of the out-of-memory error.|
|[OOM_SNAPSHOT_PATH](memory_management/OOM_SNAPSHOT_PATH.md)|Use this environment variable to configure the path for saving memory data when an out-of-memory error is reported.|
|[MULTI_STREAM_MEMORY_REUSE](memory_management/MULTI_STREAM_MEMORY_REUSE.md)|Use this environment variable to configure whether multi-stream memory reuse is enabled.|
|[TORCH_NPUGRAPH_GC](memory_management/TORCH_NPUGRAPH_GC.md)|Use this environment variable to control whether Python GC (Garbage Collection) is proactively triggered during graph capture mode (NPUGraph Capture).|
|[NPU_SHMEM_SYMMETRIC_SIZE](memory_management/NPU_SHMEM_SYMMETRIC_SIZE.md)|Use this environment variable to configure the heap size of NPU symmetric memory, used for direct memory access between devices.|

## Distributed Communication and Diagnosis

Configure the HCCL process group, communication synchronization, timeout diagnosis, and status collection.

### Communication Execution and Synchronization

| Environment Variable Name | Description |
| --- | --- |
|[TORCH_HCCL_BLOCKING_WAIT](collective_communication/TORCH_HCCL_BLOCKING_WAIT.md)|When HCCL is used as the communication backend, use this environment variable to control the synchronization mode (blocking or non-blocking) of `wait()` and `synchronize()` in `ProcessGroupHCCL`.|
|[TORCH_HCCL_ASYNC_ERROR_HANDLING](collective_communication/TORCH_HCCL_ASYNC_ERROR_HANDLING.md)|When HCCL is used as the communication backend, use this environment variable to control whether asynchronous error handling is enabled.|
|[HCCL_EVENT_TIMEOUT](collective_communication/HCCL_EVENT_TIMEOUT.md)|When HCCL is used as the communication backend, use this environment variable to set the timeout for waiting for an event to complete.|
|[P2P_HCCL_BUFFSIZE](collective_communication/P2P_HCCL_BUFFSIZE.md)|Use this environment variable to configure whether to enable point-to-point communication (`torch.distributed.isend`, `torch.distributed.irecv`, and `torch.distributed.batch_isend_irecv`) and use the independent communicator feature.|
|[RANK_TABLE_FILE](collective_communication/RANK_TABLE_FILE.md)|Use this environment variable to configure the path of the RANK_TABLE_FILE file, used for link establishment in the collective communicator.|
|[ROOTINFO_SCALABLE_ENABLE](collective_communication/ROOTINFO_SCALABLE_ENABLE.md)|Use this environment variable to control whether Scalable RootInfo hierarchical link establishment is enabled.|
|[TORCH_HCCL_RANKS_PER_ROOT](collective_communication/TORCH_HCCL_RANKS_PER_ROOT.md)|After Scalable RootInfo is enabled, use this environment variable to configure the number of ranks that each root is expected to manage.|
|[(beta) TORCH_HCCL_ZERO_COPY](collective_communication/（beta）TORCH_HCCL_ZERO_COPY.md)|In training or online inference scenarios, you can use this environment variable to enable the on-chip zero-copy feature of collective communication, reducing the number of on-chip copies performed by communication operators during communication, improving collective communication efficiency, and reducing communication latency. In computing-communication parallel scenarios, it also reduces the preemption of memory bandwidth during communication.|

### Timeout Monitoring and Fault Recording

| Environment Variable Name | Description |
| --- | --- |
|[TORCH_HCCL_DESYNC_DEBUG](collective_communication/TORCH_HCCL_DESYNC_DEBUG.md)|When HCCL is used as the communication backend, use this environment variable to control whether communication timeout analysis is performed.|

### Status Profiling

| Environment Variable Name | Description |
| --- | --- |
|[TORCH_HCCL_STATUS_SAVE_ENABLE](collective_communication/TORCH_HCCL_STATUS_SAVE_ENABLE.md)|Use this environment variable to control the periodic saving of HCCL process group status information.|
|[TORCH_HCCL_STATUS_SAVE_PATH](collective_communication/TORCH_HCCL_STATUS_SAVE_PATH.md)|Use this environment variable to configure the directory for saving HCCL status files.|
|[TORCH_HCCL_STATUS_SAVE_INTERVAL](collective_communication/TORCH_HCCL_STATUS_SAVE_INTERVAL.md)|Use this environment variable to configure the interval for saving HCCL status.|

## Logs and Alarms

Configure TorchNPU general logs, alarms, and error output; dedicated diagnosis configurations for communication and graph compilation are retained in the corresponding categories.

| Environment Variable Name | Description |
| --- | --- |
|[TORCH_NPU_WARNING_DISABLE](alarm_message_printing/TORCH_NPU_WARNING_DISABLE.md)|Use this environment variable to configure whether TorchNPU alarm information is printed.|
|[TORCH_NPU_DISABLED_WARNING](alarm_message_printing/TORCH_NPU_DISABLED_WARNING.md)|This environment variable is deprecated. It is recommended to use TORCH_NPU_WARNING_DISABLE instead. It is used to configure whether TorchNPU alarm information is printed.|
|[TORCH_NPU_COMPACT_ERROR_OUTPUT](alarm_message_printing/TORCH_NPU_COMPACT_ERROR_OUTPUT.md)|Use this environment variable to simplify printed error information. After it is enabled, custom error information such as CANN internal call stacks and TorchNPU error codes is moved to plog, retaining only valid error descriptions to improve the readability of exception information.|
|[TORCH_NPU_LOGS](alarm_message_printing/TORCH_NPU_LOGS.md)|This environment variable is used to configure the log printing function of the newly added TorchNPU module, providing developers with precise debugging and locating capabilities in debugging scenarios.|
|[TORCH_NPU_LOGS_FILTER](alarm_message_printing/TORCH_NPU_LOGS_FILTER.md)|This environment variable is used to filter TorchNPU log output content. The whitelist/blacklist mechanism is used to filter the log information to be displayed, helping developers quickly locate key information among a large number of logs.|

## Feature Value Detection

Configure the eigenvalue detection feature and its detection thresholds.

| Environment Variable Name | Description |
| --- | --- |
|[NPU_ASD_ENABLE](eigenvalue_detection/NPU_ASD_ENABLE.md)|In TorchNPU 7.0.0 and earlier versions, use this environment variable to control whether the eigenvalue detection feature is enabled.|
|[NPU_ASD_UPPER_THRESH](eigenvalue_detection/NPU_ASD_UPPER_THRESH.md)|In TorchNPU 7.0.0 and earlier versions, use this environment variable to configure the absolute threshold of the eigenvalue detection feature.|
|[NPU_ASD_SIGMA_THRESH](eigenvalue_detection/NPU_ASD_SIGMA_THRESH.md)|In TorchNPU 7.0.0 and earlier versions, use this environment variable to configure the relative threshold of the eigenvalue detection feature.|
|[NPU_ASD_CONFIG](eigenvalue_detection/NPU_ASD_CONFIG.md)|In TorchNPU 7.1.0 and later versions, use this environment variable to control whether the TorchNPU eigenvalue detection feature is enabled.|

## Performance Optimization and Profiling

Configure CPU core binding, Eager operator fusion, and dynamic Profiler collection.

### Execution Optimization

| Environment Variable Name | Description |
| --- | --- |
|[CPU_AFFINITY_CONF](performance_tuning/CPU_AFFINITY_CONF.md)|TorchNPU can enable coarse/fine-grained core binding by setting the CPU_AFFINITY_CONF environment variable. This configuration can avoid thread preemption, improve the cache hit rate, avoid memory access across NUMA (Non-Uniform Memory Access) nodes, reduce task scheduling overhead, and optimize task execution efficiency.|
|[TORCH_NPU_LAZY_FUSION](performance_tuning/TORCH_NPU_LAZY_FUSION.md)|Use this environment variable to enable DVM operator fusion, which performs cross-operator fusion on operators such as elementwise and activation functions, reducing kernel launches and HBM transfers and accelerating training and inference.|

### Profiler Profiling

| Environment Variable Name | Description |
| --- | --- |
|[PROF_CONFIG_PATH](performance_tuning/PROF_CONFIG_PATH.md)|In TorchNPU training scenarios, use this environment variable to specify the path of the profiler_config.json configuration file of the dynamic_profile collection function of the TorchNPU Profiler interface.|
|[KINETO_USE_DAEMON](performance_tuning/KINETO_USE_DAEMON.md)|This environment variable is used in training scenarios to set whether to enable the dynamic_profile collection function through the msMonitor nputrace method.|

## Graph Compilation (torch.compile / Inductor)

Look up backend selection, caching, fusion tuning, and debugging diagnosis configurations by compilation workflow.

### Compilation Modes and Fallback

| Environment Variable Name | Description |
| --- | --- |
|[TORCHINDUCTOR_NPU_BACKEND](inductor/TORCHINDUCTOR_NPU_BACKEND.md)|Use this environment variable to configure the optimization mode in graph mode, supporting optimization modes such as Triton, DVM, and Ascend C.|

### Caching and Compilation Concurrency

| Environment Variable Name | Description |
| --- | --- |
|[TORCH_CACHING_PRECOMPILE](inductor/TORCH_CACHING_PRECOMPILE.md)|Use this environment variable to enable the experimental automatic cache precompilation feature, which automatically saves and loads Dynamo compilation caches to accelerate subsequent compilation, consistent with upstream PyTorch behavior.|
|[TORCHINDUCTOR_COMPILE_THREADS](inductor/TORCHINDUCTOR_COMPILE_THREADS.md)|Use this environment variable to configure the number of processes for concurrent compilation, consistent with upstream PyTorch behavior.|
|[TORCHNPU_PRECOMPILE_THREADS](inductor/TORCHNPU_PRECOMPILE_THREADS.md)|Use this environment variable to configure the number of precompilation threads of the torch_npu Inductor.|

### Operator Fusion and Automatic Tuning

| Environment Variable Name | Description |
| --- | --- |
|[TORCHINDUCTOR_MAX_AUTOTUNE](inductor/TORCHINDUCTOR_MAX_AUTOTUNE.md)|Use this environment variable to control whether the max autotune feature is enabled, consistent with upstream PyTorch behavior.|
|[TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS](inductor/TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS.md)|Use this environment variable to configure the list of candidate implementations for matrix multiplication operators to participate in tuning during the max autotune process.|
|[TORCHINDUCTOR_NPU_CATLASS_DIR](inductor/TORCHINDUCTOR_NPU_CATLASS_DIR.md)|Use this environment variable to configure the path of the Catlass template library, corresponding to `TORCHINDUCTOR_CUTLASS_DIR` in upstream PyTorch.|
|[TORCHINDUCTOR_CATLASS_ENABLED_OPS](inductor/TORCHINDUCTOR_CATLASS_ENABLED_OPS.md)|Use this environment variable to configure the list of matrix multiplication type operators supported by the Catlass template library, corresponding to `TORCHINDUCTOR_CUTLASS_ENABLED_OPS` in upstream PyTorch.|
|[CATLASS_EPILOGUE_FUSION](inductor/CATLASS_EPILOGUE_FUSION.md)|Use this environment variable to control whether the Catlass epilogue fusion feature is enabled, corresponding to `CUTLASS_EPILOGUE_FUSION` in upstream PyTorch.|
|[TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING](inductor/TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING.md)|Use this environment variable to control whether profiling is used for performance measurement during autotune, consistent with upstream PyTorch behavior.|
|[INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE](inductor/INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE.md)|Use this environment variable to control whether the batch profiler is enabled for batch performance measurement during autotune.|
|[INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE](inductor/INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE.md)|Use this environment variable to control whether dynamic shape grouping autotune is enabled, reusing tuning results grouped by shape features.|
|[INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES](inductor/INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES.md)|Use this environment variable to configure the list of template types that participate in dynamic shape grouping autotune.|
|[INDUCTOR_ASCEND_ENABLE_COSTMODEL](inductor/INDUCTOR_ASCEND_ENABLE_COSTMODEL.md)|Use this environment variable to control whether the CostModel pre-screening feature is enabled, reducing the number of configs for subsequent compilation and actual profiling.|
|[INDUCTOR_ASCEND_COSTMODEL_RATIO](inductor/INDUCTOR_ASCEND_COSTMODEL_RATIO.md)|Use this environment variable to control the proportion of configs retained after CostModel pre-screening.|

### Graph Optimization and Scheduling

| Environment Variable Name | Description |
| --- | --- |
|[ENABLE_PARALLEL_SCHEDULER](inductor/ENABLE_PARALLEL_SCHEDULER.md)|Use this environment variable to control whether the parallel scheduler FX pass is enabled.|
|[SHUT_DOWN_FX_PASS_LIST](inductor/SHUT_DOWN_FX_PASS_LIST.md)|Use this environment variable to specify the list of torch_npu graph optimization FX passes to be disabled.|

### Debugging, Logs, and Accuracy Checks

| Environment Variable Name | Description |
| --- | --- |
|[（beta）INDUCTOR_ASCEND_CHECK_ACCURACY](inductor/INDUCTOR_ASCEND_CHECK_ACCURACY.md)|INDUCTOR_ASCEND_CHECK_ACCURACY is an accuracy verification tool provided by TorchNPU. In the Triton mode and DVM mode of the torch.compile graph compilation mode (Inductor), it automatically detects the numerical accuracy of fused operators.|
|[INDUCTOR_ASCEND_LOG_LEVEL](inductor/INDUCTOR_ASCEND_LOG_LEVEL.md)|Use this environment variable to configure the log level of the Inductor module.|
