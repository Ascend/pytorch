# 环境变量列表

本手册描述开发者在TorchNPU训练和在线推理过程中可使用的环境变量。基于CANN构建AI应用和业务过程中使用的环境变量请参考《[CANN 环境变量参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/maintenref/envvar/envref_07_0001.html)》。

环境变量按使用场景分类。每项在列表中归入一个主要类别；取值、默认行为、设置时机和支持型号以详情页为准。

与PyTorch的对应关系和差异可集中查阅[PyTorch环境变量对照表](pytorch_comparison.md)。

## 分类导航

| 分类 | 主要用途 |
| --- | --- |
| [设备初始化与管理](device_management/_menu_device_management.md) | 配置设备初始化、可用性检测、流资源和设备同步等待。 |
| [算子执行与兼容性](op_execution/_menu_op_execution.md) | 控制算子下发、同步执行、数值模式、溢出检测、实现切换及CPU回退行为。 |
| [单算子编译与缓存](op_compilation/_menu_op_compilation.md) | 配置单算子模式的编译缓存；torch.compile配置见图编译分类。 |
| [内存管理](memory_management/_menu_memory_management.md) | 配置缓存分配器、内存复用、OOM快照和对称内存。 |
| [分布式通信与诊断](collective_communication/_menu_collective_communication.md) | 配置HCCL进程组、通信同步、超时诊断和状态采集。 |
| [日志与告警](alarm_message_printing/_menu_alarm_message_printing.md) | 配置TorchNPU通用日志、告警和错误输出；通信及图编译专用诊断配置保留在对应分类。 |
| [特征值检测](eigenvalue_detection/_menu_eigenvalue_detection.md) | 配置特征值检测功能及其检测阈值。 |
| [性能优化与采集](performance_tuning/_menu_performance_tuning.md) | 配置CPU绑核、Eager算子融合和Profiler动态采集。 |
| [图编译（torch.compile / Inductor）](inductor/_menu_inductor.md) | 按编译流程查找后端选择、缓存、融合调优以及调试诊断配置。 |

## 设备初始化与管理

配置设备初始化、可用性检测、流资源和设备同步等待。

| 环境变量名称 | 简介 |
| --- | --- |
|[TORCH_ACL_INIT_CONFIG_PATH](op_execution/TORCH_ACL_INIT_CONFIG_PATH.md)|通过此环境变量可指定aclInit初始化配置文件的路径，用于在NPU初始化阶段传入自定义ACL JSON配置。|
|[PYTORCH_HAL_BASED_NPU_CHECK](device_management/PYTORCH_HAL_BASED_NPU_CHECK.md)|选择NPU可用性检测路径，配置为1时优先通过Ascend HAL查询设备数量，失败时回退到Runtime。|
|[STREAMS_PER_DEVICE](device_management/STREAMS_PER_DEVICE.md)|通过此环境变量可配置stream pool的最大流数。|
|[ACL_DEVICE_SYNC_TIMEOUT](synchronization_timeout/ACL_DEVICE_SYNC_TIMEOUT.md)|通过此环境变量可配置设备同步的超时时间。|
|[TORCH_NPU_DEVICE_CAPABILITY](device_management/TORCH_NPU_DEVICE_CAPABILITY.md)|通过此环境变量可配置`torch_npu.npu.get_device_capability()`的返回值。|
|[TORCH_TRANSFER_TO_NPU](device_management/TORCH_TRANSFER_TO_NPU.md)|通过此环境变量可配置是否自动启用transfer_to_npu功能，将PyTorch的CUDA相关API自动替换为NPU对应API。|

## 算子执行与兼容性

控制算子下发、同步执行、数值模式、溢出检测、实现切换及CPU回退行为。

| 环境变量名称 | 简介 |
| --- | --- |
|[INF_NAN_MODE_ENABLE](op_execution/INF_NAN_MODE_ENABLE.md)|通过此环境变量可控制AI处理器对输入数据为Inf/NaN的处理方式，即控制AI处理器使用饱和模式还是INF_NAN模式。|
|[INF_NAN_MODE_FORCE_DISABLE](op_execution/INF_NAN_MODE_FORCE_DISABLE.md)|<term>Atlas A2训练系列产品</term>/<term>Atlas A3训练系列产品</term>，通过此环境变量可强制关闭INF_NAN模式。|
|[FORCE_OVERFLOW_CHECK](op_execution/FORCE_OVERFLOW_CHECK.md)|通过此环境变量可在非饱和模式（INF_NAN模式）下开启溢出检测开关，用于Inf/NaN问题的异步定位。|
|[COMBINED_ENABLE](op_execution/COMBINED_ENABLE.md)|通过此环境变量可控制是否启用组合连续化优化，用于优化由多个view操作产生的非连续张量的连续化转换。|
|[ASCEND_LAUNCH_BLOCKING](op_execution/ASCEND_LAUNCH_BLOCKING.md)|通过此环境变量可控制算子执行时是否启用同步模式。|
|[TASK_QUEUE_ENABLE](op_execution/TASK_QUEUE_ENABLE.md)|通过此环境变量可配置task_queue算子下发队列是否开启和优化等级。|
|[PER_STREAM_QUEUE](op_execution/PER_STREAM_QUEUE.md)|通过此环境变量可配置是否开启一个stream一个task_queue算子下发队列。|
|[TORCH_NPU_FALLBACK_CPU_DISABLE](op_execution/TORCH_NPU_FALLBACK_CPU_DISABLE.md)|通过此环境变量可控制TorchNPU已纳管的隐式CPU fallback路径是否允许执行。|
|[TORCH_NPU_USE_COMPATIBLE_IMPL](op_execution/TORCH_NPU_USE_COMPATIBLE_IMPL.md)|该环境变量用于控制API的实现是否与PyTorch原生社区完全对齐。|
|[TORCH_NPU_LEGACY_IMPL_LIST](op_execution/TORCH_NPU_LEGACY_IMPL_LIST.md)|通过此环境变量可指定需要使用旧版本实现的配置项。|

## 单算子编译与缓存

配置单算子模式的编译缓存；torch.compile配置见图编译分类。

| 环境变量名称 | 简介 |
| --- | --- |
|[ACL_OP_COMPILER_CACHE_DIR](op_compilation/ACL_OP_COMPILER_CACHE_DIR.md)|通过此环境变量可配置算子编译磁盘缓存的目录。|
|[ACL_OP_COMPILER_CACHE_MODE](op_compilation/ACL_OP_COMPILER_CACHE_MODE.md)|通过此环境变量可配置算子编译磁盘缓存模式。|

## 内存管理

配置缓存分配器、内存复用、OOM快照和对称内存。

| 环境变量名称 | 简介 |
| --- | --- |
|[PYTORCH_ALLOC_CONF](memory_management/PYTORCH_ALLOC_CONF.md)|配置缓存分配器的统一入口，与PYTORCH_NPU_ALLOC_CONF功能相同，二者不能同时设置。|
|[PYTORCH_NPU_ALLOC_CONF](memory_management/PYTORCH_NPU_ALLOC_CONF.md)|通过此环境变量可控制缓存分配器行为。配置此环境变量会改变内存占用量，可能造成性能波动。|
|[PYTORCH_NO_NPU_MEMORY_CACHING](memory_management/PYTORCH_NO_NPU_MEMORY_CACHING.md)|通过此环境变量可配置是否关闭内存复用机制。|
|[OOM_SNAPSHOT_ENABLE](memory_management/OOM_SNAPSHOT_ENABLE.md)|通过此环境变量可配置在内存不足报错时是否保存内存数据，以供分析内存不足原因。|
|[OOM_SNAPSHOT_PATH](memory_management/OOM_SNAPSHOT_PATH.md)|通过此环境变量可配置在内存不足报错时内存数据的保存路径。|
|[MULTI_STREAM_MEMORY_REUSE](memory_management/MULTI_STREAM_MEMORY_REUSE.md)|通过此环境变量可配置多流内存复用是否开启。|
|[TORCH_NPUGRAPH_GC](memory_management/TORCH_NPUGRAPH_GC.md)|通过此环境变量可控制图捕获模式（NPUGraph Capture）过程中是否主动触发Python GC（Garbage Collection）。|
|[NPU_SHMEM_SYMMETRIC_SIZE](memory_management/NPU_SHMEM_SYMMETRIC_SIZE.md)|通过此环境变量可配置NPU对称内存的堆大小，用于设备间直接内存访问。|

## 分布式通信与诊断

配置HCCL进程组、通信同步、超时诊断和状态采集。

### 通信执行与同步

| 环境变量名称 | 简介 |
| --- | --- |
|[TORCH_HCCL_BLOCKING_WAIT](collective_communication/TORCH_HCCL_BLOCKING_WAIT.md)|当使用HCCL作为通信后端时，通过此环境变量可控制`ProcessGroupHCCL`中`wait()`和`synchronize()`的同步模式（阻塞或非阻塞）。|
|[TORCH_HCCL_ASYNC_ERROR_HANDLING](collective_communication/TORCH_HCCL_ASYNC_ERROR_HANDLING.md)|当使用HCCL作为通信后端时，通过此环境变量可控制是否开启异步错误处理。|
|[HCCL_EVENT_TIMEOUT](collective_communication/HCCL_EVENT_TIMEOUT.md)|当使用HCCL作为通信后端时，通过此环境变量可设置等待event完成的超时时间。|
|[P2P_HCCL_BUFFSIZE](collective_communication/P2P_HCCL_BUFFSIZE.md)|通过此环境变量可配置是否开启点对点通信（`torch.distributed.isend`、`torch.distributed.irecv`和`torch.distributed.batch_isend_irecv`），并使用独立通信域功能。|
|[RANK_TABLE_FILE](collective_communication/RANK_TABLE_FILE.md)|通过此环境变量可配置RANK_TABLE_FILE文件的路径，用于集合通信域建链。|
|[ROOTINFO_SCALABLE_ENABLE](collective_communication/ROOTINFO_SCALABLE_ENABLE.md)|通过此环境变量可控制是否开启Scalable RootInfo分级建链。|
|[TORCH_HCCL_RANKS_PER_ROOT](collective_communication/TORCH_HCCL_RANKS_PER_ROOT.md)|开启Scalable RootInfo后，通过此环境变量可配置每个root期望管理的rank数量。|
|[(beta) TORCH_HCCL_ZERO_COPY](collective_communication/（beta）TORCH_HCCL_ZERO_COPY.md)|训练或在线推理场景下，可通过此环境变量开启集合通信片内零拷贝功能，减少通信算子在通信过程中片内拷贝次数，提升集合通信效率，降低通信耗时。同时在计算通信并行场景下，降低通信过程中对显存带宽的抢占。|

### 超时监控与故障记录

| 环境变量名称 | 简介 |
| --- | --- |
|[TORCH_HCCL_DESYNC_DEBUG](collective_communication/TORCH_HCCL_DESYNC_DEBUG.md)|当使用HCCL作为通信后端时，通过此环境变量可控制是否进行通信超时分析。|

### 状态采集

| 环境变量名称 | 简介 |
| --- | --- |
|[TORCH_HCCL_STATUS_SAVE_ENABLE](collective_communication/TORCH_HCCL_STATUS_SAVE_ENABLE.md)|通过此环境变量可控制HCCL进程组状态信息的周期性保存。|
|[TORCH_HCCL_STATUS_SAVE_PATH](collective_communication/TORCH_HCCL_STATUS_SAVE_PATH.md)|通过此环境变量可配置HCCL状态文件的保存目录。|
|[TORCH_HCCL_STATUS_SAVE_INTERVAL](collective_communication/TORCH_HCCL_STATUS_SAVE_INTERVAL.md)|通过此环境变量可配置HCCL状态保存的间隔时间。|

## 日志与告警

配置TorchNPU通用日志、告警和错误输出；通信及图编译专用诊断配置保留在对应分类。

| 环境变量名称 | 简介 |
| --- | --- |
|[TORCH_NPU_WARNING_DISABLE](alarm_message_printing/TORCH_NPU_WARNING_DISABLE.md)|通过此环境变量可配置是否打印TorchNPU的告警信息。|
|[TORCH_NPU_DISABLED_WARNING](alarm_message_printing/TORCH_NPU_DISABLED_WARNING.md)|该环境变量已废弃，建议使用TORCH_NPU_WARNING_DISABLE替代。用于配置是否打印TorchNPU的告警信息。|
|[TORCH_NPU_COMPACT_ERROR_OUTPUT](alarm_message_printing/TORCH_NPU_COMPACT_ERROR_OUTPUT.md)|通过此环境变量可精简打印错误信息，开启后会将CANN内部调用栈、TorchNPU错误码等自定义报错信息转移到plog中，仅保留有效的错误说明，提高异常信息的可读性。|
|[TORCH_NPU_LOGS](alarm_message_printing/TORCH_NPU_LOGS.md)|此环境变量用于配置TorchNPU新增模块的日志打印功能，为开发者在debugging场景下提供精准的调试定位能力。|
|[TORCH_NPU_LOGS_FILTER](alarm_message_printing/TORCH_NPU_LOGS_FILTER.md)|此环境变量用于过滤TorchNPU日志输出内容，通过黑白名单机制筛选需要显示的日志信息，帮助开发者在大量日志中快速定位关键信息。|

## 特征值检测

配置特征值检测功能及其检测阈值。

| 环境变量名称 | 简介 |
| --- | --- |
|[NPU_ASD_ENABLE](eigenvalue_detection/NPU_ASD_ENABLE.md)|TorchNPU 7.0.0及之前版本，通过此环境变量可控制是否开启特征值检测功能。|
|[NPU_ASD_UPPER_THRESH](eigenvalue_detection/NPU_ASD_UPPER_THRESH.md)|TorchNPU 7.0.0及之前版本，通过此环境变量可配置特征值检测功能的绝对阈值。|
|[NPU_ASD_SIGMA_THRESH](eigenvalue_detection/NPU_ASD_SIGMA_THRESH.md)|TorchNPU 7.0.0及之前版本，通过此环境变量可配置特征值检测功能的相对阈值。|
|[NPU_ASD_CONFIG](eigenvalue_detection/NPU_ASD_CONFIG.md)|TorchNPU 7.1.0及之后版本，通过此环境变量可控制是否开启TorchNPU的特征值检测功能。|

## 性能优化与采集

配置CPU绑核、Eager算子融合和Profiler动态采集。

### 执行优化

| 环境变量名称 | 简介 |
| --- | --- |
|[CPU_AFFINITY_CONF](performance_tuning/CPU_AFFINITY_CONF.md)|TorchNPU可以通过设置环境变量CPU_AFFINITY_CONF来开启粗/细粒度绑核。该配置能够避免线程间抢占，提高缓存命中，避免跨NUMA（非统一内存访问架构）节点的内存访问，减少任务调度开销，优化任务执行效率。|
|[TORCH_NPU_LAZY_FUSION](performance_tuning/TORCH_NPU_LAZY_FUSION.md)|通过此环境变量可开启DVM算子融合，对elementwise、激活函数等算子做跨算子融合，减少kernel launch和HBM搬运，加速训练和推理。|

### Profiler采集

| 环境变量名称 | 简介 |
| --- | --- |
|[PROF_CONFIG_PATH](performance_tuning/PROF_CONFIG_PATH.md)|在TorchNPU训练场景中，通过此环境变量可指定TorchNPU Profiler接口的dynamic_profile采集功能的profiler_config.json配置文件路径。|
|[KINETO_USE_DAEMON](performance_tuning/KINETO_USE_DAEMON.md)|该环境变量用于在训练场景中设置是否通过msMonitor nputrace方式开启dynamic_profile采集功能。|

## 图编译（torch.compile / Inductor）

按编译流程查找后端选择、缓存、融合调优以及调试诊断配置。

### 编译模式与回退

| 环境变量名称 | 简介 |
| --- | --- |
|[TORCHINDUCTOR_NPU_BACKEND](inductor/TORCHINDUCTOR_NPU_BACKEND.md)|通过该环境变量可配置图模式下的优化模式，支持Triton、DVM、Ascend C等优化模式。|

### 缓存与编译并发

| 环境变量名称 | 简介 |
| --- | --- |
|[TORCH_CACHING_PRECOMPILE](inductor/TORCH_CACHING_PRECOMPILE.md)|通过此环境变量可开启自动缓存预编译实验性功能，自动保存和加载Dynamo编译缓存，加速后续编译过程，与PyTorch上游行为一致。|
|[TORCHINDUCTOR_COMPILE_THREADS](inductor/TORCHINDUCTOR_COMPILE_THREADS.md)|通过此环境变量可配置并发编译的进程数量，与PyTorch上游行为一致。|
|[TORCHNPU_PRECOMPILE_THREADS](inductor/TORCHNPU_PRECOMPILE_THREADS.md)|通过此环境变量可配置torch_npu Inductor的预编译线程数。|

### 算子融合与自动调优

| 环境变量名称 | 简介 |
| --- | --- |
|[TORCHINDUCTOR_MAX_AUTOTUNE](inductor/TORCHINDUCTOR_MAX_AUTOTUNE.md)|通过此环境变量可控制是否开启max autotune功能，与PyTorch上游行为一致。|
|[TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS](inductor/TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS.md)|通过此环境变量可配置max autotune过程中矩阵乘算子参与调优的候选实现列表。|
|[TORCHINDUCTOR_NPU_CATLASS_DIR](inductor/TORCHINDUCTOR_NPU_CATLASS_DIR.md)|通过此环境变量可配置Catlass模板库的路径，与PyTorch上游的`TORCHINDUCTOR_CUTLASS_DIR`对应。|
|[TORCHINDUCTOR_CATLASS_ENABLED_OPS](inductor/TORCHINDUCTOR_CATLASS_ENABLED_OPS.md)|通过此环境变量可配置Catlass模板库支持的矩阵乘类型算子列表，与PyTorch上游的`TORCHINDUCTOR_CUTLASS_ENABLED_OPS`对应。|
|[CATLASS_EPILOGUE_FUSION](inductor/CATLASS_EPILOGUE_FUSION.md)|通过此环境变量可控制是否开启Catlass epilogue融合功能，与PyTorch上游的`CUTLASS_EPILOGUE_FUSION`对应。|
|[TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING](inductor/TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING.md)|通过此环境变量可控制autotune过程中是否使用profiling进行性能测量，与PyTorch上游行为一致。|
|[INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE](inductor/INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE.md)|通过此环境变量可控制autotune过程中是否启用batch profiler进行批量性能测量。|
|[INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE](inductor/INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE.md)|通过此环境变量可控制是否启用动态shape分组autotune，按shape特征分组复用调优结果。|
|[INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES](inductor/INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES.md)|通过此环境变量可配置参与动态shape分组autotune的模板类型列表。|
|[INDUCTOR_ASCEND_ENABLE_COSTMODEL](inductor/INDUCTOR_ASCEND_ENABLE_COSTMODEL.md)|通过此环境变量可控制是否启用CostModel预筛选功能，减少后续编译和实测profiling的config数量。|
|[INDUCTOR_ASCEND_COSTMODEL_RATIO](inductor/INDUCTOR_ASCEND_COSTMODEL_RATIO.md)|通过此环境变量可控制CostModel预筛选后保留的config比例。|

### 图优化与调度

| 环境变量名称 | 简介 |
| --- | --- |
|[ENABLE_PARALLEL_SCHEDULER](inductor/ENABLE_PARALLEL_SCHEDULER.md)|通过此环境变量可控制是否启用并行调度器FX pass。|
|[SHUT_DOWN_FX_PASS_LIST](inductor/SHUT_DOWN_FX_PASS_LIST.md)|通过此环境变量可指定需要关闭的torch_npu图优化FX pass列表。|

### 调试、日志与精度检查

| 环境变量名称 | 简介 |
| --- | --- |
|[（beta）INDUCTOR_ASCEND_CHECK_ACCURACY](inductor/INDUCTOR_ASCEND_CHECK_ACCURACY.md)|INDUCTOR_ASCEND_CHECK_ACCURACY是TorchNPU提供的精度校验工具，在torch.compile图编译模式（Inductor）的Triton模式与DVM模式下自动检测融合算子的数值精度。|
|[INDUCTOR_ASCEND_LOG_LEVEL](inductor/INDUCTOR_ASCEND_LOG_LEVEL.md)|通过此环境变量可配置Inductor模块的日志级别。|
