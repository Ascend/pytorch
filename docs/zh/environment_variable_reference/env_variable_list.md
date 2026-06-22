# 环境变量列表

本手册描述开发者在Ascend Extension for PyTorch训练和在线推理过程中可使用的环境变量。基于CANN构建AI应用和业务过程中使用的环境变量请参考《[CANN 环境变量参考](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/envvar/envref_07_0001.html)》。

**表 1**  算子执行环境变量列表

|环境变量名称|简介|
|--|--|
|[INF_NAN_MODE_ENABLE](INF_NAN_MODE_ENABLE.md)|通过此环境变量可控制AI处理器对输入数据为Inf/NaN的处理方式，即控制AI处理器使用饱和模式还是INF_NAN模式。|
|[INF_NAN_MODE_FORCE_DISABLE](INF_NAN_MODE_FORCE_DISABLE.md)|<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>，通过此环境变量可强制关闭INF_NAN模式。|
|[COMBINED_ENABLE](COMBINED_ENABLE.md)|通过此环境变量可设置combined标志。|
|[ASCEND_LAUNCH_BLOCKING](ASCEND_LAUNCH_BLOCKING.md)|通过此环境变量可控制算子执行时是否启用同步模式。|
|[TASK_QUEUE_ENABLE](TASK_QUEUE_ENABLE.md)|通过此环境变量可配置task_queue算子下发队列是否开启和优化等级。|
|[PER_STREAM_QUEUE](PER_STREAM_QUEUE.md)|通过此环境变量可配置是否开启一个stream一个task_queue算子下发队列。|
|[TORCH_NPU_USE_COMPATIBLE_IMPL](TORCH_NPU_USE_COMPATIBLE_IMPL.md)|该环境变量用于控制API的实现是否与PyTorch原生社区完全对齐。|

**表 2**  算子编译环境变量列表

|环境变量名称|简介|
|--|--|
|[ACL_OP_COMPILER_CACHE_DIR](ACL_OP_COMPILER_CACHE_DIR.md)|通过此环境变量可配置算子编译磁盘缓存的目录。|
|[ACL_OP_COMPILER_CACHE_MODE](ACL_OP_COMPILER_CACHE_MODE.md)|通过此环境变量可配置算子编译磁盘缓存模式。|

**表 3**  内存管理环境变量列表

|环境变量名称|简介|
|--|--|
|[PYTORCH_NPU_ALLOC_CONF](PYTORCH_NPU_ALLOC_CONF.md)|通过此环境变量可控制缓存分配器行为。配置此环境变量会改变内存占用量，可能造成性能波动。|
|[PYTORCH_NO_NPU_MEMORY_CACHING](PYTORCH_NO_NPU_MEMORY_CACHING.md)|通过此环境变量可配置是否关闭内存复用机制。|
|[OOM_SNAPSHOT_ENABLE](OOM_SNAPSHOT_ENABLE.md)|通过此环境变量可配置在内存不足报错时是否保存内存数据，以供分析内存不足原因。|
|[OOM_SNAPSHOT_PATH](OOM_SNAPSHOT_PATH.md)|通过此环境变量可配置在内存不足报错时内存数据的保存路径。|
|[MULTI_STREAM_MEMORY_REUSE](MULTI_STREAM_MEMORY_REUSE.md)|通过此环境变量可配置多流内存复用是否开启。|
|[TORCH_NPUGRAPH_GC](TORCH_NPUGRAPH_GC.md)|通过此环境变量可控制图捕获模式（NPUGraph Capture）过程中是否主动触发Python GC（Garbage Collection）。|

**表 4**  集合通信环境变量列表

|环境变量名称|简介|
|--|--|
|[TORCH_HCCL_BLOCKING_WAIT](TORCH_HCCL_BLOCKING_WAIT.md)|当使用HCCL作为通信后端时，通过此环境变量可控制`ProcessGroupHCCL`中`wait()`和`synchronize()`的同步模式（阻塞或非阻塞）。|
|[HCCL_ASYNC_ERROR_HANDLING](HCCL_ASYNC_ERROR_HANDLING.md)|当使用HCCL作为通信后端时，通过此环境变量可控制是否开启异步错误处理。|
|[HCCL_DESYNC_DEBUG](HCCL_DESYNC_DEBUG.md)|当使用HCCL作为通信后端时，通过此环境变量可控制是否进行通信超时分析。|
|[HCCL_EVENT_TIMEOUT](HCCL_EVENT_TIMEOUT.md)|当使用HCCL作为通信后端时，通过此环境变量可设置等待event完成的超时时间。|
|[P2P_HCCL_BUFFSIZE](P2P_HCCL_BUFFSIZE.md)|通过此环境变量可配置是否开启点对点通信（`torch.distributed.isend`、`torch.distributed.irecv`和`torch.distributed.batch_isend_irecv`），并使用独立通信域功能。|
|[RANK_TABLE_FILE](RANK_TABLE_FILE.md)|通过此环境变量可配置RANK_TABLE_FILE文件的路径，用于集合通信域建链。|
|[(beta) TORCH_HCCL_ZERO_COPY](（beta）TORCH_HCCL_ZERO_COPY.md)|训练或在线推理场景下，可通过此环境变量开启集合通信片内零拷贝功能，减少通信算子在通信过程中片内拷贝次数，提升集合通信效率，降低通信耗时。同时在计算通信并行场景下，降低通信过程中对显存带宽的抢占。|

**表 5**  告警信息打印环境变量列表

|环境变量名称|简介|
|--|--|
|[TORCH_NPU_DISABLED_WARNING](TORCH_NPU_DISABLED_WARNING.md)|通过此环境变量可配置是否打印Ascend Extension for PyTorch的告警信息。|
|[TORCH_NPU_COMPACT_ERROR_OUTPUT](TORCH_NPU_COMPACT_ERROR_OUTPUT.md)|通过此环境变量可精简打印错误信息，开启后会将CANN内部调用栈、Ascend Extension for PyTorch错误码等自定义报错信息转移到plog中，仅保留有效的错误说明，提高异常信息的可读性。|
|[TORCH_NPU_LOGS](TORCH_NPU_LOGS.md)|此环境变量用于配置Ascend Extension for PyTorch新增模块的日志打印功能，为开发者在debugging场景下提供精准的调试定位能力。|
|[TORCH_NPU_LOGS_FILTER](TORCH_NPU_LOGS_FILTER.md)|此环境变量用于过滤Ascend Extension for PyTorch日志输出内容，通过黑白名单机制筛选需要显示的日志信息，帮助开发者在大量日志中快速定位关键信息。|

**表 6**  同步超时变量列表

|环境变量名称|简介|
|--|--|
|[ACL_DEVICE_SYNC_TIMEOUT](ACL_DEVICE_SYNC_TIMEOUT.md)|通过此环境变量可配置设备同步的超时时间。|

**表 7**  特征值检测环境变量列表

|环境变量名称|简介|
|--|--|
|[NPU_ASD_ENABLE](NPU_ASD_ENABLE.md)|Ascend Extension for PyTorch 7.0.0及之前版本，通过此环境变量可控制是否开启特征值检测功能。|
|[NPU_ASD_UPPER_THRESH](NPU_ASD_UPPER_THRESH.md)|Ascend Extension for PyTorch 7.0.0及之前版本，通过此环境变量可配置特征值检测功能的绝对阈值。|
|[NPU_ASD_SIGMA_THRESH](NPU_ASD_SIGMA_THRESH.md)|Ascend Extension for PyTorch 7.0.0及之前版本，通过此环境变量可配置特征值检测功能的相对阈值。|
|[NPU_ASD_CONFIG](NPU_ASD_CONFIG.md)|Ascend Extension for PyTorch 7.1.0及之后版本，通过此环境变量可控制是否开启Ascend Extension for PyTorch的特征值检测功能。|

**表 8**  性能优化环境变量列表

|环境变量名称|简介|
|--|--|
|[CPU_AFFINITY_CONF](CPU_AFFINITY_CONF.md)|Ascend Extension for PyTorch可以通过设置环境变量CPU_AFFINITY_CONF来开启粗/细粒度绑核。该配置能够避免线程间抢占，提高缓存命中，避免跨NUMA（非统一内存访问架构）节点的内存访问，减少任务调度开销，优化任务执行效率。|
|[PROF_CONFIG_PATH](PROF_CONFIG_PATH.md)|在PyTorch训练场景中，通过此环境变量可指定Ascend PyTorch Profiler接口的dynamic_profile采集功能的profiler_config.json配置文件路径。|
|[KINETO_USE_DAEMON](KINETO_USE_DAEMON.md)|该环境变量用于在训练场景中设置是否通过msMonitor nputrace方式开启dynamic_profile采集功能。|
|[TORCH_NPU_LAZY_FUSION](TORCH_NPU_LAZY_FUSION.md)|通过此环境变量可开启DVM算子融合，对elementwise、激活函数等算子做跨算子融合，减少kernel launch和HBM搬运，加速训练和推理。|

**表 9**  设备管理环境变量列表

|环境变量名称|简介|
|--|--|
|[STREAMS_PER_DEVICE](STREAMS_PER_DEVICE.md)|通过此环境变量可配置stream pool的最大流数。|
|[TORCH_NPU_DEVICE_CAPABILITY](TORCH_NPU_DEVICE_CAPABILITY.md)|通过此环境变量可配置`torch_npu.npu.get_device_capability()`的返回值。|
|[TORCH_TRANSFER_TO_NPU](TORCH_TRANSFER_TO_NPU.md)|通过此环境变量可配置是否自动启用transfer_to_npu功能，将PyTorch的CUDA相关API自动替换为NPU对应API。|

**表 10**  图模式环境变量列表

|环境变量名称|简介|
|--|--|
|[TORCHINDUCTOR_NPU_BACKEND](TORCHINDUCTOR_NPU_BACKEND.md)|通过该环境变量可配置图模式下的优化模式，支持Triton、MLIR、DVM等优化模式。|
|[（beta）INDUCTOR_ASCEND_CHECK_ACCURACY](INDUCTOR_ASCEND_CHECK_ACCURACY.md)|INDUCTOR_ASCEND_CHECK_ACCURACY是Ascend Extension for PyTorch提供的精度校验工具，仅在torch.compile图编译后端为"Inductor"且模式为"Triton"时自动检测融合算子的数值精度。|
|[NPU_INDUCTOR_FALLBACK_LIST](NPU_INDUCTOR_FALLBACK_LIST.md)|通过此环境变量可指定需要回退到PyTorch原生的算子列表。|
|[（beta）TORCHINDUCTOR_ENABLE_MFUSION](TORCHINDUCTOR_ENABLE_MFUSION.md)|通过此环境变量可控制是否启用MFusion融合优化功能，仅在torch.compile图编译后端为"Inductor"生效。|
|[TORCHINDUCTOR_USE_AKG](TORCHINDUCTOR_USE_AKG.md)|通过此环境变量可配置torch.compile图模式（Inductor）下MLIR（Multi-Level Intermediate Representation）模式启用AKG（Auto Kernel Generator）后端优化。|
