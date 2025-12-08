# 环境变量列表

本手册描述开发者在Ascend Extension for PyTorch训练和在线推理过程中可使用的环境变量。基于CANN构建AI应用和业务过程中使用的环境变量请参考《CANN 环境变量参考》。

**表 1**  环境变量列表

|环境变量类型|环境变量名称|简介|
|--|--|--|
|算子执行|[INF_NAN_MODE_ENABLE](INF_NAN_MODE_ENABLE.md)|通过此环境变量可控制AI处理器对输入数据为Inf/NaN的处理能力，即控制AI处理器使用饱和模式还是INF_NAN模式。|
|算子执行|[INF_NAN_MODE_FORCE_DISABLE](INF_NAN_MODE_FORCE_DISABLE.md)|<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>，通过此环境变量可强制关闭INF_NAN模式。|
|算子执行|[COMBINED_ENABLE](COMBINED_ENABLE.md)|通过此环境变量可设置combined标志。|
|算子执行|[ASCEND_LAUNCH_BLOCKING](ASCEND_LAUNCH_BLOCKING.md)|通过此环境变量可控制算子执行时是否启动同步模式。|
|算子执行|[TASK_QUEUE_ENABLE](TASK_QUEUE_ENABLE.md)|通过此环境变量可配置task_queue算子下发队列是否开启和优化等级。|
|算子执行|[PER_STREAM_QUEUE](PER_STREAM_QUEUE.md)|通过此环境变量可配置是否开启一个stream一个task_queue算子下发队列。|
|算子编译|[ACL_OP_COMPILER_CACHE_DIR](ACL_OP_COMPILER_CACHE_DIR.md)|通过此环境变量可配置算子编译磁盘缓存的目录。|
|算子编译|[ACL_OP_COMPILER_CACHE_MODE](ACL_OP_COMPILER_CACHE_MODE.md)|通过此环境变量可配置算子编译磁盘缓存模式。|
|内存管理|[PYTORCH_NPU_ALLOC_CONF](PYTORCH_NPU_ALLOC_CONF.md)|通过此环境变量可控制缓存分配器行为。配置此环境变量会改变内存占用量，可能造成性能波动。|
|内存管理|[PYTORCH_NO_NPU_MEMORY_CACHING](PYTORCH_NO_NPU_MEMORY_CACHING.md)|通过此环境变量可配置是否关闭内存复用机制。|
|内存管理|[OOM_SNAPSHOT_ENABLE](OOM_SNAPSHOT_ENABLE.md)|通过此环境变量可配置在内存不足报错时是否保存内存数据，以供分析内存不足原因。|
|内存管理|[OOM_SNAPSHOT_PATH](OOM_SNAPSHOT_PATH.md)|通过此环境变量可配置在内存不足报错时内存数据的保存路径。|
|内存管理|[MULTI_STREAM_MEMORY_REUSE](MULTI_STREAM_MEMORY_REUSE.md)|通过此环境变量可配置多流内存复用是否开启。|
|集合通信|[HCCL_ASYNC_ERROR_HANDLING](HCCL_ASYNC_ERROR_HANDLING.md)|当使用HCCL作为通信后端时，通过此环境变量可控制是否开启异步错误处理。|
|集合通信|[HCCL_DESYNC_DEBUG](HCCL_DESYNC_DEBUG.md)|当使用HCCL作为通信后端时，通过此环境变量可控制是否进行通信超时分析。|
|集合通信|[HCCL_EVENT_TIMEOUT](HCCL_EVENT_TIMEOUT.md)|当使用HCCL作为通信后端时，通过此环境变量可设置等待Event完成的超时时间。|
|集合通信|[P2P_HCCL_BUFFSIZE](P2P_HCCL_BUFFSIZE.md)|通过此环境变量可配置是否开启点对点通信（`torch.distributed.isend`、`torch.distributed.irecv`和`torch.distributed.batch_isend_irecv`）使用独立通信域功能。|
|集合通信|[RANK_TABLE_FILE](RANK_TABLE_FILE.md)|通过此环境变量可配置是否通过RANK_TABLE_FILE进行集合通信域建链。|
|集合通信|[(beta) TORCH_HCCL_ZERO_COPY]((beta)-TORCH_HCCL_ZERO_COPY.md)|训练或在线推理场景下，可通过此环境变量开启集合通信片内零拷贝功能，减少通信算子在通信过程中片内拷贝次数，提升集合通信效率，降低通信耗时。同时在计算通信并行场景下，降低通信过程中对显存带宽的抢占。|
|告警信息打印|[TORCH_NPU_DISABLED_WARNING](TORCH_NPU_DISABLED_WARNING.md)|通过此环境变量可配置是否打印Ascend Extension for PyTorch的告警信息。|
|告警信息打印|[TORCH_NPU_COMPACT_ERROR_OUTPUT](TORCH_NPU_COMPACT_ERROR_OUTPUT.md)|通过此环境变量可精简打印错误信息，开启后会将CANN内部调用栈、Ascend Extension for PyTorch错误码等自定义报错信息转移到plog中，仅保留有效的错误说明，提高异常信息的可读性。|
|同步超时|[ACL_DEVICE_SYNC_TIMEOUT](ACL_DEVICE_SYNC_TIMEOUT.md)|通过此环境变量可配置设备同步的超时时间。|
|特征值检测|[NPU_ASD_ENABLE](NPU_ASD_ENABLE.md)|Ascend Extension for PyTorch 7.0.0及之前版本，通过此环境变量可控制是否开启Ascend Extension for PyTorch的特征值检测功能。|
|特征值检测|[NPU_ASD_UPPER_THRESH](NPU_ASD_UPPER_THRESH.md)|Ascend Extension for PyTorch 7.0.0及之前版本，通过此环境变量可配置特征值检测功能的绝对阈值。|
|特征值检测|[NPU_ASD_SIGMA_THRESH](NPU_ASD_SIGMA_THRESH.md)|Ascend Extension for PyTorch 7.0.0及之前版本，通过此环境变量可配置特征值检测功能的相对阈值。|
|特征值检测|[NPU_ASD_CONFIG](NPU_ASD_CONFIG.md)|Ascend Extension for PyTorch 7.1.0及之后版本，通过此环境变量可控制是否开启Ascend Extension for PyTorch的特征值检测功能。|
|性能优化|[CPU_AFFINITY_CONF](CPU_AFFINITY_CONF.md)|Ascend Extension for PyTorch可以通过设置环境变量CPU_AFFINITY_CONF来开启粗/细粒度绑核。该配置能够避免线程间抢占，提高缓存命中，避免跨NUMA（非统一内存访问架构）节点的内存访问，减少任务调度开销，优化任务执行效率。|
|性能优化|[PROF_CONFIG_PATH](PROF_CONFIG_PATH.md)|PyTorch训练场景Ascend PyTorch Profiler接口的dynamic_profile采集功能profiler_config.json配置文件路径环境变量。|
|性能优化|[KINETO_USE_DAEMON](KINETO_USE_DAEMON.md)|PyTorch训练场景用于设置是否通过msMonitor nputrace方式开启dynamic_profile采集功能。|
|设备管理|[STREAMS_PER_DEVICE](STREAMS_PER_DEVICE.md)|通过此环境变量可配置stream pool的最大流数。|


