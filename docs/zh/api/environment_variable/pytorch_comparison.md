# PyTorch环境变量对照表

本表集中展示环境变量文档中38项具有PyTorch对应变量或相关配置的对照说明，按对应关系分组。配置取值、使用约束和支持型号请参见各变量详情；完整变量范围请参见[环境变量列表](env_variable_list.md)。

PyTorch的配置名称、默认值和兼容名称可能随版本变化，使用时需结合配套PyTorch版本确认。

## 同名环境变量

| TorchNPU环境变量 | PyTorch对应配置 | 对比说明 |
| --- | --- | --- |
| [DDP_SET_LAST_BUCKET_CAP](collective_communication/DDP_SET_LAST_BUCKET_CAP.md) | `DDP_SET_LAST_BUCKET_CAP`（环境变量） | 变量名称和行为一致。 |
| [TP_SOCKET_IFNAME](collective_communication/TP_SOCKET_IFNAME.md) | `TP_SOCKET_IFNAME`（环境变量） | 变量名称和行为一致。 |
| [TORCH_CACHING_PRECOMPILE](inductor/TORCH_CACHING_PRECOMPILE.md) | `TORCH_CACHING_PRECOMPILE`（环境变量） | 变量名称和行为一致，用于开启Dynamo自动缓存预编译功能。 |
| [TORCH_COMPILE_DEBUG](inductor/TORCH_COMPILE_DEBUG.md) | `TORCH_COMPILE_DEBUG`（环境变量） | 变量名称和行为一致，用于导出Inductor编译产物辅助调试。 |
| [TORCH_LOGS](alarm_message_printing/TORCH_LOGS.md) | `TORCH_LOGS`（环境变量） | 变量名称和行为一致，用于控制PyTorch各模块的日志输出。 |
| [TORCHINDUCTOR_CACHE_DIR](inductor/TORCHINDUCTOR_CACHE_DIR.md) | `TORCHINDUCTOR_CACHE_DIR`（环境变量） | 变量名称和行为一致，用于指定Inductor编译缓存目录。 |
| [TORCHINDUCTOR_COMPILE_THREADS](inductor/TORCHINDUCTOR_COMPILE_THREADS.md) | `TORCHINDUCTOR_COMPILE_THREADS`（环境变量） | 变量名称和行为一致，默认值为`32`。 |
| [TORCHINDUCTOR_MAX_AUTOTUNE](inductor/TORCHINDUCTOR_MAX_AUTOTUNE.md) | `TORCHINDUCTOR_MAX_AUTOTUNE`（环境变量） | 变量名称和行为一致。 |
| [TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS](inductor/TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS.md) | `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS`（环境变量） | 变量名称和行为一致。 |
| [TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING](inductor/TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING.md) | `TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING`（环境变量） | 变量名称和行为一致。 |
| [TORCHINDUCTOR_WORKER_LOGPATH](inductor/TORCHINDUCTOR_WORKER_LOGPATH.md) | `TORCHINDUCTOR_WORKER_LOGPATH`（环境变量） | 变量名称和行为一致，用于指定Inductor worker子进程的日志路径。 |
| [TORCHINDUCTOR_WORKER_SUPPRESS_LOGGING](inductor/TORCHINDUCTOR_WORKER_SUPPRESS_LOGGING.md) | `TORCHINDUCTOR_WORKER_SUPPRESS_LOGGING`（环境变量） | 变量名称和行为一致，用于控制Inductor worker子进程的日志抑制。 |
| [PYTORCH_ALLOC_CONF](memory_management/PYTORCH_ALLOC_CONF.md) | `PYTORCH_ALLOC_CONF`（环境变量） | 沿用PyTorch的同名配置入口。在TorchNPU中与[PYTORCH_NPU_ALLOC_CONF](memory_management/PYTORCH_NPU_ALLOC_CONF.md)功能相同，二者不能同时配置。 |

## 异名或相关环境变量

| TorchNPU环境变量 | PyTorch对应配置 | 对比说明 |
| --- | --- | --- |
| [TORCH_HCCL_COORD_CHECK_MILSEC](collective_communication/TORCH_HCCL_COORD_CHECK_MILSEC.md) | `TORCH_NCCL_COORD_CHECK_MILSEC`（环境变量） | 变量名称不同，行为一致，默认值均为1000ms。 |
| [TORCH_HCCL_DEBUG_INFO_PIPE_FILE](collective_communication/TORCH_HCCL_DEBUG_INFO_PIPE_FILE.md) | `TORCH_NCCL_DEBUG_INFO_PIPE_FILE`（环境变量） | 变量名称不同，行为一致，默认值均为空。 |
| [TORCH_HCCL_DEBUG_INFO_TEMP_FILE](collective_communication/TORCH_HCCL_DEBUG_INFO_TEMP_FILE.md) | `TORCH_FR_DUMP_TEMP_FILE`（环境变量，兼容名称为`TORCH_NCCL_DEBUG_INFO_TEMP_FILE`） | 变量名称和默认目录不同。PyTorch默认为`$XDG_CACHE_HOME/torch/comm_lib_trace_rank_`，未设置`XDG_CACHE_HOME`时使用`$HOME/.cache/torch/comm_lib_trace_rank_`；TorchNPU默认为`/tmp/hccl_trace_rank_`。 |
| [TORCH_HCCL_DUMP_ON_TIMEOUT](collective_communication/TORCH_HCCL_DUMP_ON_TIMEOUT.md) | `TORCH_FR_DUMP_ON_TIMEOUT`（环境变量，兼容名称为`TORCH_NCCL_DUMP_ON_TIMEOUT`） | 变量名称和默认值不同。PyTorch进程组默认开启，TorchNPU默认关闭。 |
| [TORCH_HCCL_ENABLE_MONITORING](collective_communication/TORCH_HCCL_ENABLE_MONITORING.md) | `TORCH_NCCL_ENABLE_MONITORING`（环境变量） | 变量名称和默认值不同。PyTorch默认开启，TorchNPU默认关闭。 |
| [TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC](collective_communication/TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC.md) | `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`（环境变量） | 变量名称和默认值不同。PyTorch默认为480秒，TorchNPU默认为600秒。 |
| [TORCH_HCCL_HIGH_PRIORITY](collective_communication/TORCH_HCCL_HIGH_PRIORITY.md) | `TORCH_NCCL_HIGH_PRIORITY`（环境变量） | 变量名称不同，行为一致，默认均关闭。 |
| [TORCH_HCCL_TRACE_BUFFER_SIZE](collective_communication/TORCH_HCCL_TRACE_BUFFER_SIZE.md) | `TORCH_FR_BUFFER_SIZE`（环境变量，兼容名称为`TORCH_NCCL_TRACE_BUFFER_SIZE`） | 变量名称和默认值不同。PyTorch默认为2000，TorchNPU默认为0（关闭记录）。 |
| [TORCH_HCCL_TRACE_CPP_STACK](collective_communication/TORCH_HCCL_TRACE_CPP_STACK.md) | `TORCH_FR_CPP_STACK`（环境变量，兼容名称为`TORCH_NCCL_TRACE_CPP_STACK`） | 变量名称不同，行为一致，默认均关闭。PyTorch使用`TORCH_FR_*`作为主名称。 |
| [TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC](collective_communication/TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC.md) | `TORCH_FR_WAIT_TIMEOUT_DUMP_MILSEC`（环境变量，兼容名称为`TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC`） | 变量名称和默认值不同。PyTorch默认为15000ms（15秒），TorchNPU默认为60000ms（60秒）。 |
| [PYTORCH_HAL_BASED_NPU_CHECK](device_management/PYTORCH_HAL_BASED_NPU_CHECK.md) | [PYTORCH_NVML_BASED_CUDA_CHECK](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)（环境变量） | 命名与取值语义设计参考了PyTorch的对应变量，在功能模式和使用逻辑上保持一致。 |
| [TORCH_NPU_ELASTIC_USE_AGENT_STORE](distributed_startup/TORCH_NPU_ELASTIC_USE_AGENT_STORE.md) | `TORCHELASTIC_USE_AGENT_STORE`（环境变量） | 用途相近，变量名称不同。TorchNPU通过此变量控制是否使用agent已启动的ParallelStore。 |
| [CATLASS_EPILOGUE_FUSION](inductor/CATLASS_EPILOGUE_FUSION.md) | `CUTLASS_EPILOGUE_FUSION`（环境变量） | 变量名称不同，行为一致，默认值为`0`。 |
| [INDUCTOR_ASCEND_FX_GRAPH_CACHE](inductor/INDUCTOR_ASCEND_FX_GRAPH_CACHE.md) | `TORCHINDUCTOR_FX_GRAPH_CACHE`（环境变量） | 变量名称和配置用途不同。PyTorch变量用于控制FX图缓存；TorchNPU通过此变量提供独立的缓存路径配置。 |
| [TORCHINDUCTOR_CATLASS_ENABLED_OPS](inductor/TORCHINDUCTOR_CATLASS_ENABLED_OPS.md) | `TORCHINDUCTOR_CUTLASS_ENABLED_OPS`（环境变量） | 变量名称不同，行为一致，默认值为`"mm,addmm,bmm"`。 |
| [TORCHINDUCTOR_NPU_CATLASS_DIR](inductor/TORCHINDUCTOR_NPU_CATLASS_DIR.md) | `TORCHINDUCTOR_CUTLASS_DIR`（环境变量） | 变量名称和模板库不同。PyTorch配置CUTLASS（CUDA Templates for Linear Algebra Subroutines）库路径，TorchNPU配置Catlass库路径。 |
| [TORCHNPU_PRECOMPILE_THREADS](inductor/TORCHNPU_PRECOMPILE_THREADS.md) | `TORCHINDUCTOR_COMPILE_THREADS`（相关环境变量） | 控制对象不同。PyTorch变量控制编译并发度；TorchNPU在此基础上扩展预编译线程数控制。 |

## 通过PyTorch配置项对照

| TorchNPU环境变量 | PyTorch对应配置 | 对比说明 |
| --- | --- | --- |
| [ENABLE_INPLACE_BUFFERS](inductor/ENABLE_INPLACE_BUFFERS.md) | `torch._inductor.config.inplace_buffers`（相关配置项） | PyTorch通过`torch._inductor.config.inplace_buffers`配置项控制原地缓冲区，无同名环境变量。TorchNPU通过此环境变量提供相关配置。 |
| [FX_SUBGRAPH_DUMP_PATH](inductor/FX_SUBGRAPH_DUMP_PATH.md) | `torch._inductor.config`（相关配置项） | PyTorch通过`torch._inductor.config`的debug输出控制，无同名环境变量。TorchNPU通过此环境变量提供相关配置。 |
| [INDUCTOR_ASCEND_DEBUG](inductor/INDUCTOR_ASCEND_DEBUG.md) | `torch._inductor.config.debug`（相关配置项） | PyTorch通过`torch._inductor.config.debug`配置调试模式，无同名环境变量。TorchNPU通过此环境变量提供相关配置。 |
| [INDUCTOR_ASCEND_DUMP_FX_GRAPH](inductor/INDUCTOR_ASCEND_DUMP_FX_GRAPH.md) | `torch._inductor.config.trace.enabled`、`torch._inductor.config.trace.log_url`（相关配置项） | PyTorch通过`torch._inductor.config.trace.enabled`和`torch._inductor.config.trace.log_url`控制FX图导出，无同名环境变量。TorchNPU通过此环境变量提供相关配置。 |
| [INDUCTOR_ASCEND_LOG_LEVEL](inductor/INDUCTOR_ASCEND_LOG_LEVEL.md) | `torch._inductor.config.log_level`（相关配置项） | PyTorch通过`torch._inductor.config.log_level`配置日志级别，无同名环境变量。TorchNPU通过此环境变量提供相关配置。 |
| [NPU_INDUCTOR_FALLBACK_LIST](inductor/NPU_INDUCTOR_FALLBACK_LIST.md) | `torch._inductor.config.fallback_kernel_list`（配置项） | 配置方式不同。PyTorch通过配置项控制回退算子列表，TorchNPU通过环境变量提供类似功能。 |
| [TORCHINDUCTOR_ENABLE_FAST_GELU](inductor/TORCHINDUCTOR_ENABLE_FAST_GELU.md) | `torch._inductor.config`（相关配置项） | PyTorch通过`torch._inductor.config`的激活函数decomposition控制，无同名环境变量。TorchNPU通过此环境变量提供相关配置。 |
| [TORCHINDUCTOR_NPU_BACKEND](inductor/TORCHINDUCTOR_NPU_BACKEND.md) | `torch._inductor.config`（相关配置） | PyTorch通过配置项控制后端行为，无同名环境变量。TorchNPU通过此变量选择NPU上的Inductor编译模式。 |
