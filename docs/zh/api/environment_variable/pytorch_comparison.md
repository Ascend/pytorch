# PyTorch环境变量对照表

本表集中展示环境变量与PyTorch对应变量或相关配置的对照说明，按对应关系分组。配置取值、使用约束和支持型号请参见各变量详情；完整变量范围请参见[环境变量列表](env_variable_list.md)。

PyTorch的配置名称、默认值和兼容名称可能随版本变化，使用时需结合配套PyTorch版本确认。WatchDog、心跳监控和Flight Recorder相关条目以PyTorch 2.15的NCCL后端为对照。

## 同名环境变量

| TorchNPU环境变量 | PyTorch对应配置 | 对比说明 |
| --- | --- | --- |
| [TORCH_CACHING_PRECOMPILE](inductor/TORCH_CACHING_PRECOMPILE.md) | `TORCH_CACHING_PRECOMPILE`（环境变量） | 变量名称和行为一致，用于开启Dynamo自动缓存预编译功能。 |
| [TORCHINDUCTOR_COMPILE_THREADS](inductor/TORCHINDUCTOR_COMPILE_THREADS.md) | `TORCHINDUCTOR_COMPILE_THREADS`（环境变量） | 变量名称和行为一致，默认值为`min(32, CPU核心数)`。 |
| [TORCHINDUCTOR_MAX_AUTOTUNE](inductor/TORCHINDUCTOR_MAX_AUTOTUNE.md) | `TORCHINDUCTOR_MAX_AUTOTUNE`（环境变量） | 变量名称和行为一致。 |
| [TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS](inductor/TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS.md) | `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS`（环境变量） | 变量名称和行为一致。 |
| [TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING](inductor/TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING.md) | `TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING`（环境变量） | 变量名称和行为一致。 |
| [PYTORCH_ALLOC_CONF](memory_management/PYTORCH_ALLOC_CONF.md) | `PYTORCH_ALLOC_CONF`（环境变量） | 沿用PyTorch的同名配置入口。在TorchNPU中与[PYTORCH_NPU_ALLOC_CONF](memory_management/PYTORCH_NPU_ALLOC_CONF.md)功能相同，二者不能同时配置。 |
| [TORCH_DEVICE_BACKEND_AUTOLOAD](device_management/TORCH_DEVICE_BACKEND_AUTOLOAD.md) | `TORCH_DEVICE_BACKEND_AUTOLOAD`（环境变量） | 变量名称、默认值和取值语义一致，用于控制`import torch`时是否自动加载树外设备后端（如torch_npu）。 |
| [TORCH_FR_BUFFER_SIZE](collective_communication/TORCH_FR_BUFFER_SIZE.md) | `TORCH_FR_BUFFER_SIZE`（环境变量） | 变量名称一致，默认值不同。PyTorch默认值为`2000`，TorchNPU默认值为`0`。两者均在设为正整数时开启通信记录，设为`0`时关闭。 |
| [TORCH_FR_CPP_STACK](collective_communication/TORCH_FR_CPP_STACK.md) | `TORCH_FR_CPP_STACK`（环境变量） | 变量名称一致，均用于控制是否采集通信操作的C++调用栈，默认值为`0`。两者兼容各自后端的旧变量。 |
| [TORCH_FR_DUMP_TEMP_FILE](collective_communication/TORCH_FR_DUMP_TEMP_FILE.md) | `TORCH_FR_DUMP_TEMP_FILE`（环境变量） | 变量名称一致，默认转储前缀和路径规则一致。静态模式在前缀后追加rank，动态模式使用精确路径。 |
| [TORCH_FR_DUMP_DYNAMIC_FILE_NAME](collective_communication/TORCH_FR_DUMP_DYNAMIC_FILE_NAME.md) | `TORCH_FR_DUMP_DYNAMIC_FILE_NAME`（环境变量） | 变量名称一致，行为和默认值一致。默认值为`0`。设为`1`时，每次转储按`TORCH_FR_DUMP_TEMP_FILE`指定的精确路径写入，路径为空时不回退。 |
| [TORCH_INCLUDE_STACK_TRACE](collective_communication/TORCH_INCLUDE_STACK_TRACE.md) | `TORCH_INCLUDE_STACK_TRACE`（环境变量） | 变量名称一致，默认值均为`1`，均可控制常规监控转储是否输出条目调用栈。TorchNPU的管道转储也受该变量控制，PyTorch的NCCL管道转储固定请求输出调用栈。 |
| [TORCH_INCLUDE_ONLY_ACTIVE](collective_communication/TORCH_INCLUDE_ONLY_ACTIVE.md) | `TORCH_INCLUDE_ONLY_ACTIVE`（环境变量） | 变量名称一致，行为和默认值一致。默认值为`0`。设为`1`时，转储只输出尚未观测到完成的条目。 |

## 异名或相关环境变量

| TorchNPU环境变量 | PyTorch对应配置 | 对比说明 |
| --- | --- | --- |
| [PYTORCH_HAL_BASED_NPU_CHECK](device_management/PYTORCH_HAL_BASED_NPU_CHECK.md) | [PYTORCH_NVML_BASED_CUDA_CHECK](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)（环境变量） | 命名与取值语义设计参考了PyTorch的对应变量，在功能模式和使用逻辑上保持一致。 |
| [CATLASS_EPILOGUE_FUSION](inductor/CATLASS_EPILOGUE_FUSION.md) | `CUTLASS_EPILOGUE_FUSION`（环境变量） | 变量名称不同，行为一致，默认值为`0`。 |
| [TORCHINDUCTOR_CATLASS_ENABLED_OPS](inductor/TORCHINDUCTOR_CATLASS_ENABLED_OPS.md) | `TORCHINDUCTOR_CUTLASS_ENABLED_OPS`（环境变量） | 变量名称和默认值不同。PyTorch默认值为`all`；TorchNPU默认值为`"mm,addmm,bmm"`。 |
| [TORCHINDUCTOR_NPU_CATLASS_DIR](inductor/TORCHINDUCTOR_NPU_CATLASS_DIR.md) | `TORCHINDUCTOR_CUTLASS_DIR`（环境变量） | 变量名称和模板库不同。PyTorch配置CUTLASS（CUDA Templates for Linear Algebra Subroutines）库路径，TorchNPU配置Catlass库路径。 |
| [TORCHNPU_PRECOMPILE_THREADS](inductor/TORCHNPU_PRECOMPILE_THREADS.md) | `TORCHINDUCTOR_COMPILE_THREADS`（相关环境变量） | 控制对象不同。PyTorch变量控制编译并发度；TorchNPU在此基础上扩展预编译线程数控制。 |
| [TORCH_HCCL_ENABLE_TIMING](collective_communication/TORCH_HCCL_ENABLE_TIMING.md) | `TORCH_NCCL_ENABLE_TIMING`（环境变量） | 变量名称不同，均用于启用通信设备计时，默认值为`0`。PyTorch使用CUDA事件，TorchNPU使用NPU事件。开启各自后端的通信失步检查也会启用计时；PyTorch设为`TORCH_DISTRIBUTED_DEBUG=DETAIL`时也会启用。 |
| [TORCH_HCCL_EXTRA_DUMP_ON_EXEC](collective_communication/TORCH_HCCL_EXTRA_DUMP_ON_EXEC.md) | `TORCH_NCCL_EXTRA_DUMP_ON_EXEC`（环境变量） | 变量名称和触发范围不同，默认值均为`0`。两者均可在WatchDog捕获标准异常时额外采集诊断快照，PyTorch的NCCL进程组在执行`abort()`时也会尝试采集。 |
| [TORCH_HCCL_DEBUG_INFO_PIPE_FILE](collective_communication/TORCH_HCCL_DEBUG_INFO_PIPE_FILE.md) | `TORCH_NCCL_DEBUG_INFO_PIPE_FILE`（环境变量） | 变量名称不同，均用于配置手动触发转储的命名管道前缀，默认值为空。管道由心跳监控线程创建。PyTorch在NCCL WatchDog运行时启动该线程，TorchNPU还需显式开启HCCL心跳监控。 |
| [TORCH_HCCL_ASYNC_ERROR_HANDLING](collective_communication/TORCH_HCCL_ASYNC_ERROR_HANDLING.md) | `TORCH_NCCL_ASYNC_ERROR_HANDLING`（环境变量） | 变量名称不同，默认值均为`3`，均支持`0`、`1`、`2`、`3`四种异常处理策略。默认跳过通信域清理并要求退出；各模式的具体行为和HCCL使用约束请参见变量详情。 |
| [TORCH_HCCL_BLOCKING_WAIT](collective_communication/TORCH_HCCL_BLOCKING_WAIT.md) | `TORCH_NCCL_BLOCKING_WAIT`（环境变量） | 变量名称不同，默认值均为`0`。设为`1`时，由调用Work等待接口的线程等待通信完成并检查超时，不启动WatchDog及其心跳监控线程。 |
| [TORCH_HCCL_ENABLE_MONITORING](collective_communication/TORCH_HCCL_ENABLE_MONITORING.md) | `TORCH_NCCL_ENABLE_MONITORING`（环境变量） | TorchNPU默认为`0`，控制是否启动心跳监控线程。PyTorch默认为`1`，控制监控线程在处理通信超时、通信异常或心跳停滞等故障后是否终止进程。设为`0`时，NCCL心跳监控线程仍会启动。 |
| [TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC](collective_communication/TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC.md) | `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`（环境变量） | 均以秒为单位，用于判断WatchDog心跳是否长期未更新，并用于故障诊断后的额外等待。TorchNPU默认`600`秒，PyTorch默认`480`秒；TorchNPU需先开启心跳监控。 |
| [TORCH_HCCL_COORD_CHECK_MILSEC](collective_communication/TORCH_HCCL_COORD_CHECK_MILSEC.md) | `TORCH_NCCL_COORD_CHECK_MILSEC`（环境变量） | 变量名称不同，默认值均为`1000`毫秒。用于全局进程组心跳监控线程轮询跨rank转储请求和手动转储管道；实际检查还需满足相应的转储配置条件。 |
| [TORCH_HCCL_DUMP_ON_TIMEOUT](collective_communication/TORCH_HCCL_DUMP_ON_TIMEOUT.md) | `TORCH_NCCL_DUMP_ON_TIMEOUT`（环境变量） | 均控制Work异常、通信超时等故障的常规自动FR转储，默认值不同：TorchNPU为`0`，PyTorch为`1`。TorchNPU还需开启心跳监控，并配置有效的FR缓冲区。该变量不代表所有转储入口的总开关。 |
| [TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC](collective_communication/TORCH_HCCL_WAIT_TIMEOUT_DUMP_MILSEC.md) | `TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC`（环境变量） | 变量名称不同，默认值均为`15000`毫秒，即`15`秒。用于故障转储等待，包括WatchDog预留的诊断等待时间和监控线程等待异步转储结果的时间。 |
| [TORCH_HCCL_RETHROW_HCCL_ERRORS](collective_communication/TORCH_HCCL_RETHROW_HCCL_ERRORS.md) | `TORCH_NCCL_RETHROW_CUDA_ERRORS`（环境变量） | 变量名称不同，默认值均为`1`。控制WatchDog捕获标准异常后是否继续重抛。设为`0`会结束WatchDog线程，不会恢复监控；超时强制重抛和未知异常等路径仍可能导致进程退出。|
| [TORCH_HCCL_TEARDOWN_ON_TIMEOUT](collective_communication/TORCH_HCCL_TEARDOWN_ON_TIMEOUT.md) | `TORCH_NCCL_TEARDOWN_ON_TIMEOUT`（环境变量） | 变量名称不同，默认值均为`0`。设为`1`时，异步错误处理模式`1`或`3`下的Work超时可在关闭一般异常重抛后仍触发重抛退出。 |
| [TORCH_HCCL_PROPAGATE_ERROR](collective_communication/TORCH_HCCL_PROPAGATE_ERROR.md) | `TORCH_NCCL_PROPAGATE_ERROR`（环境变量） | 变量名称不同，默认值均为`0`。开启后，通过进程组Store传播错误来源，接收方可将进程组错误状态设置为`REMOTE_ERROR`。 |
| [TORCH_HCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN](collective_communication/TORCH_HCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN.md) | `TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN`（环境变量） | 变量名称不同，默认值均为`1`。心跳监控线程进入故障诊断时，在C++栈采集器已注册的条件下尝试采集当前线程栈。 |
| [TORCH_HCCL_DESYNC_DEBUG](collective_communication/TORCH_HCCL_DESYNC_DEBUG.md) | `TORCH_NCCL_DESYNC_DEBUG`（环境变量） | 变量名称不同，默认值均为`0`。均用于记录各rank通信进度，并在Work超时时尝试生成通信失步报告；`TORCH_DISTRIBUTED_DEBUG=DETAIL`也会启用相关诊断。非阻塞等待配置下，启用诊断且异步错误处理模式为`0`时，会将该模式调整为`3`。 |

## 通过PyTorch配置项对照

| TorchNPU环境变量 | PyTorch对应配置 | 对比说明 |
| --- | --- | --- |
| [INDUCTOR_ASCEND_LOG_LEVEL](inductor/INDUCTOR_ASCEND_LOG_LEVEL.md) | `torch._inductor.config.log_level`（相关配置项） | PyTorch通过`torch._inductor.config.log_level`配置日志级别，无同名环境变量。TorchNPU通过此环境变量提供相关配置。 |
| [TORCHINDUCTOR_NPU_BACKEND](inductor/TORCHINDUCTOR_NPU_BACKEND.md) | `torch._inductor.config`（相关配置项） | PyTorch通过配置项控制后端行为，无同名环境变量。TorchNPU通过此变量选择NPU上的Inductor编译模式。 |
