# PyTorch环境变量对照表

本表集中展示环境变量与PyTorch对应变量或相关配置的对照说明，按对应关系分组。配置取值、使用约束和支持型号请参见各变量详情；完整变量范围请参见[环境变量列表](env_variable_list.md)。

PyTorch的配置名称、默认值和兼容名称可能随版本变化，使用时需结合配套PyTorch版本确认。Flight Recorder相关条目以PyTorch 2.15的NCCL后端为对照。

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

## 通过PyTorch配置项对照

| TorchNPU环境变量 | PyTorch对应配置 | 对比说明 |
| --- | --- | --- |
| [INDUCTOR_ASCEND_LOG_LEVEL](inductor/INDUCTOR_ASCEND_LOG_LEVEL.md) | `torch._inductor.config.log_level`（相关配置项） | PyTorch通过`torch._inductor.config.log_level`配置日志级别，无同名环境变量。TorchNPU通过此环境变量提供相关配置。 |
| [TORCHINDUCTOR_NPU_BACKEND](inductor/TORCHINDUCTOR_NPU_BACKEND.md) | `torch._inductor.config`（相关配置项） | PyTorch通过配置项控制后端行为，无同名环境变量。TorchNPU通过此变量选择NPU上的Inductor编译模式。 |
