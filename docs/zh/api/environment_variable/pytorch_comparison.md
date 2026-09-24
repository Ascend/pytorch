# PyTorch环境变量对照表

本表集中展示环境变量与PyTorch对应变量或相关配置的对照说明，按对应关系分组。配置取值、使用约束和支持型号请参见各变量详情；完整变量范围请参见[环境变量列表](env_variable_list.md)。

PyTorch的配置名称、默认值和兼容名称可能随版本变化，使用时需结合配套PyTorch版本确认。

## 同名环境变量

| TorchNPU环境变量 | PyTorch对应配置 | 对比说明 |
| --- | --- | --- |
| [TORCH_CACHING_PRECOMPILE](inductor/TORCH_CACHING_PRECOMPILE.md) | `TORCH_CACHING_PRECOMPILE`（环境变量） | 变量名称和行为一致，用于开启Dynamo自动缓存预编译功能。 |
| [TORCHINDUCTOR_COMPILE_THREADS](inductor/TORCHINDUCTOR_COMPILE_THREADS.md) | `TORCHINDUCTOR_COMPILE_THREADS`（环境变量） | 变量名称和行为一致，默认值为`min(32, CPU核心数)`。 |
| [TORCHINDUCTOR_MAX_AUTOTUNE](inductor/TORCHINDUCTOR_MAX_AUTOTUNE.md) | `TORCHINDUCTOR_MAX_AUTOTUNE`（环境变量） | 变量名称和行为一致。 |
| [TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS](inductor/TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS.md) | `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS`（环境变量） | 变量名称和行为一致。 |
| [TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING](inductor/TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING.md) | `TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING`（环境变量） | 变量名称和行为一致。 |
| [PYTORCH_ALLOC_CONF](memory_management/PYTORCH_ALLOC_CONF.md) | `PYTORCH_ALLOC_CONF`（环境变量） | 沿用PyTorch的同名配置入口。在TorchNPU中与[PYTORCH_NPU_ALLOC_CONF](memory_management/PYTORCH_NPU_ALLOC_CONF.md)功能相同，二者不能同时配置。 |

## 异名或相关环境变量

| TorchNPU环境变量 | PyTorch对应配置 | 对比说明 |
| --- | --- | --- |
| [PYTORCH_HAL_BASED_NPU_CHECK](device_management/PYTORCH_HAL_BASED_NPU_CHECK.md) | [PYTORCH_NVML_BASED_CUDA_CHECK](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)（环境变量） | 命名与取值语义设计参考了PyTorch的对应变量，在功能模式和使用逻辑上保持一致。 |
| [CATLASS_EPILOGUE_FUSION](inductor/CATLASS_EPILOGUE_FUSION.md) | `CUTLASS_EPILOGUE_FUSION`（环境变量） | 变量名称不同，行为一致，默认值为`0`。 |
| [TORCHINDUCTOR_CATLASS_ENABLED_OPS](inductor/TORCHINDUCTOR_CATLASS_ENABLED_OPS.md) | `TORCHINDUCTOR_CUTLASS_ENABLED_OPS`（环境变量） | 变量名称和默认值不同。PyTorch默认值为`all`；TorchNPU默认值为`"mm,addmm,bmm"`。 |
| [TORCHINDUCTOR_NPU_CATLASS_DIR](inductor/TORCHINDUCTOR_NPU_CATLASS_DIR.md) | `TORCHINDUCTOR_CUTLASS_DIR`（环境变量） | 变量名称和模板库不同。PyTorch配置CUTLASS（CUDA Templates for Linear Algebra Subroutines）库路径，TorchNPU配置Catlass库路径。 |
| [TORCHNPU_PRECOMPILE_THREADS](inductor/TORCHNPU_PRECOMPILE_THREADS.md) | `TORCHINDUCTOR_COMPILE_THREADS`（相关环境变量） | 控制对象不同。PyTorch变量控制编译并发度；TorchNPU在此基础上扩展预编译线程数控制。 |

## 通过PyTorch配置项对照

| TorchNPU环境变量 | PyTorch对应配置 | 对比说明 |
| --- | --- | --- |
| [INDUCTOR_ASCEND_LOG_LEVEL](inductor/INDUCTOR_ASCEND_LOG_LEVEL.md) | `torch._inductor.config.log_level`（相关配置项） | PyTorch通过`torch._inductor.config.log_level`配置日志级别，无同名环境变量。TorchNPU通过此环境变量提供相关配置。 |
| [TORCHINDUCTOR_NPU_BACKEND](inductor/TORCHINDUCTOR_NPU_BACKEND.md) | `torch._inductor.config`（相关配置） | PyTorch通过配置项控制后端行为，无同名环境变量。TorchNPU通过此变量选择NPU上的Inductor编译模式。 |
