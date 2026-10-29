# PyTorch Environment Variable Comparison Table

This table provides a centralized comparison between environment variables and their PyTorch counterparts or related configurations, grouped by correspondence. For configuration values, usage constraints, and supported models, see the details of each variable; for the complete variable list, see [Environment Variable List](env_variable_list.md).

PyTorch configuration names, default values, and compatibility names may vary with versions. When using them, confirm against the matched PyTorch version.

## Environment Variables with the Same Name

| TorchNPU Environment Variable | PyTorch Corresponding Configuration | Comparison Description |
| --- | --- | --- |
| [TORCH_CACHING_PRECOMPILE](inductor/TORCH_CACHING_PRECOMPILE.md) | `TORCH_CACHING_PRECOMPILE` (environment variable) | The variable name and behavior are consistent, used to enable the Dynamo automatic cache precompilation feature. |
| [TORCHINDUCTOR_COMPILE_THREADS](inductor/TORCHINDUCTOR_COMPILE_THREADS.md) | `TORCHINDUCTOR_COMPILE_THREADS` (environment variable) | The variable name and behavior are consistent. The default value is `min(32, number of CPU cores)`. |
| [TORCHINDUCTOR_MAX_AUTOTUNE](inductor/TORCHINDUCTOR_MAX_AUTOTUNE.md) | `TORCHINDUCTOR_MAX_AUTOTUNE` (environment variable) | The variable name and behavior are consistent. |
| [TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS](inductor/TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS.md) | `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` (environment variable) | The variable name and behavior are consistent. |
| [TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING](inductor/TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING.md) | `TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING` (environment variable) | The variable name and behavior are consistent. |
| [PYTORCH_ALLOC_CONF](memory_management/PYTORCH_ALLOC_CONF.md) | `PYTORCH_ALLOC_CONF` (environment variable) | The configuration entry with the same name as PyTorch is adopted. In TorchNPU, it has the same function as [PYTORCH_NPU_ALLOC_CONF](memory_management/PYTORCH_NPU_ALLOC_CONF.md), and the two cannot be configured at the same time. |
| [TORCH_DEVICE_BACKEND_AUTOLOAD](device_management/TORCH_DEVICE_BACKEND_AUTOLOAD.md) | `TORCH_DEVICE_BACKEND_AUTOLOAD` (environment variable) | The variable name, default value, and value semantics are consistent, used to control whether the out-of-tree device backend (such as torch_npu) is automatically loaded when `import torch` is executed. |

## Environment Variables with Different Names or Related Environment Variables

| TorchNPU Environment Variable | PyTorch Corresponding Configuration | Comparison Description |
| --- | --- | --- |
| [PYTORCH_HAL_BASED_NPU_CHECK](device_management/PYTORCH_HAL_BASED_NPU_CHECK.md) | [PYTORCH_NVML_BASED_CUDA_CHECK](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html) (environment variable) | The naming and value semantics are designed with reference to the corresponding PyTorch variable and remain consistent in functional mode and usage logic. |
| [CATLASS_EPILOGUE_FUSION](inductor/CATLASS_EPILOGUE_FUSION.md) | `CUTLASS_EPILOGUE_FUSION` (environment variable) | The variable names differ, but the behavior is consistent. The default value is `0`. |
| [TORCHINDUCTOR_CATLASS_ENABLED_OPS](inductor/TORCHINDUCTOR_CATLASS_ENABLED_OPS.md) | `TORCHINDUCTOR_CUTLASS_ENABLED_OPS` (environment variable) | The variable names and default values differ. The PyTorch default value is `all`; the TorchNPU default value is `"mm,addmm,bmm"`. |
| [TORCHINDUCTOR_NPU_CATLASS_DIR](inductor/TORCHINDUCTOR_NPU_CATLASS_DIR.md) | `TORCHINDUCTOR_CUTLASS_DIR` (environment variable) | The variable names and template libraries differ. PyTorch configures the path of the CUTLASS (CUDA Templates for Linear Algebra Subroutines) library, while TorchNPU configures the path of the Catlass library. |
| [TORCHNPU_PRECOMPILE_THREADS](inductor/TORCHNPU_PRECOMPILE_THREADS.md) | `TORCHINDUCTOR_COMPILE_THREADS` (related environment variable) | The controlled objects differ. The PyTorch variable controls compilation concurrency; on this basis, TorchNPU extends the control over the number of precompilation threads. |

## Comparison via PyTorch Configuration Items

| TorchNPU Environment Variable | PyTorch Corresponding Configuration | Comparison Description |
| --- | --- | --- |
| [INDUCTOR_ASCEND_LOG_LEVEL](inductor/INDUCTOR_ASCEND_LOG_LEVEL.md) | `torch._inductor.config.log_level` (related configuration item) | PyTorch configures the log level through `torch._inductor.config.log_level` and has no environment variable with the same name. TorchNPU provides the related configuration through this environment variable. |
| [TORCHINDUCTOR_NPU_BACKEND](inductor/TORCHINDUCTOR_NPU_BACKEND.md) | `torch._inductor.config` (related configuration) | PyTorch controls backend behavior through the configuration item and has no environment variable with the same name. TorchNPU uses this variable to select the Inductor compilation mode on NPU. |
