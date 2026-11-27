# 环境变量列表

本手册描述开发者在使用inductor-ascend过程中可使用的环境变量。
TorchNPU环境变量请参考《[TorchNPU环境变量参考](https://www.hiascend.com/document/detail/zh/Pytorch/730/comref/Envvariables/docs/zh/environment_variable_reference/env_variable_list.md)》。
基于CANN构建AI应用和业务过程中使用的环境变量请参考《[CANN 环境变量参考](https://www.hiascend.com/document/detail/zh/canncommercial/850/maintenref/envvar/envref_07_0001.html)》。

以下按特性分类列出环境变量及其说明。

## Catlass

| 环境变量名称 | 简介 |
|------------|-|
| CATLASS_EPILOGUE_FUSION | 是否开启catlass cv融合，与社区CUTLASS_EPILOGUE_FUSION保持一致，社区环境变量为CUTLASS_EPILOGUE_FUSION，默认值为0 |
| TORCHINDUCTOR_CATLASS_ENABLED_OPS | catlass可作用于的矩阵乘类的算子类型，与社区TORCHINDUCTOR_CUTLASS_ENABLED_OPS保持一致，社区环境变量为TORCHINDUCTOR_CUTLASS_ENABLED_OPS，默认值为"mm,addmm,bmm" |
| TORCHINDUCTOR_MAX_AUTOTUNE | 开启max autotune功能，该环境变量与社区一致，默认值为0 |
| TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS | 确认max autotune可尝试的后端有哪些，该环境变量与社区一致，若想尝试Catlass的后端，请在该环境变量中配置上"CATLASS"，默认值为"ATEN,TRITON,CPP" |
| TORCHINDUCTOR_NPU_CATLASS_DIR | 指定 CATLASS 库的路径。未设置时，默认使用 `torch` 安装目录上一级的 `third_party/catlass`。 |
| TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING | 该环境变量与社区一致，用于管理autotune过程中是否使用profiling进行autotune，"0"为不使用profiling，"1"为使用profiling，默认值为0 |

## FXGraph图优化

| 环境变量名称 | 简介 |
|------------|-|
| SHUT_DOWN_FX_PASS_LIST | 用于精确控制生效的pass，默认为""，即所有pass都生效 |

## 计算图多流并行

| 环境变量名称 | 简介 |
|------------|-|
| ENABLE_PARALLEL_SCHEDULER | 是否开启计算图多流并行调度策略，默认为False，即不开启计算图多流并行调度策略，设置为True，表示开启计算图多流并行调度策略 |

## 离散访存

| 环境变量名称 | 简介 |
|------------|-|
| INDUCTOR_INDIRECT_MEMORY_MODE | 是否开启离散访存的融合以及配置融合方式，默认值为"simd_simt_mix" |
| USE_STORE_IN_CAT | 用于控制Inductor针对cat融合的行为，当前默认为False |

## 动态shape

| 环境变量名称 | 简介 |
|------------|-|
| INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE | 控制是否启用动态shape分组autotune（grouped autotune），默认值为"0"，配置为"1"、"true"或"yes"时开启 |
| INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES | 控制参与动态shape分组autotune的模板类型列表，多个模板以逗号分隔，默认值为"pointwise,reduction,persistent_reduction" |
| INDUCTOR_ASCEND_SYMBOLIC_GROUP_MAX_BENCHMARK_MEMORY_RATIO | 控制分组autotune benchmark的显存占用预算，超过预算时自动回退到普通autotune流程，默认值为0.25 |

## 自动Tiling优化

| 环境变量名称 | 简介 |
|------------|-|
| FASTAUTOTUNE | 控制是否使用fast autotune，默认值为0 |
| INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE | 控制是否启用batch profiler，默认值为0 |
| TORCHINDUCTOR_COMPILE_THREADS | 多进程编译进程数量，与社区保持一致，默认值为32 |
| TORCHNPU_PRECOMPILE_THREADS | 设置预编译线程数。未设置时，默认值为 `max(os.cpu_count() // max(compile_threads, 2), 32)`，其中 `compile_threads` 是 Inductor 的编译进程数。 |

## 分核/限核

| 环境变量名称 | 简介 |
|------------|-|
| NPU_DEVICE_LIMIT | 控制最多可使用的Cube和Vector的核数，默认值为全部cube和vector核 |

## CostModel

| 环境变量名称 | 简介 |
|------------|-|
| INDUCTOR_ASCEND_ENABLE_COSTMODEL | 控制是否启用CostModel预筛选，默认值为0 |
| INDUCTOR_ASCEND_COSTMODEL_RATIO | 控制CostModel预筛选后保留的config比例，默认值为0.25 |

## 其他

| 环境变量名称 | 简介 |
|------------|-|
| INDUCTOR_ASCEND_CHECK_ACCURACY | 开启triton后端精度对比工具，dump单算子用例。当启用时，会自动启用INDUCTOR_ASCEND_DUMP_FX_GRAPH功能，默认值为空。 |
| INDUCTOR_ASCEND_DUMP_FX_GRAPH | dump可执行的单算子用例，用于调试和问题排查。当INDUCTOR_ASCEND_CHECK_ACCURACY或AOTI_ASCEND_DEBUG_KERNEL启用时，会自动启用此功能，默认值为空。 |
| INDUCTOR_ASCEND_LOG_LEVEL | 设置Inductor-Ascend日志等级，控制日志输出的详细程度，默认值为WARNING。 |
| TORCHINDUCTOR_NDDMA | 启用Triton-Ascend load随路转置能力。在A2、A3代际理论性能无差异。在A5代际会通过底层nddma特性做转置加速，转置性能有明显增益。 |
| TORCHINDUCTOR_NPU_FAST_LAUNCH | 控制是否启用NPU Inductor Planned Fast Launch，用于降低torch.compile生成的Python Wrapper下发NPU Triton融合kernel的Host侧固定开销，默认关闭，需要显式启用 |
| AOTI_ASCEND_DEBUG_KERNEL | 开启AOTI调试内核。当启用时，会自动启用INDUCTOR_ASCEND_DUMP_FX_GRAPH功能，默认值为空 |
