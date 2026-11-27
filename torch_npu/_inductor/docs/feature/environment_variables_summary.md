# 环境变量列表

本手册描述开发者在使用Inductor-Ascend过程中可使用的环境变量。
TorchNPU环境变量请参考《[TorchNPU环境变量参考](https://www.hiascend.com/document/detail/zh/Pytorch/730/comref/Envvariables/docs/zh/environment_variable_reference/env_variable_list.md)》。
基于CANN构建AI应用和业务过程中使用的环境变量请参考《[CANN 环境变量参考](https://www.hiascend.com/document/detail/zh/canncommercial/850/maintenref/envvar/envref_07_0001.html)》。

以下按特性分类列出环境变量及其说明。

## Catlass

| 环境变量名称 | 简介 |
| --- | --- |
| CATLASS_EPILOGUE_FUSION | 是否开启Catlass cv融合，与社区CUTLASS_EPILOGUE_FUSION保持一致，社区环境变量为CUTLASS_EPILOGUE_FUSION，默认值为0 |
| TORCHINDUCTOR_CATLASS_ENABLED_OPS | Catlass可作用于的矩阵乘类的算子类型，与社区TORCHINDUCTOR_CUTLASS_ENABLED_OPS保持一致，社区环境变量为TORCHINDUCTOR_CUTLASS_ENABLED_OPS，默认值为"mm,addmm,bmm" |
| TORCHINDUCTOR_MAX_AUTOTUNE | 开启max autotune功能，该环境变量与社区一致，默认值为0 |
| TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS | 确认max autotune可尝试的后端有哪些，该环境变量与社区一致，若想尝试Catlass的后端，请在该环境变量中配置上"CATLASS"，默认值为"ATEN,TRITON,CPP" |
| TORCHINDUCTOR_NPU_CATLASS_DIR | 指定 CATLASS 库的路径。未设置时，默认使用 `torch` 安装目录上一级的 `third_party/catlass`。 |
| TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING | 该环境变量与社区一致，用于管理autotune过程中是否使用profiling进行autotune，"0"为不使用profiling，"1"为使用profiling，默认值为0 |

## FXGraph图优化

| 环境变量名称 | 简介 |
| --- | --- |
| SHUT_DOWN_FX_PASS_LIST | 用于精确控制需要关闭的pass（关闭列表），默认为""，即不关闭任何pass，所有pass都生效 |
| TORCHINDUCTOR_ENABLE_FUSED_MATMUL_RELU | 控制fused_matmul_relu_pass的开关，将relu(addmm)等图形态融合为npu_fused_matmul，relu不再作为独立的pointwise kernel下发，默认不设置，等价于关闭 |
| TORCHINDUCTOR_ENABLE_GRAD_MATMUL_TRANSPOSE_OPT | 控制grad_matmul_transpose_opt_pass的开关，将反向输出上的permute(mm(lhs,rhs),[1,0])改写为mm(rhs.T,lhs.T)，省去对mm结果再做一次转置搬移，默认不设置，等价于关闭 |
| TORCHINDUCTOR_ENABLE_GROUPED_MATMUL_FUSION | 控制grouped_matmul_fusion_pass的开关，将汇聚到同一个cat的多个互相独立的小GEMM合并成一次npu_grouped_matmul调用，默认不设置，等价于关闭 |
| TORCHINDUCTOR_ENABLE_MULTI_SLICE_CONCAT | 控制multi_slice_concat_pass的开关，将汇聚到同一个cat的一串固定宽度、常量偏移的列切片折叠成单个npu_ext::multi_slice_concat，默认不设置，等价于关闭 |

## 计算图多流并行

| 环境变量名称 | 简介 |
| --- | --- |
| ENABLE_PARALLEL_SCHEDULER | 是否开启计算图多流并行调度策略，默认为False，即不开启计算图多流并行调度策略，设置为True，表示开启计算图多流并行调度策略 |

## 离散访存

| 环境变量名称 | 简介 |
| --- | --- |
| USE_STORE_IN_CAT | 用于控制Inductor针对cat融合的行为，当前默认为False |

## 动态shape

| 环境变量名称 | 简介 |
| --- | --- |
| INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE | 控制是否启用动态shape分组autotune（grouped autotune），开启后按shape特征将运行时shape划分到不同分组，每组使用代表shape进行一次autotune，默认值为0 |
| INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES | 配置参与动态shape分组autotune的模板类型列表，多个模板以逗号分隔，默认值为"pointwise,reduction,persistent_reduction" |
| INDUCTOR_ASCEND_SYMBOLIC_GROUP_MAX_BENCHMARK_MEMORY_RATIO | 控制动态shape分组autotune中分组benchmark显存占用预算的比例上限，预算值为NPU总显存乘以该比例，估算显存超过预算时自动回退到普通autotune流程。取值范围为(0.0, 1.0]，默认值为0.25，取值非法时按默认值处理 |

## 自动Tiling优化

| 环境变量名称 | 简介 |
| --- | --- |
| FASTAUTOTUNE | 控制是否使用fast autotune，默认值为0 |
| INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE | 控制是否启用batch profiler，默认值为0 |
| TORCHINDUCTOR_COMPILE_THREADS | 多进程编译进程数量，与社区保持一致，默认值为32 |
| TORCHNPU_PRECOMPILE_THREADS | 设置预编译线程数。未设置时，默认值为 `max(os.cpu_count() // max(compile_threads, 2), 32)`，其中 `compile_threads` 是 Inductor 的编译进程数。 |

## 分核/限核

| 环境变量名称 | 简介 |
| --- | --- |
| NPU_DEVICE_LIMIT | 控制最多可使用的Cube和Vector的核数，默认值为全部Cube和Vector核 |

## CostModel

| 环境变量名称 | 简介 |
| --- | --- |
| INDUCTOR_ASCEND_ENABLE_COSTMODEL | 控制是否启用CostModel预筛选，默认值为0 |
| INDUCTOR_ASCEND_COSTMODEL_RATIO | 控制CostModel预筛选后保留的config比例，默认值为0.25 |

## FlexAttention

| 环境变量名称 | 简介 |
| --- | --- |
| TORCHINDUCTOR_ASCEND_FLEX_ATTENTION_BWD_DKDV_TASKLIST | 是否允许反向dK/dV任务列表调度，默认值为1 |

## 其他

| 环境变量名称 | 简介 |
| --- | --- |
| INDUCTOR_ASCEND_CHECK_ACCURACY | 开启Triton后端精度对比工具，dump单算子用例。当启用时，会自动启用INDUCTOR_ASCEND_DUMP_FX_GRAPH功能，默认值为空 |
| INDUCTOR_ASCEND_DUMP_FX_GRAPH | dump可执行的单算子用例，用于调试和问题排查。当INDUCTOR_ASCEND_CHECK_ACCURACY启用时，会自动启用此功能，默认值为空 |
| INDUCTOR_ASCEND_LOG_LEVEL | 设置Inductor-Ascend日志等级，控制日志输出的详细程度，默认值为WARNING |
| TORCHINDUCTOR_NDDMA | 启用Triton-Ascend load随路转置能力。在Atlas A2系列产品、Atlas A3系列产品代际理论性能无差异。在Ascend 950PR&950DT系列产品代际会通过底层nddma特性做转置加速，转置性能有明显增益 |
| ENABLE_INPLACE_BUFFERS | 控制Inductor-Ascend生成Triton Kernel时输入/输出参数是否复用地址空间，未设置或设置为1、true、yes时复用（默认），设置为0、false、no等时不复用，便于multi-buffer流水掩盖 |
| TORCHINDUCTOR_NPU_FAST_LAUNCH | 控制是否启用Planned Fast Launch，用于降低torch.compile生成的Python Wrapper在稳态运行时下发NPU Triton融合Kernel的Host侧固定开销，默认关闭 |
