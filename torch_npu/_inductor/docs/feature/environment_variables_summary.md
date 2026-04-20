# 环境变量列表
本手册描述开发者在使用inductor-ascend过程中可使用的环境变量。
Ascend Extension for PyTorch环境变量请参考《[Ascend Extension for PyTorch环境变量参考](https://www.hiascend.com/document/detail/zh/Pytorch/730/comref/Envvariables/docs/zh/environment_variable_reference/env_variable_list.md)》。
基于CANN构建AI应用和业务过程中使用的环境变量请参考《[CANN 环境变量参考](https://www.hiascend.com/document/detail/zh/canncommercial/850/maintenref/envvar/envref_07_0001.html)》。

**表 1**  环境变量列表

| 环境变量类型     |环境变量名称| 简介                                                                                                                              |
|------------|-|---------------------------------------------------------------------------------------------------------------------------------|
| Catlass    |CATLASS_EPILOGUE_FUSION| 是否开启catlass cv融合，与社区CUTLASS_EPILOGUE_FUSION保持一致，社区环境变量为CUTLASS_EPILOGUE_FUSION，默认值为0                                                                   |
| Catlass    |TORCHINDUCTOR_CATLASS_ENABLED_OPS| catlass可作用于的矩阵乘类的算子类型，与社区TORCHINDUCTOR_CUTLASS_ENABLED_OPS保持一致，社区环境变量为TORCHINDUCTOR_CUTLASS_ENABLED_OPS，默认值为"mm,addmm,bmm"                                  |
| Catlass    |TORCHINDUCTOR_MAX_AUTOTUNE| 开启max autotune功能，该环境变量与社区一致，默认值为0                                                |
| Catlass    |TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS| 确认max autotune可尝试的后端有哪些，该环境变量与社区一致，若想尝试Catlass的后端，请在该环境变量中配置上"CATLASS"，默认值为"ATEN,TRITON,CPP" |
| Catlass    |TORCHINDUCTOR_NPU_CATLASS_DIR| 环境中catlass库的路径，与社区TORCHINDUCTOR_CUTLASS_DIR保持一致，社区环境变量为TORCHINDUCTOR_CUTLASS_DIR，若路径配置错误，会有WARNING信息提示，并跳过尝试引入catlass后端的功能，默认值为""         |
| Catlass    |TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING| 该环境变量与社区一致，用于管理autotune过程中是否使用profiling进行autotune，"0"为不使用profiling，"1"为使用profiling，默认值为0                  |
| FXGraph图优化 |SHUT_DOWN_FX_PASS_LIST| 用于精确控制生效的pass，默认为""，即所有pass都生效                                                                                                  |
| 离散访存       |INDUCTOR_INDIRECT_MEMORY_MODE| 是否开启离散访存的融合以及配置融合方式，默认值为"simd_simt_mix"                                                                                                            |
| 离散访存       |USE_STORE_IN_CAT| 用于控制Inductor针对cat融合的行为，当前默认为False                                                                                               |
| 自动Tiling优化 |FASTAUTOTUNE| 控制是否使用fast autotune，默认值为0                                                                                                       |
| 自动Tiling优化 |INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE| 控制是否启用batch profiler，默认值为0                                                                                                      |
| 自动Tiling优化 |TORCHINDUCTOR_COMPILE_THREADS| 多进程编译进程数量，与社区保持一致，默认值为32                                                                                                                |
| 自动Tiling优化 |TORCHNPU_PRECOMPILE_THREADS| 控制多线程编译线程数量，默认为最大核数的一半（max_precompiled_thread_num = os.cpu_count() // 2），大于1时，使用并发编译                                            |
| 其他         |INDUCTOR_ASCEND_CHECK_ACCURACY| 开启triton后端精度对比工具，dump单算子用例。当启用时，会自动启用INDUCTOR_ASCEND_DUMP_FX_GRAPH功能，默认值为空。                                                     |
| 其他         |INDUCTOR_ASCEND_DUMP_FX_GRAPH| dump可执行的单算子用例，用于调试和问题排查。当INDUCTOR_ASCEND_CHECK_ACCURACY或AOTI_ASCEND_DEBUG_KERNEL启用时，会自动启用此功能，默认值为空。                             |
| 其他         |INDUCTOR_ASCEND_LOG_LEVEL| 设置Inductor-Ascend日志等级，控制日志输出的详细程度，默认值为WARNING。                                                                                  |
| 其他         |TORCHINDUCTOR_NDDMA| 启用Triton-Ascend load随路转置能力。在A2、A3代际理论性能无差异。在A5代际会通过底层nddma特性做转置加速，转置性能有明显增益。                                                    |
