# Environment Variable Reference

- [Environment Variable List](env_variable_list.md)
- [PyTorch Environment Variable Comparison Table](pytorch_comparison.md)
<!-- npu="950,A3,910b,910,310p" id19 -->
- [Device Initialization and Management](device_management/_menu_device_management.md)
  <!-- npu="A3,910b,910,310p" id1 -->
  - [TORCH_ACL_INIT_CONFIG_PATH](device_management/TORCH_ACL_INIT_CONFIG_PATH.md)
  <!-- end id1 -->
  <!-- npu="950,A3,910b,910" id2 -->
  - [PYTORCH_HAL_BASED_NPU_CHECK](device_management/PYTORCH_HAL_BASED_NPU_CHECK.md)
  <!-- end id2 -->
  <!-- npu="950,A3,910b,910" id3 -->
  - [STREAMS_PER_DEVICE](device_management/STREAMS_PER_DEVICE.md)
  <!-- end id3 -->
  <!-- npu="950,A3,910b,910,310p" id4 -->
  - [ACL_DEVICE_SYNC_TIMEOUT](device_management/ACL_DEVICE_SYNC_TIMEOUT.md)
  <!-- end id4 -->
  <!-- npu="950,A3,910b,910" id5 -->
  - [TORCH_NPU_DEVICE_CAPABILITY](device_management/TORCH_NPU_DEVICE_CAPABILITY.md)
  <!-- end id5 -->
  <!-- npu="950,A3,910b,910" id6 -->
  - [TORCH_TRANSFER_TO_NPU](device_management/TORCH_TRANSFER_TO_NPU.md)
  <!-- end id6 -->
<!-- end id19 -->
<!-- npu="950,A3,910b,910,310p" id20 -->
- [Operator Execution and Compatibility](op_execution/_menu_op_execution.md)
  <!-- npu="A3,910b" id7 -->
  - [INF_NAN_MODE_ENABLE](op_execution/INF_NAN_MODE_ENABLE.md)
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - [INF_NAN_MODE_FORCE_DISABLE](op_execution/INF_NAN_MODE_FORCE_DISABLE.md)
  <!-- end id8 -->
  <!-- npu="950,A3,910b" id9 -->
  - [FORCE_OVERFLOW_CHECK](op_execution/FORCE_OVERFLOW_CHECK.md)
  <!-- end id9 -->
  <!-- npu="A3,910b,910,310p" id10 -->
  - [COMBINED_ENABLE](op_execution/COMBINED_ENABLE.md)
  <!-- end id10 -->
  <!-- npu="950,A3,910b,910,310p" id11 -->
  - [ASCEND_LAUNCH_BLOCKING](op_execution/ASCEND_LAUNCH_BLOCKING.md)
  <!-- end id11 -->
  <!-- npu="950,A3,910b,910,310p" id12 -->
  - [TASK_QUEUE_ENABLE](op_execution/TASK_QUEUE_ENABLE.md)
  <!-- end id12 -->
  <!-- npu="950,A3,910b,910,310p" id13 -->
  - [PER_STREAM_QUEUE](op_execution/PER_STREAM_QUEUE.md)
  <!-- end id13 -->
  <!-- npu="950,A3,910b,910,310p" id14 -->
  - [TORCH_NPU_FALLBACK_CPU_DISABLE](op_execution/TORCH_NPU_FALLBACK_CPU_DISABLE.md)
  <!-- end id14 -->
  <!-- npu="950,A3,910b,910,310p" id15 -->
  - [TORCH_NPU_USE_COMPATIBLE_IMPL](op_execution/TORCH_NPU_USE_COMPATIBLE_IMPL.md)
  <!-- end id15 -->
  <!-- npu="950" id16 -->
  - [TORCH_NPU_LEGACY_IMPL_LIST](op_execution/TORCH_NPU_LEGACY_IMPL_LIST.md)
  <!-- end id16 -->
<!-- end id20 -->
<!-- npu="A3,910b,910" id17 -->
- [Single-Operator Compilation and Caching](op_compilation/_menu_op_compilation.md)
  - [ACL_OP_COMPILER_CACHE_DIR](op_compilation/ACL_OP_COMPILER_CACHE_DIR.md)
  - [ACL_OP_COMPILER_CACHE_MODE](op_compilation/ACL_OP_COMPILER_CACHE_MODE.md)
<!-- end id17 -->
<!-- npu="950,A3,910b,910,310p" id28 -->
- [Memory Management](memory_management/_menu_memory_management.md)
  <!-- npu="A3,910b,910,310p" id18 -->
  - [PYTORCH_ALLOC_CONF](memory_management/PYTORCH_ALLOC_CONF.md)
  <!-- end id18 -->
  <!-- npu="A3,910b,910,310p" id21 -->
  - [PYTORCH_NPU_ALLOC_CONF](memory_management/PYTORCH_NPU_ALLOC_CONF.md)
  <!-- end id21 -->
  <!-- npu="950,A3,910b,910,310p" id22 -->
  - [PYTORCH_NO_NPU_MEMORY_CACHING](memory_management/PYTORCH_NO_NPU_MEMORY_CACHING.md)
  <!-- end id22 -->
  <!-- npu="950,A3,910b,910,310p" id23 -->
  - [OOM_SNAPSHOT_ENABLE](memory_management/OOM_SNAPSHOT_ENABLE.md)
  <!-- end id23 -->
  <!-- npu="950,A3,910b,910,310p" id24 -->
  - [OOM_SNAPSHOT_PATH](memory_management/OOM_SNAPSHOT_PATH.md)
  <!-- end id24 -->
  <!-- npu="950,A3,910b,910" id25 -->
  - [MULTI_STREAM_MEMORY_REUSE](memory_management/MULTI_STREAM_MEMORY_REUSE.md)
  <!-- end id25 -->
  <!-- npu="950,A3,910b,910,310p" id26 -->
  - [TORCH_NPUGRAPH_GC](memory_management/TORCH_NPUGRAPH_GC.md)
  <!-- end id26 -->
  <!-- npu="950,A3,910b,910" id27 -->
  - [NPU_SHMEM_SYMMETRIC_SIZE](memory_management/NPU_SHMEM_SYMMETRIC_SIZE.md)
  <!-- end id27 -->
<!-- end id28 -->
<!-- npu="950,A3,910b,910,310p" id41 -->
- [Distributed Communication and Diagnosis](collective_communication/_menu_collective_communication.md)
  <!-- npu="950,A3,910b,910,310p" id29 -->
  - [TORCH_HCCL_BLOCKING_WAIT](collective_communication/TORCH_HCCL_BLOCKING_WAIT.md)
  <!-- end id29 -->
  <!-- npu="A3,910b,910,310p" id30 -->
  - [TORCH_HCCL_ASYNC_ERROR_HANDLING](collective_communication/TORCH_HCCL_ASYNC_ERROR_HANDLING.md)
  <!-- end id30 -->
  <!-- npu="950,A3,910b,910" id31 -->
  - [HCCL_EVENT_TIMEOUT](collective_communication/HCCL_EVENT_TIMEOUT.md)
  <!-- end id31 -->
  <!-- npu="950,A3,910b,910" id32 -->
  - [P2P_HCCL_BUFFSIZE](collective_communication/P2P_HCCL_BUFFSIZE.md)
  <!-- end id32 -->
  <!-- npu="950,A3,910b,910" id33 -->
  - [RANK_TABLE_FILE](collective_communication/RANK_TABLE_FILE.md)
  <!-- end id33 -->
  <!-- npu="A3,910b" id34 -->
  - [ROOTINFO_SCALABLE_ENABLE](collective_communication/ROOTINFO_SCALABLE_ENABLE.md)
  <!-- end id34 -->
  <!-- npu="A3,910b" id35 -->
  - [TORCH_HCCL_RANKS_PER_ROOT](collective_communication/TORCH_HCCL_RANKS_PER_ROOT.md)
  <!-- end id35 -->
  <!-- npu="A3" id36 -->
  - [(beta) TORCH_HCCL_ZERO_COPY](collective_communication/（beta）TORCH_HCCL_ZERO_COPY.md)
  <!-- end id36 -->
  <!-- npu="950,A3,910b,910,310p" id37 -->
  - [TORCH_HCCL_DESYNC_DEBUG](collective_communication/TORCH_HCCL_DESYNC_DEBUG.md)
  <!-- end id37 -->
  <!-- npu="950,A3,910b,910" id38 -->
  - [TORCH_HCCL_STATUS_SAVE_ENABLE](collective_communication/TORCH_HCCL_STATUS_SAVE_ENABLE.md)
  <!-- end id38 -->
  <!-- npu="950,A3,910b,910" id39 -->
  - [TORCH_HCCL_STATUS_SAVE_PATH](collective_communication/TORCH_HCCL_STATUS_SAVE_PATH.md)
  <!-- end id39 -->
  <!-- npu="950,A3,910b,910" id40 -->
  - [TORCH_HCCL_STATUS_SAVE_INTERVAL](collective_communication/TORCH_HCCL_STATUS_SAVE_INTERVAL.md)
  <!-- end id40 -->
<!-- end id41 -->
<!-- npu="950,A3,910b,910,310p" id47 -->
- [Logs and Alarms](alarm_message_printing/_menu_alarm_message_printing.md)
  <!-- npu="950,A3,910b,910,310p" id42 -->
  - [TORCH_NPU_WARNING_DISABLE](alarm_message_printing/TORCH_NPU_WARNING_DISABLE.md)
  <!-- end id42 -->
  <!-- npu="950,A3,910b,910,310p" id43 -->
  - [TORCH_NPU_DISABLED_WARNING](alarm_message_printing/TORCH_NPU_DISABLED_WARNING.md)
  <!-- end id43 -->
  <!-- npu="950,A3,910b,910" id44 -->
  - [TORCH_NPU_COMPACT_ERROR_OUTPUT](alarm_message_printing/TORCH_NPU_COMPACT_ERROR_OUTPUT.md)
  <!-- end id44 -->
  <!-- npu="950,A3,910b,910,310p" id45 -->
  - [TORCH_NPU_LOGS](alarm_message_printing/TORCH_NPU_LOGS.md)
  <!-- end id45 -->
  <!-- npu="950,A3,910b,910,310p" id46 -->
  - [TORCH_NPU_LOGS_FILTER](alarm_message_printing/TORCH_NPU_LOGS_FILTER.md)
  <!-- end id46 -->
<!-- end id47 -->
<!-- npu="950,A3,910b,910,310p" id52 -->
- [Feature Value Detection](eigenvalue_detection/_menu_eigenvalue_detection.md)
  <!-- npu="950,A3,910b,910,310p" id48 -->
  - [NPU_ASD_ENABLE](eigenvalue_detection/NPU_ASD_ENABLE.md)
  <!-- end id48 -->
  <!-- npu="A3,910b,910,310p" id49 -->
  - [NPU_ASD_UPPER_THRESH](eigenvalue_detection/NPU_ASD_UPPER_THRESH.md)
  <!-- end id49 -->
  <!-- npu="A3,910b,910,310p" id50 -->
  - [NPU_ASD_SIGMA_THRESH](eigenvalue_detection/NPU_ASD_SIGMA_THRESH.md)
  <!-- end id50 -->
  <!-- npu="950,A3,910b,910" id51 -->
  - [NPU_ASD_CONFIG](eigenvalue_detection/NPU_ASD_CONFIG.md)
  <!-- end id51 -->
<!-- end id52 -->
<!-- npu="950,A3,910b,910" id57 -->
- [Performance Optimization and Collection](performance_tuning/_menu_performance_tuning.md)
  <!-- npu="950,A3,910b,910" id53 -->
  - [CPU_AFFINITY_CONF](performance_tuning/CPU_AFFINITY_CONF.md)
  <!-- end id53 -->
  <!-- npu="A3,910b" id54 -->
  - [TORCH_NPU_LAZY_FUSION](performance_tuning/TORCH_NPU_LAZY_FUSION.md)
  <!-- end id54 -->
  <!-- npu="950,A3,910b,910" id55 -->
  - [PROF_CONFIG_PATH](performance_tuning/PROF_CONFIG_PATH.md)
  <!-- end id55 -->
  <!-- npu="950,A3,910b,910" id56 -->
  - [KINETO_USE_DAEMON](performance_tuning/KINETO_USE_DAEMON.md)
  <!-- end id56 -->
<!-- end id57 -->
<!-- npu="950,A3,910b,910" id77 -->
- [Graph Compilation (torch.compile / Inductor)](inductor/_menu_inductor.md)
  <!-- npu="950,A3,910b,910" id58 -->
  - [TORCHINDUCTOR_NPU_BACKEND](inductor/TORCHINDUCTOR_NPU_BACKEND.md)
  <!-- end id58 -->
  <!-- npu="950,A3,910b,910" id59 -->
  - [TORCH_CACHING_PRECOMPILE](inductor/TORCH_CACHING_PRECOMPILE.md)
  <!-- end id59 -->
  <!-- npu="950,910" id60 -->
  - [TORCHINDUCTOR_COMPILE_THREADS](inductor/TORCHINDUCTOR_COMPILE_THREADS.md)
  <!-- end id60 -->
  <!-- npu="950,910" id61 -->
  - [TORCHNPU_PRECOMPILE_THREADS](inductor/TORCHNPU_PRECOMPILE_THREADS.md)
  <!-- end id61 -->
  <!-- npu="950,910" id62 -->
  - [TORCHINDUCTOR_MAX_AUTOTUNE](inductor/TORCHINDUCTOR_MAX_AUTOTUNE.md)
  <!-- end id62 -->
  <!-- npu="950" id63 -->
  - [TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS](inductor/TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS.md)
  <!-- end id63 -->
  <!-- npu="950" id64 -->
  - [TORCHINDUCTOR_NPU_CATLASS_DIR](inductor/TORCHINDUCTOR_NPU_CATLASS_DIR.md)
  <!-- end id64 -->
  <!-- npu="950" id65 -->
  - [TORCHINDUCTOR_CATLASS_ENABLED_OPS](inductor/TORCHINDUCTOR_CATLASS_ENABLED_OPS.md)
  <!-- end id65 -->
  <!-- npu="950" id66 -->
  - [CATLASS_EPILOGUE_FUSION](inductor/CATLASS_EPILOGUE_FUSION.md)
  <!-- end id66 -->
  <!-- npu="950" id67 -->
  - [TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING](inductor/TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING.md)
  <!-- end id67 -->
  <!-- npu="950" id68 -->
  - [INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE](inductor/INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE.md)
  <!-- end id68 -->
  <!-- npu="950" id69 -->
  - [INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE](inductor/INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE.md)
  <!-- end id69 -->
  <!-- npu="950" id70 -->
  - [INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES](inductor/INDUCTOR_ASCEND_SYMBOLIC_GROUP_TEMPLATES.md)
  <!-- end id70 -->
  <!-- npu="950" id71 -->
  - [INDUCTOR_ASCEND_ENABLE_COSTMODEL](inductor/INDUCTOR_ASCEND_ENABLE_COSTMODEL.md)
  <!-- end id71 -->
  <!-- npu="950" id72 -->
  - [INDUCTOR_ASCEND_COSTMODEL_RATIO](inductor/INDUCTOR_ASCEND_COSTMODEL_RATIO.md)
  <!-- end id72 -->
  <!-- npu="950,910" id73 -->
  - [ENABLE_PARALLEL_SCHEDULER](inductor/ENABLE_PARALLEL_SCHEDULER.md)
  <!-- end id73 -->
  <!-- npu="950,910" id74 -->
  - [SHUT_DOWN_FX_PASS_LIST](inductor/SHUT_DOWN_FX_PASS_LIST.md)
  <!-- end id74 -->
  <!-- npu="950,A3,910b,910" id75 -->
  - [(beta) INDUCTOR_ASCEND_CHECK_ACCURACY](inductor/INDUCTOR_ASCEND_CHECK_ACCURACY.md)
  <!-- end id75 -->
  <!-- npu="950,910" id76 -->
  - [INDUCTOR_ASCEND_LOG_LEVEL](inductor/INDUCTOR_ASCEND_LOG_LEVEL.md)
  <!-- end id76 -->
<!-- end id77 -->
- [Appendix](appendix/_menu_appendix.md)
  - [Error "RuntimeError: The Inner Error ..." Reported During Distributed Model Training](appendix/runtimeerror_Inner_Error.md)
