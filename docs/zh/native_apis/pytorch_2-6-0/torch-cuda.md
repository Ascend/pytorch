# torch.cuda

> [!NOTE]  
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。<br>
> 在使用支持的cuda接口时，需要将API名称中的cuda变换为NPU形式才能使用：torch.cuda.变换为torch_npu.npu.或torch.npu.。torch_npu.npu.和torch.npu.两种调用方式，功能一致。举例如下：
>```
>torch.cuda.current_device --> torch_npu.npu.current_device
>torch.cuda.current_device --> torch.npu.current_device
>```

|API名称|NPU形式名称|是否支持|限制与说明|
|--|--|--|--|
|torch.cuda.StreamContext|torch.npu.StreamContext|是|-|
|torch.cuda.can_device_access_peer|torch_npu.npu.can_device_access_peer|是|-|
|torch.cuda.current_blas_handle|torch_npu.npu.current_blas_handle|是|-|
|torch.cuda.current_device|torch_npu.npu.current_device|是|-|
|torch.cuda.current_stream|torch_npu.npu.current_stream|是|未设置device时，调用该接口会隐式地初始化当前device（默认0卡）|
|torch.cuda.default_stream|torch_npu.npu.default_stream|是|未设置device时，调用该接口会隐式地初始化当前device（默认0卡）|
|torch.cuda.device|torch_npu.npu.device|是|-|
|torch.cuda.device_count|torch_npu.npu.device_count|是|-|
|torch.cuda.device_of|torch_npu.npu.device_of|是|-|
|torch.cuda.get_device_capability|-|否|NPU设备无对应概念|
|torch.cuda.get_device_name|torch_npu.npu.get_device_name|是|-|
|torch.cuda.get_device_properties|torch_npu.npu.get_device_properties|是|仅支持name、total_memory、L2_cache_size、cube_core_num和vector_core_num属性，原cuda上支持的其余属性均返回空字段|
|torch.cuda.get_sync_debug_mode|torch_npu.npu.get_sync_debug_mode|是|-|
|torch.cuda.init|torch_npu.npu.init|是|-|
|torch.cuda.ipc_collect|torch_npu.npu.ipc_collect|是|-|
|torch.cuda.is_available|torch_npu.npu.is_available|是|-|
|torch.cuda.is_initialized|torch_npu.npu.is_initialized|是|-|
|torch.cuda.memory_usage|-|否|-|
|torch.cuda.set_device|torch_npu.npu.set_device|是|-|
|torch.cuda.set_stream|torch_npu.npu.set_stream|是|-|
|torch.cuda.set_sync_debug_mode|torch_npu.npu.set_sync_debug_mode|是|-|
|torch.cuda.stream|torch_npu.npu.stream|是|-|
|torch.cuda.synchronize|torch_npu.npu.synchronize|是|-|
|torch.cuda.utilization|torch_npu.npu.utilization|是|-|
|torch.cuda.get_rng_state|torch_npu.npu.get_rng_state|是|-|
|torch.cuda.set_rng_state|torch_npu.npu.set_rng_state|是|-|
|torch.cuda.set_rng_state_all|torch_npu.npu.set_rng_state_all|是|-|
|torch.cuda.manual_seed|torch_npu.npu.manual_seed|是|-|
|torch.cuda.manual_seed_all|torch_npu.npu.manual_seed_all|是|-|
|torch.cuda.seed|torch_npu.npu.seed|是|-|
|torch.cuda.seed_all|torch_npu.npu.seed_all|是|-|
|torch.cuda.initial_seed|torch_npu.npu.initial_seed|是|-|
|torch.cuda.comm.scatter|-|否|-|
|torch.cuda.comm.gather|-|否|-|
|torch.cuda.Stream|torch_npu.npu.Stream|是|-|
|torch.cuda.Stream.wait_stream|torch_npu.npu.Stream.wait_stream|是|-|
|torch.cuda.Event|torch_npu.npu.Event|是|-|
|torch.cuda.Event.elapsed_time|torch_npu.npu.Event.elapsed_time|是|-|
|torch.cuda.Event.query|torch_npu.npu.Event.query|是|-|
|torch.cuda.Event.wait|torch_npu.npu.Event.wait|是|-|
|torch.cuda.is_current_stream_capturing|torch.npu.is_current_stream_capturing|是|-|
|torch.cuda.graph_pool_handle|torch.npu.graph_pool_handle|是|当前仅支持推理场景，不支持训练场景|
|torch.cuda.CUDAGraph|torch.npu.NPUGraph|是|当前仅支持推理场景，不支持训练场景|
|torch.cuda.CUDAGraph.capture_begin|torch.npu.NPUGraph.capture_begin|是|当前仅支持推理场景，不支持训练场景|
|torch.cuda.CUDAGraph.capture_end|torch.npu.NPUGraph.capture_end|是|当前仅支持推理场景，不支持训练场景|
|torch.cuda.CUDAGraph.debug_dump|torch.npu.NPUGraph.debug_dump|是|当前仅支持推理场景，不支持训练场景导出文件内容为json格式|
|torch.cuda.CUDAGraph.pool|torch.npu.NPUGraph.pool|是|当前仅支持推理场景，不支持训练场景|
|torch.cuda.CUDAGraph.replay|torch.npu.NPUGraph.replay|是|当前仅支持推理场景，不支持训练场景|
|torch.cuda.CUDAGraph.reset|torch.npu.NPUGraph.reset|是|当前仅支持推理场景，不支持训练场景|
|torch.cuda.graph|torch.npu.graph|是|当前仅支持推理场景，不支持训练场景|
|torch.cuda.make_graphed_callables|torch.npu.make_graphed_callables|是|当前仅支持推理场景，不支持训练场景|
|torch.cuda.empty_cache|torch_npu.npu.empty_cache|是|-|
|torch.cuda.mem_get_info|torch_npu.npu.mem_get_info|是|-|
|torch.cuda.memory_stats|torch_npu.npu.memory_stats|是|-|
|torch.cuda.memory_summary|torch_npu.npu.memory_summary|是|-|
|torch.cuda.memory_allocated|torch_npu.npu.memory_allocated|是|-|
|torch.cuda.max_memory_allocated|torch_npu.npu.max_memory_allocated|是|-|
|torch.cuda.reset_max_memory_allocated|torch_npu.npu.reset_max_memory_allocated|是|-|
|torch.cuda.memory_reserved|torch_npu.npu.memory_reserved|是|-|
|torch.cuda.max_memory_reserved|torch_npu.npu.max_memory_reserved|是|-|
|torch.cuda.set_per_process_memory_fraction|torch_npu.npu.set_per_process_memory_fraction|是|-|
|torch.cuda.memory_cached|torch_npu.npu.memory_cached|是|-|
|torch.cuda.max_memory_cached|torch_npu.npu.max_memory_cached|是|-|
|torch.cuda.reset_max_memory_cached|torch_npu.npu.reset_max_memory_cached|是|-|
|torch.cuda.reset_peak_memory_stats|torch_npu.npu.reset_peak_memory_stats|是|-|
|torch.cuda.caching_allocator_alloc|torch_npu.npu.caching_allocator_alloc|是|-|
|torch.cuda.caching_allocator_delete|torch_npu.npu.caching_allocator_delete|是|-|
|torch.cuda.get_allocator_backend|torch_npu.npu.get_allocator_backend|是|-|
|torch.cuda.CUDAPluggableAllocator|torch_npu.npu.NPUPluggableAllocator|是|该接口涉及高危操作，使用请参考《Ascend Extension for PyTorch 自定义 API参考》中的“[torch_npu.npu.NPUPluggableAllocator](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch-npu-npu-NPUPluggableAllocator.md)”章节。|
|torch.cuda.change_current_allocator|torch_npu.npu.change_current_allocator|是|该接口涉及高危操作，使用请参考《Ascend Extension for PyTorch 自定义 API参考》中的“[torch_npu.npu.change_current_allocator](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch-npu-npu-change_current_allocator.md)”章节。|
|torch.cuda._sanitizer.enable_cuda_sanitizer|torch_npu.npu._sanitizer.enable_npu_sanitizer|是|-|


