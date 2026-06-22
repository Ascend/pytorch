# torch.cuda

> [!NOTE]  
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。<br>
> 在使用支持的CUDA接口时，需要将API名称中的CUDA替换为NPU形式才能使用：将torch.cuda.替换为torch_npu.npu.或torch.npu.。torch_npu.npu.和torch.npu.两种调用方式，功能一致。举例如下：
>
> `torch.cuda.current_device` --> `torch_npu.npu.current_device`<br>
> `torch.cuda.current_device` --> `torch.npu.current_device`

|API名称|NPU形式名称|是否支持|限制与说明|
|--|--|--|--|
|[torch.cuda.StreamContext](https://pytorch.org/docs/2.12/generated/torch.cuda.StreamContext.html)|torch.npu.StreamContext|是|-|
|[torch.cuda.can_device_access_peer](https://pytorch.org/docs/2.12/generated/torch.cuda.can_device_access_peer.html)|torch_npu.npu.can_device_access_peer|是|-|
|[torch.cuda.current_blas_handle](https://pytorch.org/docs/2.12/generated/torch.cuda.current_blas_handle.html)|torch_npu.npu.current_blas_handle|是|-|
|[torch.cuda.current_device](https://pytorch.org/docs/2.12/generated/torch.cuda.current_device.html)|torch_npu.npu.current_device|是|-|
|[torch.cuda.current_stream](https://pytorch.org/docs/2.12/generated/torch.cuda.current_stream.html)|torch_npu.npu.current_stream|是|未设置device时，调用该接口会隐式地初始化当前device（默认0卡）|
|[torch.cuda.default_stream](https://pytorch.org/docs/2.12/generated/torch.cuda.default_stream.html)|torch_npu.npu.default_stream|是|未设置device时，调用该接口会隐式地初始化当前device（默认0卡）|
|[torch.cuda.device](https://pytorch.org/docs/2.12/generated/torch.cuda.device.html)|torch_npu.npu.device|是|-|
|[torch.cuda.device_count](https://pytorch.org/docs/2.12/generated/torch.cuda.device_count.html)|torch_npu.npu.device_count|是|-|
|[torch.cuda.device_of](https://pytorch.org/docs/2.12/generated/torch.cuda.device_of.html)|torch_npu.npu.device_of|是|-|
|[torch.cuda.get_device_capability](https://pytorch.org/docs/2.12/generated/torch.cuda.get_device_capability.html)|torch_npu.npu.get_device_capability|是|通过环境变量TORCH_NPU_DEVICE_CAPABILITY配置`torch_npu.npu.get_device_capability()`的返回值，仅用于兼容原生PyTorch，不代表NPU硬件实际能力|
|[torch.cuda.get_device_name](https://pytorch.org/docs/2.12/generated/torch.cuda.get_device_name.html)|torch_npu.npu.get_device_name|是|-|
|[torch.cuda.get_device_properties](https://pytorch.org/docs/2.12/generated/torch.cuda.get_device_properties.html)|torch_npu.npu.get_device_properties|是|仅支持name、total_memory、L2_cache_size、cube_core_num和vector_core_num属性，原CUDA上支持的其余属性均返回空字段|
|[torch.cuda.get_sync_debug_mode](https://pytorch.org/docs/2.12/generated/torch.cuda.get_sync_debug_mode.html)|torch_npu.npu.get_sync_debug_mode|是|-|
|[torch.cuda.init](https://pytorch.org/docs/2.12/generated/torch.cuda.init.html)|torch_npu.npu.init|是|-|
|[torch.cuda.ipc_collect](https://pytorch.org/docs/2.12/generated/torch.cuda.ipc_collect.html)|torch_npu.npu.ipc_collect|是|-|
|[torch.cuda.is_available](https://pytorch.org/docs/2.12/generated/torch.cuda.is_available.html)|torch_npu.npu.is_available|是|-|
|[torch.cuda.is_initialized](https://pytorch.org/docs/2.12/generated/torch.cuda.is_initialized.html)|torch_npu.npu.is_initialized|是|-|
|[torch.cuda.memory_usage](https://pytorch.org/docs/2.12/generated/torch.cuda.memory_usage.html)|-|否|-|
|[torch.cuda.set_device](https://pytorch.org/docs/2.12/generated/torch.cuda.set_device.html)|torch_npu.npu.set_device|是|-|
|[torch.cuda.set_stream](https://pytorch.org/docs/2.12/generated/torch.cuda.set_stream.html)|torch_npu.npu.set_stream|是|-|
|[torch.cuda.set_sync_debug_mode](https://pytorch.org/docs/2.12/generated/torch.cuda.set_sync_debug_mode.html)|torch_npu.npu.set_sync_debug_mode|是|-|
|[torch.cuda.stream](https://pytorch.org/docs/2.12/cuda.html#torch.cuda.stream)|torch_npu.npu.stream|是|-|
|[torch.cuda.synchronize](https://pytorch.org/docs/2.12/generated/torch.cuda.synchronize.html)|torch_npu.npu.synchronize|是|-|
|[torch.cuda.utilization](https://pytorch.org/docs/2.12/generated/torch.cuda.utilization.html)|torch_npu.npu.utilization|是|-|
|[torch.cuda.get_rng_state](https://pytorch.org/docs/2.12/generated/torch.cuda.get_rng_state.html)|torch_npu.npu.get_rng_state|是|-|
|[torch.cuda.set_rng_state](https://pytorch.org/docs/2.12/generated/torch.cuda.set_rng_state.html)|torch_npu.npu.set_rng_state|是|-|
|[torch.cuda.set_rng_state_all](https://pytorch.org/docs/2.12/generated/torch.cuda.set_rng_state_all.html)|torch_npu.npu.set_rng_state_all|是|-|
|[torch.cuda.manual_seed](https://pytorch.org/docs/2.12/generated/torch.cuda.manual_seed.html)|torch_npu.npu.manual_seed|是|-|
|[torch.cuda.manual_seed_all](https://pytorch.org/docs/2.12/generated/torch.cuda.manual_seed_all.html)|torch_npu.npu.manual_seed_all|是|-|
|[torch.cuda.seed](https://pytorch.org/docs/2.12/generated/torch.cuda.seed.html)|torch_npu.npu.seed|是|-|
|[torch.cuda.seed_all](https://pytorch.org/docs/2.12/generated/torch.cuda.seed_all.html)|torch_npu.npu.seed_all|是|-|
|[torch.cuda.initial_seed](https://pytorch.org/docs/2.12/generated/torch.cuda.initial_seed.html)|torch_npu.npu.initial_seed|是|-|
|[torch.cuda.comm.scatter](https://pytorch.org/docs/2.12/generated/torch.cuda.comm.scatter.html)|-|否|-|
|[torch.cuda.comm.gather](https://pytorch.org/docs/2.12/generated/torch.cuda.comm.gather.html)|-|否|-|
|[torch.cuda.Stream](https://pytorch.org/docs/2.12/cuda.html#torch.cuda.Stream)|torch_npu.npu.Stream|是|-|
|[torch.cuda.Stream.wait_stream](https://pytorch.org/docs/2.12/generated/torch.cuda.Stream_class.html#torch.cuda.Stream.wait_stream)|torch_npu.npu.Stream.wait_stream|是|-|
|[torch.cuda.Event](https://pytorch.org/docs/2.12/generated/torch.cuda.Event.html)|torch_npu.npu.Event|是|-|
|[torch.cuda.Event.elapsed_time](https://pytorch.org/docs/2.12/generated/torch.cuda.Event.html#torch.cuda.Event.elapsed_time)|torch_npu.npu.Event.elapsed_time|是|-|
|[torch.cuda.Event.from_ipc_handle](https://pytorch.org/docs/2.12/generated/torch.cuda.Event.html#torch.cuda.Event.from_ipc_handle)|torch_npu.npu.Event.from_ipc_handle|是|-|
|[torch.cuda.Event.ipc_handle](https://pytorch.org/docs/2.12/generated/torch.cuda.Event.html#torch.cuda.Event.ipc_handle)|torch_npu.npu.Event.ipc_handle|是|-|
|[torch.cuda.Event.query](https://pytorch.org/docs/2.12/generated/torch.cuda.Event.html#torch.cuda.Event.query)|torch_npu.npu.Event.query|是|-|
|[torch.cuda.Event.wait](https://pytorch.org/docs/2.12/generated/torch.cuda.Event.html#torch.cuda.Event.wait)|torch_npu.npu.Event.wait|是|-|
|[torch.cuda.is_current_stream_capturing](https://pytorch.org/docs/2.12/generated/torch.cuda.is_current_stream_capturing.html)|torch.npu.is_current_stream_capturing|是|-|
|[torch.cuda.graph_pool_handle](https://pytorch.org/docs/2.12/generated/torch.cuda.graph_pool_handle.html)|torch.npu.graph_pool_handle|是|当前仅支持推理场景，不支持训练场景|
|[torch.cuda.CUDAGraph](https://pytorch.org/docs/2.12/generated/torch.cuda.CUDAGraph.html)|torch.npu.NPUGraph|是|当前仅支持推理场景，不支持训练场景|
|[torch.cuda.CUDAGraph.capture_begin](https://pytorch.org/docs/2.12/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.capture_begin)|torch.npu.NPUGraph.capture_begin|是|当前仅支持推理场景，不支持训练场景|
|[torch.cuda.CUDAGraph.capture_end](https://pytorch.org/docs/2.12/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.capture_end)|torch.npu.NPUGraph.capture_end|是|当前仅支持推理场景，不支持训练场景|
|[torch.cuda.CUDAGraph.debug_dump](https://pytorch.org/docs/2.12/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.debug_dump)|torch.npu.NPUGraph.debug_dump|是|当前仅支持推理场景，不支持训练场景<br>导出文件内容为json格式|
|[torch.cuda.CUDAGraph.pool](https://pytorch.org/docs/2.12/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.pool)|torch.npu.NPUGraph.pool|是|当前仅支持推理场景，不支持训练场景|
|[torch.cuda.CUDAGraph.replay](https://pytorch.org/docs/2.12/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.replay)|torch.npu.NPUGraph.replay|是|当前仅支持推理场景，不支持训练场景|
|[torch.cuda.CUDAGraph.reset](https://pytorch.org/docs/2.12/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.reset)|torch.npu.NPUGraph.reset|是|当前仅支持推理场景，不支持训练场景|
|[torch.cuda.graph](https://pytorch.org/docs/2.12/generated/torch.cuda.graph.html)|torch.npu.graph|是|当前仅支持推理场景，不支持训练场景|
|[torch.cuda.make_graphed_callables](https://pytorch.org/docs/2.12/generated/torch.cuda.make_graphed_callables.html)|torch.npu.make_graphed_callables|是|当前仅支持推理场景，不支持训练场景|
|[torch.cuda.empty_cache](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.empty_cache.html)|torch_npu.npu.empty_cache|是|-|
|[torch.cuda.mem_get_info](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.mem_get_info.html)|torch_npu.npu.mem_get_info|是|-|
|[torch.cuda.memory_stats](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.memory_stats.html)|torch_npu.npu.memory_stats|是|-|
|[torch.cuda.memory_summary](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.memory_summary.html)|torch_npu.npu.memory_summary|是|-|
|[torch.cuda.memory_allocated](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.memory_allocated.html)|torch_npu.npu.memory_allocated|是|-|
|[torch.cuda.max_memory_allocated](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.max_memory_allocated.html)|torch_npu.npu.max_memory_allocated|是|-|
|[torch.cuda.reset_max_memory_allocated](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.reset_max_memory_allocated.html)|torch_npu.npu.reset_max_memory_allocated|是|-|
|[torch.cuda.memory_reserved](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.memory_reserved.html)|torch_npu.npu.memory_reserved|是|-|
|[torch.cuda.max_memory_reserved](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.max_memory_reserved.html)|torch_npu.npu.max_memory_reserved|是|-|
|[torch.cuda.set_per_process_memory_fraction](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.set_per_process_memory_fraction.html)|torch_npu.npu.set_per_process_memory_fraction|是|-|
|[torch.cuda.memory_cached](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.memory_cached.html)|torch_npu.npu.memory_cached|是|-|
|[torch.cuda.max_memory_cached](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.max_memory_cached.html)|torch_npu.npu.max_memory_cached|是|-|
|[torch.cuda.reset_max_memory_cached](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.reset_max_memory_cached.html)|torch_npu.npu.reset_max_memory_cached|是|-|
|[torch.cuda.reset_peak_memory_stats](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.reset_peak_memory_stats.html)|torch_npu.npu.reset_peak_memory_stats|是|-|
|[torch.cuda.caching_allocator_alloc](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.caching_allocator_alloc.html)|torch_npu.npu.caching_allocator_alloc|是|-|
|[torch.cuda.caching_allocator_delete](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.caching_allocator_delete.html)|torch_npu.npu.caching_allocator_delete|是|-|
|[torch.cuda.get_allocator_backend](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.get_allocator_backend.html)|torch_npu.npu.get_allocator_backend|是|-|
|[torch.cuda.CUDAPluggableAllocator](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.CUDAPluggableAllocator.html)|torch_npu.npu.NPUPluggableAllocator|是|该接口涉及高危操作，使用请参考《Ascend Extension for PyTorch 自定义 API参考》中的“[torch_npu.npu.NPUPluggableAllocator](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-npu/torch-npu-npu-NPUPluggableAllocator.md)”章节。|
|[torch.cuda.change_current_allocator](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.change_current_allocator.html)|torch_npu.npu.change_current_allocator|是|该接口涉及高危操作，使用请参考《Ascend Extension for PyTorch 自定义 API参考》中的“[torch_npu.npu.change_current_allocator](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-npu/torch-npu-npu-change_current_allocator.md)”章节。|
|[torch.cuda._sanitizer.enable_cuda_sanitizer](https://pytorch.org/docs/2.12/cuda._sanitizer.html#torch.cuda._sanitizer.enable_cuda_sanitizer)|torch_npu.npu._sanitizer.enable_npu_sanitizer|是|-|
|[torch.cuda.reset_accumulated_host_memory_stats](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.reset_accumulated_host_memory_stats.html)| torch_npu.npu.reset_accumulated_host_memory_stats |是|-|
|[torch.cuda.reset_peak_host_memory_stats](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.reset_peak_host_memory_stats.html)| torch_npu.npu.reset_peak_host_memory_stats          |是|-|
|[torch.cuda.host_memory_stats_as_nested_dict](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.host_memory_stats_as_nested_dict.html)| torch_npu.npu.host_memory_stats_as_nested_dict             |是|-|
|[torch.cuda.host_memory_stats](https://pytorch.org/docs/2.12/generated/torch.cuda.memory.host_memory_stats.html)| torch_npu.npu.host_memory_stats             |是|-|
