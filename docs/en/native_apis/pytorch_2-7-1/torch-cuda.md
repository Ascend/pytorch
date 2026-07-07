# torch.cuda

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T01:05:17.440Z pushedAt=2026-06-15T02:04:36.496Z -->

> [!NOTE]  
> If the "support status" for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.<br>
> When using supported CUDA interfaces, replace "cuda" in the API name with the NPU-adapted name: replace `torch.cuda.` with `torch_npu.npu.` or `torch.npu.`. Both `torch_npu.npu.` and `torch.npu.` calling methods have the same functionality. Examples are as follows:
>
> - `torch.cuda.current_device` --> `torch_npu.npu.current_device`
> - `torch.cuda.current_device` --> `torch.npu.current_device`

|API Name|NPU-Adapted Name|Supported|Restrictions and Notes|
|--|--|--|--|
|[torch.cuda.StreamContext](https://pytorch.org/docs/2.7/generated/torch.cuda.StreamContext.html)|torch.npu.StreamContext|Yes|-|
|[torch.cuda.can_device_access_peer](https://pytorch.org/docs/2.7/generated/torch.cuda.can_device_access_peer.html)|torch_npu.npu.can_device_access_peer|Yes|-|
|[torch.cuda.current_blas_handle](https://pytorch.org/docs/2.7/generated/torch.cuda.current_blas_handle.html)|torch_npu.npu.current_blas_handle|Yes|-|
|[torch.cuda.current_device](https://pytorch.org/docs/2.7/generated/torch.cuda.current_device.html)|torch_npu.npu.current_device|Yes|-|
|[torch.cuda.current_stream](https://pytorch.org/docs/2.7/generated/torch.cuda.current_stream.html)|torch_npu.npu.current_stream|Yes|When no device is set, calling this interface implicitly initializes the current device (default device 0)|
|[torch.cuda.default_stream](https://pytorch.org/docs/2.7/generated/torch.cuda.default_stream.html)|torch_npu.npu.default_stream|Yes|When no device is set, calling this interface implicitly initializes the current device (default device 0)|
|[torch.cuda.device](https://pytorch.org/docs/2.7/generated/torch.cuda.device.html)|torch_npu.npu.device|Yes|-|
|[torch.cuda.device_count](https://pytorch.org/docs/2.7/generated/torch.cuda.device_count.html)|torch_npu.npu.device_count|Yes|-|
|[torch.cuda.device_of](https://pytorch.org/docs/2.7/generated/torch.cuda.device_of.html)|torch_npu.npu.device_of|Yes|-|
|[torch.cuda.get_device_capability](https://pytorch.org/docs/2.7/generated/torch.cuda.get_device_capability.html)|torch_npu.npu.get_device_capability|Yes|The return value of `torch_npu.npu.get_device_capability()` is configured via the environment variable TORCH_NPU_DEVICE_CAPABILITY, which is only used for compatibility with native PyTorch and does not represent the actual capabilities of the NPU hardware|
|[torch.cuda.get_device_name](https://pytorch.org/docs/2.7/generated/torch.cuda.get_device_name.html)|torch_npu.npu.get_device_name|Yes|-|
|[torch.cuda.get_device_properties](https://pytorch.org/docs/2.7/generated/torch.cuda.get_device_properties.html)|torch_npu.npu.get_device_properties|Yes|Only the name, total_memory, L2_cache_size, cube_core_num, and vector_core_num properties are supported. All other properties supported on native CUDA return empty fields|
|[torch.cuda.get_sync_debug_mode](https://pytorch.org/docs/2.7/generated/torch.cuda.get_sync_debug_mode.html)|torch_npu.npu.get_sync_debug_mode|Yes|-|
|[torch.cuda.init](https://pytorch.org/docs/2.7/generated/torch.cuda.init.html)|torch_npu.npu.init|Yes|-|
|[torch.cuda.ipc_collect](https://pytorch.org/docs/2.7/generated/torch.cuda.ipc_collect.html)|torch_npu.npu.ipc_collect|Yes|-|
|[torch.cuda.is_available](https://pytorch.org/docs/2.7/generated/torch.cuda.is_available.html)|torch_npu.npu.is_available|Yes|-|
|[torch.cuda.is_initialized](https://pytorch.org/docs/2.7/generated/torch.cuda.is_initialized.html)|torch_npu.npu.is_initialized|Yes|-|
|[torch.cuda.memory_usage](https://pytorch.org/docs/2.7/generated/torch.cuda.memory_usage.html)|-|No|-|
|[torch.cuda.set_device](https://pytorch.org/docs/2.7/generated/torch.cuda.set_device.html)|torch_npu.npu.set_device|Yes|-|
|[torch.cuda.set_stream](https://pytorch.org/docs/2.7/generated/torch.cuda.set_stream.html)|torch_npu.npu.set_stream|Yes|-|
|[torch.cuda.set_sync_debug_mode](https://pytorch.org/docs/2.7/generated/torch.cuda.set_sync_debug_mode.html)|torch_npu.npu.set_sync_debug_mode|Yes|-|
|[torch.cuda.stream](https://pytorch.org/docs/2.7/generated/torch.cuda.stream.html)|torch_npu.npu.stream|Yes|-|
|[torch.cuda.synchronize](https://pytorch.org/docs/2.7/generated/torch.cuda.synchronize.html)|torch_npu.npu.synchronize|Yes|-|
|[torch.cuda.utilization](https://pytorch.org/docs/2.7/generated/torch.cuda.utilization.html)|torch_npu.npu.utilization|Yes|-|
|[torch.cuda.get_rng_state](https://pytorch.org/docs/2.7/generated/torch.cuda.get_rng_state.html)|torch_npu.npu.get_rng_state|Yes|-|
|[torch.cuda.set_rng_state](https://pytorch.org/docs/2.7/generated/torch.cuda.set_rng_state.html)|torch_npu.npu.set_rng_state|Yes|-|
|[torch.cuda.set_rng_state_all](https://pytorch.org/docs/2.7/generated/torch.cuda.set_rng_state_all.html)|torch_npu.npu.set_rng_state_all|Yes|-|
|[torch.cuda.manual_seed](https://pytorch.org/docs/2.7/generated/torch.cuda.manual_seed.html)|torch_npu.npu.manual_seed|Yes|-|
|[torch.cuda.manual_seed_all](https://pytorch.org/docs/2.7/generated/torch.cuda.manual_seed_all.html)|torch_npu.npu.manual_seed_all|Yes|-|
|[torch.cuda.seed](https://pytorch.org/docs/2.7/generated/torch.cuda.seed.html)|torch_npu.npu.seed|Yes|-|
|[torch.cuda.seed_all](https://pytorch.org/docs/2.7/generated/torch.cuda.seed_all.html)|torch_npu.npu.seed_all|Yes|-|
|[torch.cuda.initial_seed](https://pytorch.org/docs/2.7/generated/torch.cuda.initial_seed.html)|torch_npu.npu.initial_seed|Yes|-|
|[torch.cuda.comm.scatter](https://pytorch.org/docs/2.7/generated/torch.cuda.comm.scatter.html)|-|No|-|
|[torch.cuda.comm.gather](https://pytorch.org/docs/2.7/generated/torch.cuda.comm.gather.html)|-|No|-|
|[torch.cuda.Stream](https://pytorch.org/docs/2.7/generated/torch.cuda.Stream.html)|torch_npu.npu.Stream|Yes|-|
|[torch.cuda.Stream.wait_stream](https://pytorch.org/docs/2.7/generated/torch.cuda.Stream.html#torch.cuda.Stream.wait_stream)|torch_npu.npu.Stream.wait_stream|Yes|-|
|[torch.cuda.Event](https://pytorch.org/docs/2.7/generated/torch.cuda.Event.html)|torch_npu.npu.Event|Yes|-|
|[torch.cuda.Event.elapsed_time](https://pytorch.org/docs/2.7/generated/torch.cuda.Event.html#torch.cuda.Event.elapsed_time)|torch_npu.npu.Event.elapsed_time|Yes|-|
|[torch.cuda.Event.from_ipc_handle](https://pytorch.org/docs/2.7/generated/torch.cuda.Event.html#torch.cuda.Event.from_ipc_handle)|torch_npu.npu.Event.from_ipc_handle|Yes|-|
|[torch.cuda.Event.ipc_handle](https://pytorch.org/docs/2.7/generated/torch.cuda.Event.html#torch.cuda.Event.ipc_handle)|torch_npu.npu.Event.ipc_handle|Yes|-|
|[torch.cuda.Event.query](https://pytorch.org/docs/2.7/generated/torch.cuda.Event.html#torch.cuda.Event.query)|torch_npu.npu.Event.query|Yes|-|
|[torch.cuda.Event.wait](https://pytorch.org/docs/2.7/generated/torch.cuda.Event.html#torch.cuda.Event.wait)|torch_npu.npu.Event.wait|Yes|-|
|[torch.cuda.is_current_stream_capturing](https://pytorch.org/docs/2.7/generated/torch.cuda.is_current_stream_capturing.html)|torch.npu.is_current_stream_capturing|Yes|-|
|[torch.cuda.graph_pool_handle](https://pytorch.org/docs/2.7/generated/torch.cuda.graph_pool_handle.html)|torch.npu.graph_pool_handle|Yes|currently supports inference scenarios only, does not support training scenarios|
|[torch.cuda.CUDAGraph](https://pytorch.org/docs/2.7/generated/torch.cuda.CUDAGraph.html)|torch.npu.NPUGraph|Yes|currently supports inference scenarios only, does not support training scenarios|
|[torch.cuda.CUDAGraph.capture_begin](https://pytorch.org/docs/2.7/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.capture_begin)|torch.npu.NPUGraph.capture_begin|Yes|currently supports inference scenarios only, does not support training scenarios|
|[torch.cuda.CUDAGraph.capture_end](https://pytorch.org/docs/2.7/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.capture_end)|torch.npu.NPUGraph.capture_end|Yes|currently supports inference scenarios only, does not support training scenarios|
|[torch.cuda.CUDAGraph.debug_dump](https://pytorch.org/docs/2.7/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.debug_dump)|torch.npu.NPUGraph.debug_dump|Yes|currently supports inference scenarios only, does not support training scenarios<br>The exported file content is in JSON format|
|[torch.cuda.CUDAGraph.pool](https://pytorch.org/docs/2.7/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.pool)|torch.npu.NPUGraph.pool|Yes|currently supports inference scenarios only, does not support training scenarios|
|[torch.cuda.CUDAGraph.replay](https://pytorch.org/docs/2.7/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.replay)|torch.npu.NPUGraph.replay|Yes|currently supports inference scenarios only, does not support training scenarios|
|[torch.cuda.CUDAGraph.reset](https://pytorch.org/docs/2.7/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.reset)|torch.npu.NPUGraph.reset|Yes|currently supports inference scenarios only, does not support training scenarios|
|[torch.cuda.graph](https://pytorch.org/docs/2.7/generated/torch.cuda.graph.html)|torch.npu.graph|Yes|currently supports inference scenarios only, does not support training scenarios|
|[torch.cuda.make_graphed_callables](https://pytorch.org/docs/2.7/generated/torch.cuda.make_graphed_callables.html)|torch.npu.make_graphed_callables|Yes|currently supports inference scenarios only, does not support training scenarios|
|[torch.cuda.empty_cache](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.empty_cache)|torch_npu.npu.empty_cache|Yes|-|
|[torch.cuda.mem_get_info](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.mem_get_info)|torch_npu.npu.mem_get_info|Yes|-|
|[torch.cuda.memory_stats](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.memory_stats)|torch_npu.npu.memory_stats|Yes|-|
|[torch.cuda.memory_summary](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.memory_summary)|torch_npu.npu.memory_summary|Yes|-|
|[torch.cuda.memory_allocated](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.memory_allocated)|torch_npu.npu.memory_allocated|Yes|-|
|[torch.cuda.max_memory_allocated](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.max_memory_allocated)|torch_npu.npu.max_memory_allocated|Yes|-|
|[torch.cuda.reset_max_memory_allocated](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.reset_max_memory_allocated)|torch_npu.npu.reset_max_memory_allocated|Yes|-|
|[torch.cuda.memory_reserved](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.memory_reserved)|torch_npu.npu.memory_reserved|Yes|-|
|[torch.cuda.max_memory_reserved](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.max_memory_reserved)|torch_npu.npu.max_memory_reserved|Yes|-|
|[torch.cuda.set_per_process_memory_fraction](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.set_per_process_memory_fraction)|torch_npu.npu.set_per_process_memory_fraction|Yes|-|
|[torch.cuda.memory_cached](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.memory_cached)|torch_npu.npu.memory_cached|Yes|-|
|[torch.cuda.max_memory_cached](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.max_memory_cached)|torch_npu.npu.max_memory_cached|Yes|-|
|[torch.cuda.reset_max_memory_cached](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.reset_max_memory_cached)|torch_npu.npu.reset_max_memory_cached|Yes|-|
|[torch.cuda.reset_peak_memory_stats](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.reset_peak_memory_stats)|torch_npu.npu.reset_peak_memory_stats|Yes|-|
|[torch.cuda.caching_allocator_alloc](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.caching_allocator_alloc)|torch_npu.npu.caching_allocator_alloc|Yes|-|
|[torch.cuda.caching_allocator_delete](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.caching_allocator_delete)|torch_npu.npu.caching_allocator_delete|Yes|-|
|[torch.cuda.get_allocator_backend](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.get_allocator_backend)|torch_npu.npu.get_allocator_backend|Yes|-|
|[torch.cuda.CUDAPluggableAllocator](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.CUDAPluggableAllocator)|torch_npu.npu.NPUPluggableAllocator|Yes|This interface involves high-risk operations. For usage, refer to the "[torch_npu.npu.NPUPluggableAllocator](https://gitcode.com/Ascend/op-plugin/blob/26.0.0/docs/en/custom_APIs/torch_npu-npu/torch-npu-npu-NPUPluggableAllocator.md)" section in the *Ascend Extension for PyTorch Custom API Reference*.|
|[torch.cuda.change_current_allocator](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.change_current_allocator)|torch_npu.npu.change_current_allocator|Yes|This interface involves high-risk operations. For usage, refer to the "[torch_npu.npu.change_current_allocator](https://gitcode.com/Ascend/op-plugin/blob/26.0.0/docs/en/custom_APIs/torch_npu-npu/torch-npu-npu-change_current_allocator.md)" section in the *Ascend Extension for PyTorch Custom API Reference*.|
|[torch.cuda._sanitizer.enable_cuda_sanitizer](https://pytorch.org/docs/2.7/cuda._sanitizer.html#torch.cuda._sanitizer.enable_cuda_sanitizer)|torch_npu.npu._sanitizer.enable_npu_sanitizer|Yes|-|
|[torch.cuda.reset_accumulated_host_memory_stats](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.reset_accumulated_host_memory_stats)| torch_npu.npu.reset_accumulated_host_memory_stats |Yes|-|
|[torch.cuda.reset_peak_host_memory_stats](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.reset_peak_host_memory_stats)| torch_npu.npu.reset_peak_host_memory_stats          |Yes|-|
|[torch.cuda.host_memory_stats_as_nested_dict](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.host_memory_stats_as_nested_dict)| torch_npu.npu.host_memory_stats_as_nested_dict             |Yes|-|
|[torch.cuda.host_memory_stats](https://pytorch.org/docs/2.7/cuda.html#torch.cuda.host_memory_stats)| torch_npu.npu.host_memory_stats             |Yes|-|
