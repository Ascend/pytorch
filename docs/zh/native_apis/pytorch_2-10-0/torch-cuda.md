# torch.cuda

> [!NOTE]  
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>
> 在使用支持的cuda接口时，需要将API名称中的cuda替换为NPU形式才能使用：将torch.cuda.替换为torch_npu.npu.或torch.npu.。torch_npu.npu.和torch.npu.两种调用方式，功能一致。举例如下：
>
> `torch.cuda.current_device` --> `torch_npu.npu.current_device`<br>
> `torch.cuda.current_device` --> `torch.npu.current_device`

## 目录

- [base API](#base-api)
- [Memory management](#memory-management)
- [Random Number Generator](#random-number-generator)
- [Communication collectives](#communication-collectives)
- [Streams and events](#streams-and-events)
- [Graphs (beta)](#graphs-beta)

## base API

### _`class`_ torch.cuda.StreamContext

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.StreamContext](https://pytorch.org/docs/2.10/generated/torch.cuda.StreamContext.html)

**NPU 形式名称**：torch.npu.StreamContext

**是否支持**：是

</div>

### torch.cuda.can_device_access_peer

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.can_device_access_peer](https://pytorch.org/docs/2.10/generated/torch.cuda.can_device_access_peer.html)

**NPU 形式名称**：torch_npu.npu.can_device_access_peer

**是否支持**：是

</div>

### torch.cuda.current_blas_handle

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.current_blas_handle](https://pytorch.org/docs/2.10/generated/torch.cuda.current_blas_handle.html)

**NPU 形式名称**：torch_npu.npu.current_blas_handle

**是否支持**：是

</div>

### torch.cuda.current_stream

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.current_stream](https://pytorch.org/docs/2.10/generated/torch.cuda.current_stream.html)

**NPU 形式名称**：torch_npu.npu.current_stream

**是否支持**：是

**限制与说明**： 未设置device时，调用该接口会隐式地初始化当前device（默认0卡）

</div>

### torch.cuda.default_stream

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.default_stream](https://pytorch.org/docs/2.10/generated/torch.cuda.default_stream.html)

**NPU 形式名称**：torch_npu.npu.default_stream

**是否支持**：是

**限制与说明**： 未设置device时，调用该接口会隐式地初始化当前device（默认0卡）

</div>

### torch.cuda.device_count

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.device_count](https://pytorch.org/docs/2.10/generated/torch.cuda.device_count.html)

**NPU 形式名称**：torch_npu.npu.device_count

**是否支持**：是

</div>

### torch.cuda.device_of

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.device_of](https://pytorch.org/docs/2.10/generated/torch.cuda.device_of.html)

**NPU 形式名称**：torch_npu.npu.device_of

**是否支持**：是

</div>

### torch.cuda.get_device_capability

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.get_device_capability](https://pytorch.org/docs/2.10/generated/torch.cuda.get_device_capability.html)

**NPU 形式名称**：torch_npu.npu.get_device_capability

**是否支持**：是

**限制与说明**： 通过环境变量TORCH_NPU_DEVICE_CAPABILITY配置`torch_npu.npu.get_device_capability()`的返回值，仅用于兼容原生PyTorch，不代表NPU硬件实际能力

</div>

### torch.cuda.get_device_name

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.get_device_name](https://pytorch.org/docs/2.10/generated/torch.cuda.get_device_name.html)

**NPU 形式名称**：torch_npu.npu.get_device_name

**是否支持**：是

</div>

### torch.cuda.get_device_properties

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.get_device_properties](https://pytorch.org/docs/2.10/generated/torch.cuda.get_device_properties.html)

**NPU 形式名称**：torch_npu.npu.get_device_properties

**是否支持**：是

**限制与说明**： 仅支持name、total_memory、L2_cache_size、cube_core_num和vector_core_num属性，原CUDA上支持的其余属性均返回空字段

</div>

### torch.cuda.get_sync_debug_mode

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.get_sync_debug_mode](https://pytorch.org/docs/2.10/generated/torch.cuda.get_sync_debug_mode.html)

**NPU 形式名称**：torch_npu.npu.get_sync_debug_mode

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.init

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.init](https://pytorch.org/docs/2.10/generated/torch.cuda.init.html)

**NPU 形式名称**：torch_npu.npu.init

**是否支持**：是

</div>

### torch.cuda.ipc_collect

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.ipc_collect](https://pytorch.org/docs/2.10/generated/torch.cuda.ipc_collect.html)

**NPU 形式名称**：torch_npu.npu.ipc_collect

**是否支持**：是

</div>

### torch.cuda.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.is_available](https://pytorch.org/docs/2.10/generated/torch.cuda.is_available.html)

**NPU 形式名称**：torch_npu.npu.is_available

**是否支持**：是

</div>

### torch.cuda.is_initialized

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.is_initialized](https://pytorch.org/docs/2.10/generated/torch.cuda.is_initialized.html)

**NPU 形式名称**：torch_npu.npu.is_initialized

**是否支持**：是

</div>

### torch.cuda.memory_usage

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory_usage](https://pytorch.org/docs/2.10/generated/torch.cuda.memory_usage.html)

**是否支持**：否

</div>

### torch.cuda.set_device

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.set_device](https://pytorch.org/docs/2.10/generated/torch.cuda.set_device.html)

**NPU 形式名称**：torch_npu.npu.set_device

**是否支持**：是

</div>

### torch.cuda.set_stream

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.set_stream](https://pytorch.org/docs/2.10/generated/torch.cuda.set_stream.html)

**NPU 形式名称**：torch_npu.npu.set_stream

**是否支持**：是

</div>

### torch.cuda.set_sync_debug_mode

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.set_sync_debug_mode](https://pytorch.org/docs/2.10/generated/torch.cuda.set_sync_debug_mode.html)

**NPU 形式名称**：torch_npu.npu.set_sync_debug_mode

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.stream

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.stream](https://pytorch.org/docs/2.10/cuda.html#torch.cuda.stream)

**NPU 形式名称**：torch_npu.npu.stream

**是否支持**：是

</div>

### torch.cuda.synchronize

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.synchronize](https://pytorch.org/docs/2.10/generated/torch.cuda.synchronize.html)

**NPU 形式名称**：torch_npu.npu.synchronize

**是否支持**：是

</div>

### torch.cuda.utilization

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.utilization](https://pytorch.org/docs/2.10/generated/torch.cuda.utilization.html)

**NPU 形式名称**：torch_npu.npu.utilization

**是否支持**：是

</div>

### torch.cuda._sanitizer.enable_cuda_sanitizer

<div style="margin-left: 2em">

**原生文档**：[torch.cuda._sanitizer.enable_cuda_sanitizer](https://pytorch.org/docs/2.10/cuda._sanitizer.html#torch.cuda._sanitizer.enable_cuda_sanitizer)

**NPU 形式名称**：torch_npu.npu._sanitizer.enable_npu_sanitizer

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## Memory management

### torch.cuda.current_device

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.current_device](https://pytorch.org/docs/2.10/generated/torch.cuda.current_device.html)

**NPU 形式名称**：torch_npu.npu.current_device

**是否支持**：是

</div>

### torch.cuda.device

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.device](https://pytorch.org/docs/2.10/generated/torch.cuda.device.html)

**NPU 形式名称**：torch_npu.npu.device

**是否支持**：是

</div>

### torch.cuda.memory.empty_cache

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.empty_cache](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.empty_cache.html)

**NPU 形式名称**：torch_npu.npu.empty_cache

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.mem_get_info

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.mem_get_info](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.mem_get_info.html)

**NPU 形式名称**：torch_npu.npu.mem_get_info

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.memory_stats

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.memory_stats](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.memory_stats.html)

**NPU 形式名称**：torch_npu.npu.memory_stats

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.memory_summary

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.memory_summary](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.memory_summary.html)

**NPU 形式名称**：torch_npu.npu.memory_summary

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.memory_allocated

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.memory_allocated](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.memory_allocated.html)

**NPU 形式名称**：torch_npu.npu.memory_allocated

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.max_memory_allocated

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.max_memory_allocated](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.max_memory_allocated.html)

**NPU 形式名称**：torch_npu.npu.max_memory_allocated

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.reset_max_memory_allocated

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.reset_max_memory_allocated](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.reset_max_memory_allocated.html)

**NPU 形式名称**：torch_npu.npu.reset_max_memory_allocated

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.memory_reserved

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.memory_reserved](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.memory_reserved.html)

**NPU 形式名称**：torch_npu.npu.memory_reserved

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.max_memory_reserved

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.max_memory_reserved](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.max_memory_reserved.html)

**NPU 形式名称**：torch_npu.npu.max_memory_reserved

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.set_per_process_memory_fraction

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.set_per_process_memory_fraction](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.set_per_process_memory_fraction.html)

**NPU 形式名称**：torch_npu.npu.set_per_process_memory_fraction

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.memory_cached

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.memory_cached](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.memory_cached.html)

**NPU 形式名称**：torch_npu.npu.memory_cached

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.max_memory_cached

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.max_memory_cached](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.max_memory_cached.html)

**NPU 形式名称**：torch_npu.npu.max_memory_cached

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.reset_max_memory_cached

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.reset_max_memory_cached](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.reset_max_memory_cached.html)

**NPU 形式名称**：torch_npu.npu.reset_max_memory_cached

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.reset_peak_memory_stats

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.reset_peak_memory_stats](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.reset_peak_memory_stats.html)

**NPU 形式名称**：torch_npu.npu.reset_peak_memory_stats

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.caching_allocator_alloc

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.caching_allocator_alloc](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.caching_allocator_alloc.html)

**NPU 形式名称**：torch_npu.npu.caching_allocator_alloc

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.caching_allocator_delete

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.caching_allocator_delete](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.caching_allocator_delete.html)

**NPU 形式名称**：torch_npu.npu.caching_allocator_delete

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.get_allocator_backend

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.get_allocator_backend](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.get_allocator_backend.html)

**NPU 形式名称**：torch_npu.npu.get_allocator_backend

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.cuda.memory.CUDAPluggableAllocator

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.CUDAPluggableAllocator](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.CUDAPluggableAllocator.html)

**NPU 形式名称**：torch_npu.npu.NPUPluggableAllocator

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 该接口涉及高危操作，使用请参考《自定义API》中的“[torch_npu.npu.NPUPluggableAllocator](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/zh/custom_APIs/torch_npu-npu/torch-npu-npu-NPUPluggableAllocator.md)”章节。

</div>

### torch.cuda.memory.change_current_allocator

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.change_current_allocator](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.change_current_allocator.html)

**NPU 形式名称**：torch_npu.npu.change_current_allocator

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 该接口涉及高危操作，使用请参考《自定义API》中的“[torch_npu.npu.change_current_allocator](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/zh/custom_APIs/torch_npu-npu/torch-npu-npu-change_current_allocator.md)”章节。

</div>

### torch.cuda.memory.reset_accumulated_host_memory_stats

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.reset_accumulated_host_memory_stats](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.reset_accumulated_host_memory_stats.html)

**NPU 形式名称**：torch_npu.npu.reset_accumulated_host_memory_stats

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.reset_peak_host_memory_stats

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.reset_peak_host_memory_stats](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.reset_peak_host_memory_stats.html)

**NPU 形式名称**：torch_npu.npu.reset_peak_host_memory_stats

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.host_memory_stats_as_nested_dict

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.host_memory_stats_as_nested_dict](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.host_memory_stats_as_nested_dict.html)

**NPU 形式名称**：torch_npu.npu.host_memory_stats_as_nested_dict

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.cuda.memory.host_memory_stats

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.host_memory_stats](https://pytorch.org/docs/2.10/generated/torch.cuda.memory.host_memory_stats.html)

**NPU 形式名称**：torch_npu.npu.host_memory_stats

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## Random Number Generator

### torch.cuda.get_rng_state

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.get_rng_state](https://pytorch.org/docs/2.10/generated/torch.cuda.get_rng_state.html)

**NPU 形式名称**：torch_npu.npu.get_rng_state

**是否支持**：是

</div>

### torch.cuda.set_rng_state

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.set_rng_state](https://pytorch.org/docs/2.10/generated/torch.cuda.set_rng_state.html)

**NPU 形式名称**：torch_npu.npu.set_rng_state

**是否支持**：是

</div>

### torch.cuda.set_rng_state_all

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.set_rng_state_all](https://pytorch.org/docs/2.10/generated/torch.cuda.set_rng_state_all.html)

**NPU 形式名称**：torch_npu.npu.set_rng_state_all

**是否支持**：是

</div>

### torch.cuda.manual_seed

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.manual_seed](https://pytorch.org/docs/2.10/generated/torch.cuda.manual_seed.html)

**NPU 形式名称**：torch_npu.npu.manual_seed

**是否支持**：是

</div>

### torch.cuda.manual_seed_all

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.manual_seed_all](https://pytorch.org/docs/2.10/generated/torch.cuda.manual_seed_all.html)

**NPU 形式名称**：torch_npu.npu.manual_seed_all

**是否支持**：是

</div>

### torch.cuda.seed

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.seed](https://pytorch.org/docs/2.10/generated/torch.cuda.seed.html)

**NPU 形式名称**：torch_npu.npu.seed

**是否支持**：是

</div>

### torch.cuda.seed_all

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.seed_all](https://pytorch.org/docs/2.10/generated/torch.cuda.seed_all.html)

**NPU 形式名称**：torch_npu.npu.seed_all

**是否支持**：是

</div>

### torch.cuda.initial_seed

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.initial_seed](https://pytorch.org/docs/2.10/generated/torch.cuda.initial_seed.html)

**NPU 形式名称**：torch_npu.npu.initial_seed

**是否支持**：是

</div>

## Communication collectives

### torch.cuda.comm.scatter

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.comm.scatter](https://pytorch.org/docs/2.10/generated/torch.cuda.comm.scatter.html)

**是否支持**：否

</div>

### torch.cuda.comm.gather

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.comm.gather](https://pytorch.org/docs/2.10/generated/torch.cuda.comm.gather.html)

**是否支持**：否

</div>

## Streams and events

### _`class`_ torch.cuda.Stream

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Stream](https://pytorch.org/docs/2.10/generated/torch.cuda.Stream.html)

**NPU 形式名称**：torch_npu.npu.Stream

**是否支持**：是

> <font size="3">wait_stream()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Stream.wait_stream](https://pytorch.org/docs/2.10/generated/torch.cuda.Stream.html#torch.cuda.Stream.wait_stream)

**是否支持**：是

</div>

</div>

### _`class`_ torch.cuda.Event

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Event](https://pytorch.org/docs/2.10/generated/torch.cuda.Event.html)

**NPU 形式名称**：torch_npu.npu.Event

**是否支持**：是

> <font size="3">elapsed_time()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Event.elapsed_time](https://pytorch.org/docs/2.10/generated/torch.cuda.Event.html#torch.cuda.Event.elapsed_time)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">from_ipc_handle()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Event.from_ipc_handle](https://pytorch.org/docs/2.10/generated/torch.cuda.Event.html#torch.cuda.Event.from_ipc_handle)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">ipc_handle()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Event.ipc_handle](https://pytorch.org/docs/2.10/generated/torch.cuda.Event.html#torch.cuda.Event.ipc_handle)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">query()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Event.query](https://pytorch.org/docs/2.10/generated/torch.cuda.Event.html#torch.cuda.Event.query)

**是否支持**：是

</div>

> <font size="3">wait()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Event.wait](https://pytorch.org/docs/2.10/generated/torch.cuda.Event.html#torch.cuda.Event.wait)

**是否支持**：是

</div>

</div>

## Graphs (beta)

### torch.cuda.is_current_stream_capturing

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.is_current_stream_capturing](https://pytorch.org/docs/2.10/generated/torch.cuda.is_current_stream_capturing.html)

**NPU 形式名称**：torch.npu.is_current_stream_capturing

**是否支持**：是

</div>

### torch.cuda.graph_pool_handle

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.graph_pool_handle](https://pytorch.org/docs/2.10/generated/torch.cuda.graph_pool_handle.html)

**NPU 形式名称**：torch.npu.graph_pool_handle

**是否支持**：是

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

### _`class`_ torch.cuda.CUDAGraph

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph](https://pytorch.org/docs/2.10/generated/torch.cuda.CUDAGraph.html)

**NPU 形式名称**：torch.npu.NPUGraph

**是否支持**：是

**限制与说明**： 当前仅支持推理场景，不支持训练场景

> <font size="3">capture_begin()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph.capture_begin](https://pytorch.org/docs/2.10/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.capture_begin)

**是否支持**：是

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

> <font size="3">capture_end()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph.capture_end](https://pytorch.org/docs/2.10/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.capture_end)

**是否支持**：是

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

> <font size="3">debug_dump()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph.debug_dump](https://pytorch.org/docs/2.10/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.debug_dump)

**是否支持**：是

**限制与说明**：

- 当前仅支持推理场景，不支持训练场景
- 导出文件内容为json格式

</div>

> <font size="3">pool()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph.pool](https://pytorch.org/docs/2.10/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.pool)

**是否支持**：是

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

> <font size="3">replay()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph.replay](https://pytorch.org/docs/2.10/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.replay)

**是否支持**：是

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

> <font size="3">reset()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph.reset](https://pytorch.org/docs/2.10/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.reset)

**是否支持**：是

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

</div>

### torch.cuda.graph

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.graph](https://pytorch.org/docs/2.10/generated/torch.cuda.graph.html)

**NPU 形式名称**：torch.npu.graph

**是否支持**：是

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

### torch.cuda.make_graphed_callables

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.make_graphed_callables](https://pytorch.org/docs/2.10/generated/torch.cuda.make_graphed_callables.html)

**NPU 形式名称**：torch.npu.make_graphed_callables

**是否支持**：是

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>
