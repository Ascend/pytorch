# torch.cuda

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.11/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.11/cuda.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Random Number Generator](#random-number-generator)
- [Communication collectives](#communication-collectives)
- [Streams and events](#streams-and-events)
- [Graphs (beta)](#graphs-beta)
- [Memory management](#memory-management)

</div>

<div style="display:none;">

## &#8203;torch.cuda

</div>

### <code><i>class</i></code> torch.cuda.StreamContext

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch.npu.StreamContext

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.can_device_access_peer

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.can_device_access_peer

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.current_blas_handle

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.current_blas_handle

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.current_stream

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.current_stream

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: When `device` is not set, calling this API implicitly initializes the current `device` (defaults to device 0)

</div>

### torch.cuda.default_stream

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.default_stream

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: When `device` is not set, calling this API implicitly initializes the current `device` (defaults to device 0)

</div>

### torch.cuda.device_count

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.device_count

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.device_of

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.device_of

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.get_device_capability

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.get_device_capability

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: The return value of `torch_npu.npu.get_device_capability()` is configured through the environment variable TORCH_NPU_DEVICE_CAPABILITY. It is only used for compatibility with native PyTorch and does not represent the actual capability of the NPU hardware

</div>

### torch.cuda.get_device_name

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.get_device_name

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.get_device_properties

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.get_device_properties

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: Only the name, total_memory, L2_cache_size, cube_core_num, and vector_core_num attributes are supported; the remaining attributes supported on the original CUDA return empty fields

</div>

### torch.cuda.get_sync_debug_mode

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.get_sync_debug_mode

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.cuda.init

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.init

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.ipc_collect

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.ipc_collect

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.is_available

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.is_available

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.is_initialized

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.is_initialized

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.cuda.memory_usage

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.cuda.set_device

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.set_device

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.current_device

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.current_device

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.set_stream

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.set_stream

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.set_sync_debug_mode

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.set_sync_debug_mode

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.cuda.stream

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.stream

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.synchronize

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.synchronize

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.utilization

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.utilization

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda._sanitizer.enable_cuda_sanitizer

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu._sanitizer.enable_npu_sanitizer

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

## Random Number Generator

### torch.cuda.get_rng_state

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.get_rng_state

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.set_rng_state

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.set_rng_state

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.set_rng_state_all

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.set_rng_state_all

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.manual_seed

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.manual_seed

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.manual_seed_all

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.manual_seed_all

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.seed

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.seed

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.seed_all

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.seed_all

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.initial_seed

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.initial_seed

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

## Communication collectives

### torch.cuda.comm.scatter

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.cuda.comm.gather

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Not Supported
- <term>Atlas A3 training products</term>: Not Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

## Streams and events

### <code><i>class</i></code> torch.cuda.Stream

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.Stream

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

> <font size="3">wait_stream()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

</div>

### <code><i>class</i></code> torch.cuda.Event

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.Event

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

> <font size="3">elapsed_time()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">from_ipc_handle()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">ipc_handle()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

> <font size="3">query()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

> <font size="3">wait()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

</div>

## Graphs (beta)

### torch.cuda.is_current_stream_capturing

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch.npu.is_current_stream_capturing

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.graph_pool_handle

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch.npu.graph_pool_handle

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: Currently only supports inference scenarios and does not support training scenarios

</div>

### <code><i>class</i></code> torch.cuda.CUDAGraph

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch.npu.NPUGraph

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: Currently only supports inference scenarios and does not support training scenarios

> <font size="3">capture_begin()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: Currently only supports inference scenarios and does not support training scenarios

</div>

> <font size="3">capture_end()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: Currently only supports inference scenarios and does not support training scenarios

</div>

> <font size="3">debug_dump()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**:

- Currently, only the inference scenario is supported; the training scenario is not supported.
- The exported file content is in JSON format.

</div>

> <font size="3">pool()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: Currently only supports inference scenarios and does not support training scenarios

</div>

> <font size="3">replay()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: Currently only supports inference scenarios and does not support training scenarios

</div>

> <font size="3">reset()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: Currently only supports inference scenarios and does not support training scenarios

</div>

</div>

### torch.cuda.graph

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch.npu.graph

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: Currently only supports inference scenarios and does not support training scenarios

</div>

### torch.cuda.make_graphed_callables

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch.npu.make_graphed_callables

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: Currently only supports inference scenarios and does not support training scenarios

</div>

## Memory management

### torch.cuda.device

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.device

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.empty_cache

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.empty_cache

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.mem_get_info

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.mem_get_info

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.memory_stats

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.memory_stats

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.memory_summary

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.memory_summary

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.cuda.memory_allocated

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.memory_allocated

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.max_memory_allocated

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.max_memory_allocated

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.reset_max_memory_allocated

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.reset_max_memory_allocated

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.memory_reserved

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.memory_reserved

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.max_memory_reserved

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.max_memory_reserved

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.set_per_process_memory_fraction

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.set_per_process_memory_fraction

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.cuda.memory_cached

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.memory_cached

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.max_memory_cached

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.max_memory_cached

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.reset_max_memory_cached

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.reset_max_memory_cached

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.reset_peak_memory_stats

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.reset_peak_memory_stats

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### torch.cuda.caching_allocator_alloc

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.caching_allocator_alloc

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.cuda.caching_allocator_delete

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.caching_allocator_delete

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.cuda.get_allocator_backend

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.get_allocator_backend

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

</div>

### <code><i>class</i></code> torch.cuda.CUDAPluggableAllocator

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.NPUPluggableAllocator

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: This API involves high-risk operations. For usage, refer to the "[torch_npu.npu.NPUPluggableAllocator](https://gitcode.com/Ascend/op-plugin/blob/master/docs/en/custom_APIs/torch_npu-npu/torch-npu-npu-NPUPluggableAllocator.md)" section in Custom APIs

</div>

### torch.cuda.change_current_allocator

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.change_current_allocator

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: This API involves high-risk operations. For usage, refer to the "[torch_npu.npu.change_current_allocator](https://gitcode.com/Ascend/op-plugin/blob/master/docs/en/custom_APIs/torch_npu-npu/torch-npu-npu-change_current_allocator.md)" section in Custom APIs

</div>

### torch.cuda.reset_accumulated_host_memory_stats

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.reset_accumulated_host_memory_stats

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.cuda.reset_peak_host_memory_stats

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.reset_peak_host_memory_stats

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.cuda.host_memory_stats_as_nested_dict

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.host_memory_stats_as_nested_dict

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>

### torch.cuda.host_memory_stats

<div style="margin-left: 2em">

**NPU-Adapted Name**: torch_npu.npu.host_memory_stats

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

</div>
