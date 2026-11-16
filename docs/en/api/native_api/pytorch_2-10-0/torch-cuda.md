# torch.cuda

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.10/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> When using the supported CUDA APIs, replace "cuda" in the API name with the NPU form: replace "torch.cuda." with "torch_npu.npu." or "torch.npu.".
>
>   `torch.cuda.current_device` --> `torch_npu.npu.current_device`<br>
>   `torch.cuda.current_device` --> `torch.npu.current_device`
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.10/cuda.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Random Number Generator](#random-number-generator)

- [Communication collectives](#communication-collectives)
- [Streams and events](#streams-and-events)
- [Graphs (beta)](#graphs-beta)
- [Memory management](#memory-management)
- [NVIDIA Tools Extension (NVTX)](#nvidia-tools-extension-nvtx)

- [Jiterator (beta)](#jiterator-beta)

- [TunableOp](#tunableop)

- [GPUDirect Storage (prototype)](#gpudirect-storage-prototype)

</div>

<div style="display:none;">

## &#8203;torch.cuda

</div>

### <code><i>class</i></code> torch.cuda.StreamContext

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.StreamContext](https://pytorch.org/docs/2.10/generated/torch.cuda.StreamContext.html)

**NPU-Adapted Name**: torch.npu.StreamContext

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id3 -->

</div>

### torch.cuda.can_device_access_peer

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.can_device_access_peer](https://pytorch.org/docs/2.10/generated/torch.cuda.can_device_access_peer.html)

**NPU-Adapted Name**: torch_npu.npu.can_device_access_peer

**Support Status**:

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id6 -->

</div>

### torch.cuda.current_blas_handle

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.current_blas_handle](https://pytorch.org/docs/2.10/generated/torch.cuda.current_blas_handle.html)

**NPU-Adapted Name**: torch_npu.npu.current_blas_handle

**Support Status**:

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id9 -->

</div>

### torch.cuda.current_device

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.current_device](https://pytorch.org/docs/2.10/generated/torch.cuda.current_device.html)

**NPU-Adapted Name**: torch_npu.npu.current_device

**Support Status**:

<!-- npu="910b" id70 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id72 -->

</div>

### torch.cuda.current_stream

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.current_stream](https://pytorch.org/docs/2.10/generated/torch.cuda.current_stream.html)

**NPU-Adapted Name**: torch_npu.npu.current_stream

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id12 -->

**Constraints & Notes**: When `device` is not set, calling this API implicitly initializes the current `device` (defaults to device 0)

</div>

### torch.cuda.default_stream

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.default_stream](https://pytorch.org/docs/2.10/generated/torch.cuda.default_stream.html)

**NPU-Adapted Name**: torch_npu.npu.default_stream

**Support Status**:

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id15 -->

**Constraints & Notes**: When `device` is not set, calling this API implicitly initializes the current `device` (defaults to device 0)

</div>

### torch.cuda.device_count

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.device_count](https://pytorch.org/docs/2.10/generated/torch.cuda.device_count.html)

**NPU-Adapted Name**: torch_npu.npu.device_count

**Support Status**:

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id18 -->

</div>

### torch.cuda.device_memory_used

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.device_memory_used](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.device_memory_used.html#torch.cuda.device_memory_used)

**Support Status**:

<!-- npu="910b" id251 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id251 -->
<!-- npu="A3" id252 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id252 -->
<!-- npu="950" id253 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id253 -->

</div>

### torch.cuda.device_of

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.device_of](https://pytorch.org/docs/2.10/generated/torch.cuda.device_of.html)

**NPU-Adapted Name**: torch_npu.npu.device_of

**Support Status**:

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id21 -->

</div>

### torch.cuda.get_device_capability

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.get_device_capability](https://pytorch.org/docs/2.10/generated/torch.cuda.get_device_capability.html)

**NPU-Adapted Name**: torch_npu.npu.get_device_capability

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id24 -->

**Constraints & Notes**: The return value of `torch_npu.npu.get_device_capability()` is configured through the environment variable TORCH_NPU_DEVICE_CAPABILITY. It is only used for compatibility with native PyTorch and does not represent the actual capability of the NPU hardware

</div>

### torch.cuda.get_device_name

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.get_device_name](https://pytorch.org/docs/2.10/generated/torch.cuda.get_device_name.html)

**NPU-Adapted Name**: torch_npu.npu.get_device_name

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id27 -->

</div>

### torch.cuda.get_device_properties

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.get_device_properties](https://pytorch.org/docs/2.10/generated/torch.cuda.get_device_properties.html)

**NPU-Adapted Name**: torch_npu.npu.get_device_properties

**Support Status**:

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id30 -->

**Constraints & Notes**: Only the name, total_memory, L2_cache_size, cube_core_num, and vector_core_num attributes are supported; the remaining attributes supported on the original CUDA return empty fields

</div>

### torch.cuda.get\_stream\_from\_external

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.get\_stream\_from\_external](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.get_stream_from_external.html#torch.cuda.get_stream_from_external)

**Support Status**:

<!-- npu="910b" id299 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id299 -->
<!-- npu="A3" id300 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id300 -->
<!-- npu="950" id301 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id301 -->

</div>

### torch.cuda.get_sync_debug_mode

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.get_sync_debug_mode](https://pytorch.org/docs/2.10/generated/torch.cuda.get_sync_debug_mode.html)

**NPU-Adapted Name**: torch_npu.npu.get_sync_debug_mode

**Support Status**:

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id33 -->

</div>

### torch.cuda.init

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.init](https://pytorch.org/docs/2.10/generated/torch.cuda.init.html)

**NPU-Adapted Name**: torch_npu.npu.init

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id36 -->

</div>

### torch.cuda.ipc_collect

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.ipc_collect](https://pytorch.org/docs/2.10/generated/torch.cuda.ipc_collect.html)

**NPU-Adapted Name**: torch_npu.npu.ipc_collect

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id39 -->

</div>

### torch.cuda.is_available

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.is_available](https://pytorch.org/docs/2.10/generated/torch.cuda.is_available.html)

**NPU-Adapted Name**: torch_npu.npu.is_available

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id42 -->

</div>

### torch.cuda.is_bf16_supported

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.is_bf16_supported](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.is_bf16_supported.html#torch.cuda.is_bf16_supported)

**Support Status**:

<!-- npu="910b" id266 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id266 -->
<!-- npu="A3" id267 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id267 -->
<!-- npu="950" id268 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id268 -->

</div>

<!-- npu="910b,A3" id239 -->
### torch.cuda.is_initialized

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.is_initialized](https://pytorch.org/docs/2.10/generated/torch.cuda.is_initialized.html)

**NPU-Adapted Name**: torch_npu.npu.is_initialized

**Support Status**:
<!-- end id239 -->

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->

<!-- npu="910b,A3" id240 -->
</div>
<!-- end id240 -->

### torch.cuda.is\_tf32\_supported

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.is\_tf32\_supported](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.is_tf32_supported.html#torch.cuda.is_tf32_supported)

**Support Status**:

<!-- npu="910b" id302 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id302 -->
<!-- npu="A3" id303 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id303 -->
<!-- npu="950" id304 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id304 -->

</div>

### torch.cuda.memory_usage

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.memory_usage](https://pytorch.org/docs/2.10/generated/torch.cuda.memory_usage.html)

**Support Status**:

<!-- npu="910b" id46 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id48 -->

</div>

### torch.cuda.set_device

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.set_device](https://pytorch.org/docs/2.10/generated/torch.cuda.set_device.html)

**NPU-Adapted Name**: torch_npu.npu.set_device

**Support Status**:

<!-- npu="910b" id49 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id51 -->

</div>

### torch.cuda.set_stream

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.set_stream](https://pytorch.org/docs/2.10/generated/torch.cuda.set_stream.html)

**NPU-Adapted Name**: torch_npu.npu.set_stream

**Support Status**:

<!-- npu="910b" id52 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id54 -->

</div>

### torch.cuda.set_sync_debug_mode

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.set_sync_debug_mode](https://pytorch.org/docs/2.10/generated/torch.cuda.set_sync_debug_mode.html)

**NPU-Adapted Name**: torch_npu.npu.set_sync_debug_mode

**Support Status**:

<!-- npu="910b" id55 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id57 -->

</div>

### torch.cuda.stream

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.stream](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.stream_function.html#torch.cuda.stream)

**NPU-Adapted Name**: torch_npu.npu.stream

**Support Status**:

<!-- npu="910b" id58 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id60 -->

</div>

### torch.cuda.synchronize

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.synchronize](https://pytorch.org/docs/2.10/generated/torch.cuda.synchronize.html)

**NPU-Adapted Name**: torch_npu.npu.synchronize

**Support Status**:

<!-- npu="910b" id61 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id63 -->

</div>

### torch.cuda.utilization

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.utilization](https://pytorch.org/docs/2.10/generated/torch.cuda.utilization.html)

**NPU-Adapted Name**: torch_npu.npu.utilization

**Support Status**:

<!-- npu="910b" id64 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id66 -->

</div>

### torch.cuda.temperature

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.temperature](https://pytorch.org/docs/2.10/generated/torch.cuda.temperature.html)

**NPU-Adapted Name**: torch_npu.npu.temperature

**Support Status**:

<!-- npu="910b" id230 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id230 -->
<!-- npu="A3" id231 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id231 -->
<!-- npu="950" id232 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id232 -->

**Constraints & Notes**: You can use the `npu-smi info -t common -i <id>` command to view the Temperature(C) metric instead.

</div>

### torch.cuda.power_draw

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.power_draw](https://pytorch.org/docs/2.10/generated/torch.cuda.power_draw.html)

**NPU-Adapted Name**: torch_npu.npu.power_draw

**Support Status**:

<!-- npu="910b" id233 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id233 -->
<!-- npu="A3" id234 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id234 -->
<!-- npu="950" id235 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id235 -->

**Constraints & Notes**: You can use `npu-smi info -t common -i <id>` to view the NPU Real-time Power(W) metric.

</div>

<!-- npu="910b,A3" id241 -->
### torch.cuda._sanitizer.enable_cuda_sanitizer

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda._sanitizer.enable_cuda_sanitizer](https://pytorch.org/docs/2.10/cuda._sanitizer.html#torch.cuda._sanitizer.enable_cuda_sanitizer)

**NPU-Adapted Name**: torch_npu.npu._sanitizer.enable_npu_sanitizer

**Support Status**:
<!-- end id241 -->

<!-- npu="910b" id67 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id68 -->

<!-- npu="910b,A3" id242 -->
</div>
<!-- end id242 -->

### torch.cuda.clock_rate

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.clock_rate](https://pytorch.org/docs/2.10/generated/torch.cuda.clock_rate.html)

**NPU-Adapted Name**: torch_npu.npu.clock_rate

**Support Status**:

<!-- npu="910b" id236 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id236 -->
<!-- npu="A3" id237 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id237 -->
<!-- npu="950" id238 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id238 -->

**Constraints & Notes**: You can use the `npu-smi info -t common -i <id>` command to view the AI Core curFreq(MHZ) metric instead.

</div>

## Random Number Generator

### torch.cuda.get_rng_state

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.get_rng_state](https://pytorch.org/docs/2.10/generated/torch.cuda.get_rng_state.html)

**NPU-Adapted Name**: torch_npu.npu.get_rng_state

**Support Status**:

<!-- npu="910b" id73 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id75 -->

</div>

### torch.cuda.get_rng_state_all

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.get_rng_state_all](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.get_rng_state_all.html#torch.cuda.get_rng_state_all)

**Support Status**:

<!-- npu="910b" id254 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id254 -->
<!-- npu="A3" id255 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id255 -->
<!-- npu="950" id256 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id256 -->

</div>

### torch.cuda.set_rng_state

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.set_rng_state](https://pytorch.org/docs/2.10/generated/torch.cuda.set_rng_state.html)

**NPU-Adapted Name**: torch_npu.npu.set_rng_state

**Support Status**:

<!-- npu="910b" id76 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id78 -->

</div>

### torch.cuda.set_rng_state_all

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.set_rng_state_all](https://pytorch.org/docs/2.10/generated/torch.cuda.set_rng_state_all.html)

**NPU-Adapted Name**: torch_npu.npu.set_rng_state_all

**Support Status**:

<!-- npu="910b" id79 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id81 -->

</div>

### torch.cuda.manual_seed

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.manual_seed](https://pytorch.org/docs/2.10/generated/torch.cuda.manual_seed.html)

**NPU-Adapted Name**: torch_npu.npu.manual_seed

**Support Status**:

<!-- npu="910b" id82 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id84 -->

</div>

### torch.cuda.manual_seed_all

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.manual_seed_all](https://pytorch.org/docs/2.10/generated/torch.cuda.manual_seed_all.html)

**NPU-Adapted Name**: torch_npu.npu.manual_seed_all

**Support Status**:

<!-- npu="910b" id85 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id87 -->

</div>

### torch.cuda.seed

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.seed](https://pytorch.org/docs/2.10/generated/torch.cuda.seed.html)

**NPU-Adapted Name**: torch_npu.npu.seed

**Support Status**:

<!-- npu="910b" id88 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id90 -->

</div>

### torch.cuda.seed_all

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.seed_all](https://pytorch.org/docs/2.10/generated/torch.cuda.seed_all.html)

**NPU-Adapted Name**: torch_npu.npu.seed_all

**Support Status**:

<!-- npu="910b" id91 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id93 -->

</div>

### torch.cuda.initial_seed

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.initial_seed](https://pytorch.org/docs/2.10/generated/torch.cuda.initial_seed.html)

**NPU-Adapted Name**: torch_npu.npu.initial_seed

**Support Status**:

<!-- npu="910b" id94 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id96 -->

</div>

## Communication collectives

### torch.cuda.comm.broadcast

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.comm.broadcast](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.comm.broadcast.html#torch.cuda.comm.broadcast)

**Support Status**:

<!-- npu="910b" id305 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id305 -->
<!-- npu="A3" id306 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id306 -->
<!-- npu="950" id307 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id307 -->

</div>

### torch.cuda.comm.broadcast\_coalesced

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.comm.broadcast\_coalesced](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.comm.broadcast_coalesced.html#torch.cuda.comm.broadcast_coalesced)

**Support Status**:

<!-- npu="910b" id308 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id308 -->
<!-- npu="A3" id309 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id309 -->
<!-- npu="950" id310 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id310 -->

</div>

### torch.cuda.comm.reduce\_add

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.comm.reduce\_add](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.comm.reduce_add.html#torch.cuda.comm.reduce_add)

**Support Status**:

<!-- npu="910b" id311 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id311 -->
<!-- npu="A3" id312 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id312 -->
<!-- npu="950" id313 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id313 -->

</div>

## Streams and events

### <code><i>class</i></code> torch.cuda.Stream

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.Stream](https://pytorch.org/docs/2.10/generated/torch.cuda.Stream.html)

**NPU-Adapted Name**: torch_npu.npu.Stream

**Support Status**:

<!-- npu="910b" id103 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id105 -->

<!-- npu="910b,A3" id243 -->
> <font size="3">wait_stream()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.Stream.wait_stream](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.Stream_class.html#torch.cuda.Stream.wait_stream)

**Support Status**:
<!-- end id243 -->

<!-- npu="910b" id106 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id107 -->

<!-- npu="910b,A3" id244 -->
</div>
<!-- end id244 -->

> <font size="3">query()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.Stream.query](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.Stream_class.html#torch.cuda.Stream.query)

**Support Status**:

<!-- npu="910b" id287 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id287 -->
<!-- npu="A3" id288 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id288 -->
<!-- npu="950" id289 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id289 -->

</div>

> <font size="3">record_event()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.Stream.record_event](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.Stream_class.html#torch.cuda.Stream.record_event)

**Support Status**:

<!-- npu="910b" id290 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id290 -->
<!-- npu="A3" id291 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id291 -->
<!-- npu="950" id292 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id292 -->

</div>

> <font size="3">synchronize()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.Stream.synchronize](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.Stream_class.html#torch.cuda.Stream.synchronize)

**Support Status**:

<!-- npu="910b" id293 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id293 -->
<!-- npu="A3" id294 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id294 -->
<!-- npu="950" id295 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id295 -->

</div>

> <font size="3">wait_event()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.Stream.wait_event](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.Stream_class.html#torch.cuda.Stream.wait_event)

**Support Status**:

<!-- npu="910b" id296 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id296 -->
<!-- npu="A3" id297 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id297 -->
<!-- npu="950" id298 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id298 -->

</div>

</div>

### <code><i>class</i></code> torch.cuda.ExternalStream

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.ExternalStream](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.ExternalStream.html#torch.cuda.ExternalStream)

**Support Status**:

<!-- npu="910b" id263 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id263 -->
<!-- npu="A3" id264 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id264 -->
<!-- npu="950" id265 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id265 -->

</div>

### <code><i>class</i></code> torch.cuda.Event

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.Event](https://pytorch.org/docs/2.10/generated/torch.cuda.Event.html)

**NPU-Adapted Name**: torch_npu.npu.Event

**Support Status**:

<!-- npu="910b" id109 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id111 -->

> <font size="3">elapsed_time()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.Event.elapsed_time](https://pytorch.org/docs/2.10/generated/torch.cuda.Event.html#torch.cuda.Event.elapsed_time)

**Support Status**:

<!-- npu="910b" id112 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id114 -->

</div>

> <font size="3">from_ipc_handle()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.Event.from_ipc_handle](https://pytorch.org/docs/2.10/generated/torch.cuda.Event.html#torch.cuda.Event.from_ipc_handle)

**Support Status**:

<!-- npu="910b" id115 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id117 -->

</div>

> <font size="3">ipc_handle()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.Event.ipc_handle](https://pytorch.org/docs/2.10/generated/torch.cuda.Event.html#torch.cuda.Event.ipc_handle)

**Support Status**:

<!-- npu="910b" id118 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id120 -->

</div>

> <font size="3">query()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.Event.query](https://pytorch.org/docs/2.10/generated/torch.cuda.Event.html#torch.cuda.Event.query)

**Support Status**:

<!-- npu="910b" id121 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id123 -->

</div>

> <font size="3">record()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.Event.record](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.Event.html#torch.cuda.Event.record)

**Support Status**:

<!-- npu="910b" id281 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id281 -->
<!-- npu="A3" id282 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id282 -->
<!-- npu="950" id283 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id283 -->

</div>

> <font size="3">synchronize()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.Event.synchronize](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.Event.html#torch.cuda.Event.synchronize)

**Support Status**:

<!-- npu="910b" id284 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id284 -->
<!-- npu="A3" id285 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id285 -->
<!-- npu="950" id286 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id286 -->

</div>

> <font size="3">wait()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.Event.wait](https://pytorch.org/docs/2.10/generated/torch.cuda.Event.html#torch.cuda.Event.wait)

**Support Status**:

<!-- npu="910b" id124 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id126 -->

</div>

</div>

## Graphs (beta)

### torch.cuda.is_current_stream_capturing

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.is_current_stream_capturing](https://pytorch.org/docs/2.10/generated/torch.cuda.is_current_stream_capturing.html)

**NPU-Adapted Name**: torch.npu.is_current_stream_capturing

**Support Status**:

<!-- npu="910b" id127 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id129 -->

</div>

### torch.cuda.graph_pool_handle

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.graph_pool_handle](https://pytorch.org/docs/2.10/generated/torch.cuda.graph_pool_handle.html)

**NPU-Adapted Name**: torch.npu.graph_pool_handle

**Support Status**:

<!-- npu="910b" id130 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id132 -->

**Constraints & Notes**: Currently only supports inference scenarios and does not support training scenarios

</div>

### <code><i>class</i></code> torch.cuda.CUDAGraph

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.CUDAGraph](https://pytorch.org/docs/2.10/generated/torch.cuda.CUDAGraph.html)

**NPU-Adapted Name**: torch.npu.NPUGraph

**Support Status**:

<!-- npu="910b" id133 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id135 -->

**Constraints & Notes**: Currently supports only inference scenarios, not training scenarios.

> <font size="3">capture_begin()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.CUDAGraph.capture_begin](https://pytorch.org/docs/2.10/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.capture_begin)

**Support Status**:

<!-- npu="910b" id136 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id138 -->

**Constraints & Notes**: Currently supports only inference scenarios, not training scenarios.

</div>

> <font size="3">capture_end()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.CUDAGraph.capture_end](https://pytorch.org/docs/2.10/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.capture_end)

**Support Status**:

<!-- npu="910b" id139 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id141 -->

**Constraints & Notes**: Currently supports only inference scenarios, not training scenarios.

</div>

<!-- npu="910b,A3" id144 -->
> <font size="3">debug_dump()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.CUDAGraph.debug_dump](https://pytorch.org/docs/2.10/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.debug_dump)

**Support Status**:

<!-- npu="910b" id142 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id143 -->

**Constraints & Notes**:

- Currently supports only inference scenarios and does not support training scenarios.
- The exported file content is in JSON format.

</div>
<!-- end id144 -->

> <font size="3">enable_debug_mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.CUDAGraph.enable_debug_mode](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.enable_debug_mode)

**Support Status**:

<!-- npu="910b" id278 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id278 -->
<!-- npu="A3" id279 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id279 -->
<!-- npu="950" id280 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id280 -->

</div>

<!-- npu="910b,A3" id147 -->
> <font size="3">pool()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.CUDAGraph.pool](https://pytorch.org/docs/2.10/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.pool)

**Support Status**:

<!-- npu="910b" id145 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id146 -->

**Constraints & Notes**: Currently supports only inference scenarios, not training scenarios.

</div>
<!-- end id147 -->

> <font size="3">replay()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.CUDAGraph.replay](https://pytorch.org/docs/2.10/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.replay)

**Support Status**:

<!-- npu="910b" id148 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id150 -->

**Constraints & Notes**: Currently supports only inference scenarios, not training scenarios.

</div>

> <font size="3">reset()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.CUDAGraph.reset](https://pytorch.org/docs/2.10/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.reset)

**Support Status**:

<!-- npu="910b" id151 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id153 -->

**Constraints & Notes**: Currently supports only inference scenarios, not training scenarios.

</div>

</div>

### torch.cuda.graph

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.graph](https://pytorch.org/docs/2.10/generated/torch.cuda.graph.html)

**NPU-Adapted Name**: torch.npu.graph

**Support Status**:

<!-- npu="910b" id154 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id156 -->

**Constraints & Notes**: Currently only supports inference scenarios and does not support training scenarios

</div>

### torch.cuda.make_graphed_callables

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.make_graphed_callables](https://pytorch.org/docs/2.10/generated/torch.cuda.make_graphed_callables.html)

**NPU-Adapted Name**: torch.npu.make_graphed_callables

**Support Status**:

<!-- npu="910b" id157 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id159 -->

**Constraints & Notes**: Currently only supports inference scenarios and does not support training scenarios

</div>

## Memory management

### torch.cuda.device

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.device](https://pytorch.org/docs/2.10/generated/torch.cuda.device.html)

**NPU-Adapted Name**: torch_npu.npu.device

**Support Status**:

<!-- npu="910b" id160 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id162 -->

</div>

### torch.cuda.empty_cache

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.empty_cache](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.empty_cache.html#torch.cuda.memory.empty_cache)

**NPU-Adapted Name**: torch_npu.npu.empty_cache

**Support Status**:

<!-- npu="910b" id163 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id165 -->

</div>

### torch.cuda.get\_per\_process\_memory\_fraction

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.get\_per\_process\_memory\_fraction](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.get_per_process_memory_fraction.html#torch.cuda.memory.get_per_process_memory_fraction)

**Support Status**:

<!-- npu="910b" id314 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id314 -->
<!-- npu="A3" id315 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id315 -->
<!-- npu="950" id316 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id316 -->

</div>

### torch.cuda.mem_get_info

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.mem_get_info](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.mem_get_info.html#torch.cuda.memory.mem_get_info)

**NPU-Adapted Name**: torch_npu.npu.mem_get_info

**Support Status**:

<!-- npu="910b" id166 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id168 -->

</div>

### torch.cuda.memory_stats

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.memory_stats](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.memory_stats.html#torch.cuda.memory.memory_stats)

**NPU-Adapted Name**: torch_npu.npu.memory_stats

**Support Status**:

<!-- npu="910b" id169 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id171 -->

</div>

### torch.cuda.reset\_accumulated\_memory\_stats

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.reset\_accumulated\_memory\_stats](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.reset_accumulated_memory_stats.html#torch.cuda.memory.reset_accumulated_memory_stats)

**Support Status**:

<!-- npu="910b" id317 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id317 -->
<!-- npu="A3" id318 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id318 -->
<!-- npu="950" id319 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id319 -->

</div>

### torch.cuda.host_memory_stats

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.host_memory_stats](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.host_memory_stats.html#torch.cuda.memory.host_memory_stats)

**NPU-Adapted Name**: torch_npu.npu.host_memory_stats

**Support Status**:

<!-- npu="910b" id226 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id226 -->
<!-- npu="A3" id227 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id227 -->
<!-- npu="950" id228 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id228 -->

</div>

### torch.cuda.memory_summary

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.memory_summary](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.memory_summary.html#torch.cuda.memory.memory_summary)

**NPU-Adapted Name**: torch_npu.npu.memory_summary

**Support Status**:

<!-- npu="910b" id172 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id174 -->

</div>

### torch.cuda.memory\_snapshot

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.memory\_snapshot](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.memory_snapshot.html#torch.cuda.memory.memory_snapshot)

**Support Status**:

<!-- npu="910b" id320 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id320 -->
<!-- npu="A3" id321 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id321 -->
<!-- npu="950" id322 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id322 -->

</div>

### torch.cuda.memory_allocated

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.memory_allocated](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.memory_allocated.html#torch.cuda.memory.memory_allocated)

**NPU-Adapted Name**: torch_npu.npu.memory_allocated

**Support Status**:

<!-- npu="910b" id175 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id177 -->

</div>

### torch.cuda.max_memory_allocated

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.max_memory_allocated](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.max_memory_allocated.html#torch.cuda.memory.max_memory_allocated)

**NPU-Adapted Name**: torch_npu.npu.max_memory_allocated

**Support Status**:

<!-- npu="910b" id178 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id180 -->

</div>

### torch.cuda.reset_max_memory_allocated

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.reset_max_memory_allocated](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.reset_max_memory_allocated.html#torch.cuda.memory.reset_max_memory_allocated)

**NPU-Adapted Name**: torch_npu.npu.reset_max_memory_allocated

**Support Status**:

<!-- npu="910b" id181 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id183 -->

</div>

### torch.cuda.memory_reserved

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.memory_reserved](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.memory_reserved.html#torch.cuda.memory.memory_reserved)

**NPU-Adapted Name**: torch_npu.npu.memory_reserved

**Support Status**:

<!-- npu="910b" id184 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id186 -->

</div>

### torch.cuda.max_memory_reserved

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.max_memory_reserved](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.max_memory_reserved.html#torch.cuda.memory.max_memory_reserved)

**NPU-Adapted Name**: torch_npu.npu.max_memory_reserved

**Support Status**:

<!-- npu="910b" id187 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id189 -->

</div>

<!-- npu="910b,A3" id245 -->
### torch.cuda.set_per_process_memory_fraction

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.set_per_process_memory_fraction](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.set_per_process_memory_fraction.html#torch.cuda.memory.set_per_process_memory_fraction)

**NPU-Adapted Name**: torch_npu.npu.set_per_process_memory_fraction

**Support Status**:
<!-- end id245 -->

<!-- npu="910b" id190 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id191 -->

<!-- npu="910b,A3" id246 -->
</div>
<!-- end id246 -->

### torch.cuda.memory_cached

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.memory_cached](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.memory_cached.html#torch.cuda.memory.memory_cached)

**NPU-Adapted Name**: torch_npu.npu.memory_cached

**Support Status**:

<!-- npu="910b" id193 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id195 -->

</div>

### torch.cuda.max_memory_cached

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.max_memory_cached](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.max_memory_cached.html#torch.cuda.memory.max_memory_cached)

**NPU-Adapted Name**: torch_npu.npu.max_memory_cached

**Support Status**:

<!-- npu="910b" id196 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id197 -->
<!-- npu="950" id198 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id198 -->

</div>

### torch.cuda.reset_max_memory_cached

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.reset_max_memory_cached](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.reset_max_memory_cached.html#torch.cuda.memory.reset_max_memory_cached)

**NPU-Adapted Name**: torch_npu.npu.reset_max_memory_cached

**Support Status**:

<!-- npu="910b" id199 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id200 -->
<!-- npu="950" id201 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id201 -->

</div>

### torch.cuda.reset_peak_memory_stats

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.reset_peak_memory_stats](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.reset_peak_memory_stats.html#torch.cuda.memory.reset_peak_memory_stats)

**NPU-Adapted Name**: torch_npu.npu.reset_peak_memory_stats

**Support Status**:

<!-- npu="910b" id202 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id203 -->
<!-- npu="950" id204 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id204 -->

</div>

### torch.cuda.reset_peak_host_memory_stats

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.reset_peak_host_memory_stats](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.reset_peak_host_memory_stats.html#torch.cuda.memory.reset_peak_host_memory_stats)

**NPU-Adapted Name**: torch_npu.npu.reset_peak_host_memory_stats

**Support Status**:

<!-- npu="910b" id223 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id223 -->
<!-- npu="A3" id224 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id224 -->
<!-- npu="950" id225 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id225 -->

</div>

### torch.cuda.caching_allocator_alloc

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.caching_allocator_alloc](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.caching_allocator_alloc.html#torch.cuda.memory.caching_allocator_alloc)

**NPU-Adapted Name**: torch_npu.npu.caching_allocator_alloc

**Support Status**:

<!-- npu="910b" id205 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id206 -->
<!-- npu="950" id207 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id207 -->

</div>

### torch.cuda.caching_allocator_delete

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.caching_allocator_delete](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.caching_allocator_delete.html#torch.cuda.memory.caching_allocator_delete)

**NPU-Adapted Name**: torch_npu.npu.caching_allocator_delete

**Support Status**:

<!-- npu="910b" id208 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id209 -->
<!-- npu="950" id210 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id210 -->

</div>

### torch.cuda.get_allocator_backend

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.get_allocator_backend](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.get_allocator_backend.html#torch.cuda.memory.get_allocator_backend)

**NPU-Adapted Name**: torch_npu.npu.get_allocator_backend

**Support Status**:

<!-- npu="910b" id211 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id212 -->
<!-- npu="950" id213 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id213 -->

</div>

<!-- npu="910b,A3" id247 -->
### <code><i>class</i></code> torch.cuda.CUDAPluggableAllocator

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.CUDAPluggableAllocator](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.CUDAPluggableAllocator.html#torch.cuda.memory.CUDAPluggableAllocator)

**NPU-Adapted Name**: torch_npu.npu.NPUPluggableAllocator

**Support Status**:
<!-- end id247 -->

<!-- npu="910b" id214 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id215 -->

<!-- npu="910b,A3" id248 -->
**Constraints & Notes**: This API involves high-risk operations. For usage, refer to the "[torch_npu.npu.NPUPluggableAllocator](https://gitcode.com/Ascend/op-plugin/blob/26.2.0/docs/en/custom_APIs/torch_npu-npu/torch-npu-npu-NPUPluggableAllocator.md)" chapter.

</div>
<!-- end id248 -->

<!-- npu="910b,A3" id249 -->
### torch.cuda.change_current_allocator

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.change_current_allocator](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.change_current_allocator.html#torch.cuda.memory.change_current_allocator)

**NPU-Adapted Name**: torch_npu.npu.change_current_allocator

**Support Status**:
<!-- end id249 -->

<!-- npu="910b" id217 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id218 -->

<!-- npu="910b,A3" id250 -->
**Constraints & Notes**: This API involves high-risk operations. For usage, refer to the "[torch_npu.npu.change_current_allocator](https://gitcode.com/Ascend/op-plugin/blob/26.2.0/docs/en/custom_APIs/torch_npu-npu/torch-npu-npu-change_current_allocator.md)" section in Custom APIs

</div>
<!-- end id250 -->

### <code><i>class</i></code> torch.cuda.memory.MemPool

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.MemPool](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.MemPool.html#torch.cuda.memory.MemPool)

**NPU-Adapted Name**: torch.npu.memory.MemPool

**Support Status**:

<!-- npu="910b" id220 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id220 -->
<!-- npu="A3" id221 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id221 -->
<!-- npu="950" id222 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id222 -->

**Constraints & Notes**:

- `torch.npu.memory.MemPool`, `torch.npu.MemPool`, and `torch_npu.npu.MemPool` have the same functionality.
- The function prototype is `MemPool(allocator=None, use_on_oom=False, no_split=False)`. The `allocator` parameter specifies the NPU memory allocator used by the memory pool; `use_on_oom=True` indicates that when a memory request outside the memory pool triggers an OOM, the memory pool can be used as the last-resort memory allocation source; `no_split=True` indicates that the memory segments in the memory pool are not split.
- The `id` attribute returns the unique identifier of the memory pool, with a return type of `Tuple[int, int]`. For memory pools created by users through this interface, the first `int` is fixed to 0 to distinguish them from memory pools created internally by NPUGraph; the second `int` is an incrementally increasing unique number for user memory pools, which increments each time a new user memory pool is created.
- `use_count()` returns the current reference count of the memory pool, with a return type of `int`. The `MemPool` object itself holds one reference; entering the `torch.npu.use_mem_pool` context increases the reference count by 1, and exiting the context decreases it by 1.
- `snapshot()` returns a snapshot of the NPU memory allocator state filtered by the current memory pool ID, with a return type of `list`.
- `torch.npu.use_mem_pool` routes only the memory requests in the current thread to the specified memory pool, and new threads created in the context do not automatically use this memory pool. Before reclaiming the memory pool, exit the `torch.npu.use_mem_pool` context and release the Tensors that use this memory pool.
- TorchNPU does not provide the `MemPoolContext` and `MemPool.allocator` public interfaces from native PyTorch 2.7, and additionally supports the `no_split` parameter.
<!-- npu="950" id229 -->
- Ascend 950DT supports creating a memory pool with the default NPU caching allocator, but does not support specifying `NPUPluggableAllocator` through the `allocator` parameter.
<!-- end id229 -->

> <font size="3">allocator</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.MemPool.allocator](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.MemPool.html#torch.cuda.memory.MemPool.allocator)

**Support Status**:

<!-- npu="910b" id329 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id329 -->
<!-- npu="A3" id330 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id330 -->
<!-- npu="950" id331 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id331 -->

</div>

> <font size="3">id</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.MemPool.id](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.MemPool.html#torch.cuda.memory.MemPool.id)

**Support Status**:

<!-- npu="910b" id332 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id332 -->
<!-- npu="A3" id333 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id333 -->
<!-- npu="950" id334 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id334 -->

</div>

> <font size="3">snapshot()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.MemPool.snapshot](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.MemPool.html#torch.cuda.memory.MemPool.snapshot)

**Support Status**:

<!-- npu="910b" id335 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id335 -->
<!-- npu="A3" id336 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id336 -->
<!-- npu="950" id337 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id337 -->

</div>

> <font size="3">use_count()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.MemPool.use\_count](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.MemPool.html#torch.cuda.memory.MemPool.use_count)

**Support Status**:

<!-- npu="910b" id338 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id338 -->
<!-- npu="A3" id339 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id339 -->
<!-- npu="950" id340 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id340 -->

</div>

</div>

### torch.cuda.memory.caching_allocator_enable

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.memory.caching_allocator_enable](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.caching_allocator_enable.html#torch.cuda.memory.caching_allocator_enable)

**Support Status**:

<!-- npu="910b" id257 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id257 -->
<!-- npu="A3" id258 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id258 -->
<!-- npu="950" id259 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id259 -->

</div>

### <code><i>class</i></code> torch.cuda.use_mem_pool

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.use_mem_pool](https://docs.pytorch.org/docs/2.10/cuda.html#torch.cuda.use_mem_pool)

**Support Status**:

<!-- npu="910b" id260 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id260 -->
<!-- npu="A3" id261 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id261 -->
<!-- npu="950" id262 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id262 -->

</div>

## NVIDIA Tools Extension (NVTX)

### torch.cuda.nvtx.range_push

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.nvtx.range_push](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.nvtx.range_push.html#torch.cuda.nvtx.range_push)

**Support Status**:

<!-- npu="910b" id275 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id275 -->
<!-- npu="A3" id276 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id276 -->
<!-- npu="950" id277 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id277 -->

</div>

### torch.cuda.nvtx.range_pop

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.nvtx.range_pop](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.nvtx.range_pop.html#torch.cuda.nvtx.range_pop)

**Support Status**:

<!-- npu="910b" id272 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id272 -->
<!-- npu="A3" id273 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id273 -->
<!-- npu="950" id274 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id274 -->

</div>

### torch.cuda.nvtx.range

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.nvtx.range](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.nvtx.range.html#torch.cuda.nvtx.range)

**Support Status**:

<!-- npu="910b" id269 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id269 -->
<!-- npu="A3" id270 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id270 -->
<!-- npu="950" id271 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id271 -->

</div>

## Jiterator (beta)

### torch.cuda.jiterator.\_create\_multi\_output\_jit\_fn

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.jiterator.\_create\_multi\_output\_jit\_fn](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.jiterator._create_multi_output_jit_fn.html#torch.cuda.jiterator._create_multi_output_jit_fn)

**Support Status**:

<!-- npu="910b" id323 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id323 -->
<!-- npu="A3" id324 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id324 -->
<!-- npu="950" id325 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id325 -->

</div>

## TunableOp

### torch.cuda.tunable.is\_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.tunable.is\_enabled](https://docs.pytorch.org/docs/2.10/cuda.tunable.html#torch.cuda.tunable.is_enabled)

**Support Status**:

<!-- npu="910b" id350 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id350 -->
<!-- npu="A3" id351 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id351 -->
<!-- npu="950" id352 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id352 -->

</div>

### torch.cuda.tunable.tuning\_is\_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.tunable.tuning\_is\_enabled](https://docs.pytorch.org/docs/2.10/cuda.tunable.html#torch.cuda.tunable.tuning_is_enabled)

**Support Status**:

<!-- npu="910b" id353 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id353 -->
<!-- npu="A3" id354 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id354 -->
<!-- npu="950" id355 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id355 -->

</div>

### torch.cuda.tunable.record\_untuned\_enable

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.tunable.record\_untuned\_enable](https://docs.pytorch.org/docs/2.10/cuda.tunable.html#torch.cuda.tunable.record_untuned_enable)

**Support Status**:

<!-- npu="910b" id356 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id356 -->
<!-- npu="A3" id357 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id357 -->
<!-- npu="950" id358 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id358 -->

</div>

### torch.cuda.tunable.record\_untuned\_is\_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.tunable.record\_untuned\_is\_enabled](https://docs.pytorch.org/docs/2.10/cuda.tunable.html#torch.cuda.tunable.record_untuned_is_enabled)

**Support Status**:

<!-- npu="910b" id359 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id359 -->
<!-- npu="A3" id360 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id360 -->
<!-- npu="950" id361 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id361 -->

</div>

### torch.cuda.tunable.get\_max\_tuning\_duration

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.tunable.get\_max\_tuning\_duration](https://docs.pytorch.org/docs/2.10/cuda.tunable.html#torch.cuda.tunable.get_max_tuning_duration)

**Support Status**:

<!-- npu="910b" id362 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id362 -->
<!-- npu="A3" id363 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id363 -->
<!-- npu="950" id364 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id364 -->

</div>

### torch.cuda.tunable.get\_max\_tuning\_iterations

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.tunable.get\_max\_tuning\_iterations](https://docs.pytorch.org/docs/2.10/cuda.tunable.html#torch.cuda.tunable.get_max_tuning_iterations)

**Support Status**:

<!-- npu="910b" id365 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id365 -->
<!-- npu="A3" id366 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id366 -->
<!-- npu="950" id367 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id367 -->

</div>

### torch.cuda.tunable.set\_filename

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.tunable.set\_filename](https://docs.pytorch.org/docs/2.10/cuda.tunable.html#torch.cuda.tunable.set_filename)

**Support Status**:

<!-- npu="910b" id368 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id368 -->
<!-- npu="A3" id369 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id369 -->
<!-- npu="950" id370 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id370 -->

</div>

### torch.cuda.tunable.get\_filename

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.tunable.get\_filename](https://docs.pytorch.org/docs/2.10/cuda.tunable.html#torch.cuda.tunable.get_filename)

**Support Status**:

<!-- npu="910b" id371 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id371 -->
<!-- npu="A3" id372 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id372 -->
<!-- npu="950" id373 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id373 -->

</div>

### torch.cuda.tunable.get\_validators

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.tunable.get\_validators](https://docs.pytorch.org/docs/2.10/cuda.tunable.html#torch.cuda.tunable.get_validators)

**Support Status**:

<!-- npu="910b" id374 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id374 -->
<!-- npu="A3" id375 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id375 -->
<!-- npu="950" id376 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id376 -->

</div>

### torch.cuda.tunable.tune\_gemm\_in\_file

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.tunable.tune\_gemm\_in\_file](https://docs.pytorch.org/docs/2.10/cuda.tunable.html#torch.cuda.tunable.tune_gemm_in_file)

**Support Status**:

<!-- npu="910b" id377 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id377 -->
<!-- npu="A3" id378 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id378 -->
<!-- npu="950" id379 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id379 -->

</div>

### torch.cuda.tunable.get\_rotating\_buffer\_size

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.tunable.get\_rotating\_buffer\_size](https://docs.pytorch.org/docs/2.10/cuda.tunable.html#torch.cuda.tunable.get_rotating_buffer_size)

**Support Status**:

<!-- npu="910b" id380 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id380 -->
<!-- npu="A3" id381 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id381 -->
<!-- npu="950" id382 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id382 -->

</div>

## GPUDirect Storage (prototype)

### torch.cuda.gds.gds\_deregister\_buffer

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.gds.gds\_deregister\_buffer](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.gds.gds_deregister_buffer.html#torch.cuda.gds.gds_deregister_buffer)

**Support Status**:

<!-- npu="910b" id326 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id326 -->
<!-- npu="A3" id327 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id327 -->
<!-- npu="950" id328 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id328 -->

</div>

### <code><i>class</i></code> torch.cuda.gds.GdsFile

<div style="margin-left: 2em">

> <font size="3">deregister_handle()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.gds.GdsFile.deregister\_handle](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.gds.GdsFile.html#torch.cuda.gds.GdsFile.deregister_handle)

**Support Status**:

<!-- npu="910b" id341 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id341 -->
<!-- npu="A3" id342 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id342 -->
<!-- npu="950" id343 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id343 -->

</div>

> <font size="3">register_handle()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.gds.GdsFile.register\_handle](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.gds.GdsFile.html#torch.cuda.gds.GdsFile.register_handle)

**Support Status**:

<!-- npu="910b" id344 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id344 -->
<!-- npu="A3" id345 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id345 -->
<!-- npu="950" id346 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id346 -->

</div>

> <font size="3">save_storage()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.gds.GdsFile.save\_storage](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.gds.GdsFile.html#torch.cuda.gds.GdsFile.save_storage)

**Support Status**:

<!-- npu="910b" id347 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id347 -->
<!-- npu="A3" id348 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id348 -->
<!-- npu="950" id349 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id349 -->

</div>

</div>