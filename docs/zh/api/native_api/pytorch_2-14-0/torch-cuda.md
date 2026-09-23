# torch.cuda

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.14/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 在使用支持的CUDA接口时，需要将API名称中的CUDA替换为NPU形式才能使用：将torch.cuda.替换为torch_npu.npu.或torch.npu.。torch_npu.npu.和torch.npu.两种调用方式，功能一致。举例如下：
>
>   `torch.cuda.current_device` --> `torch_npu.npu.current_device`<br>
>   `torch.cuda.current_device` --> `torch.npu.current_device`
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.14/cuda.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

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

**原生文档**：[torch.cuda.StreamContext](https://pytorch.org/docs/2.14/generated/torch.cuda.StreamContext.html)

**NPU 形式名称**：torch.npu.StreamContext

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id3 -->

</div>

### torch.cuda.can_device_access_peer

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.can_device_access_peer](https://pytorch.org/docs/2.14/generated/torch.cuda.can_device_access_peer.html)

**NPU 形式名称**：torch_npu.npu.can_device_access_peer

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id6 -->

</div>

### torch.cuda.current_blas_handle

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.current_blas_handle](https://pytorch.org/docs/2.14/generated/torch.cuda.current_blas_handle.html)

**NPU 形式名称**：torch_npu.npu.current_blas_handle

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id9 -->

</div>

### torch.cuda.current_stream

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.current_stream](https://pytorch.org/docs/2.14/generated/torch.cuda.current_stream.html)

**NPU 形式名称**：torch_npu.npu.current_stream

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id12 -->

**限制与说明**： 未设置`device`时，调用该接口会隐式地初始化当前`device`（默认0卡）

</div>

### torch.cuda.default_stream

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.default_stream](https://pytorch.org/docs/2.14/generated/torch.cuda.default_stream.html)

**NPU 形式名称**：torch_npu.npu.default_stream

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id15 -->

**限制与说明**： 未设置`device`时，调用该接口会隐式地初始化当前`device`（默认0卡）

</div>

### torch.cuda.device_count

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.device_count](https://pytorch.org/docs/2.14/generated/torch.cuda.device_count.html)

**NPU 形式名称**：torch_npu.npu.device_count

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id18 -->

</div>

### torch.cuda.device_of

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.device_of](https://pytorch.org/docs/2.14/generated/torch.cuda.device_of.html)

**NPU 形式名称**：torch_npu.npu.device_of

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id21 -->

</div>

### torch.cuda.get_device_capability

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.get_device_capability](https://pytorch.org/docs/2.14/generated/torch.cuda.get_device_capability.html)

**NPU 形式名称**：torch_npu.npu.get_device_capability

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id24 -->

**限制与说明**： 通过环境变量TORCH_NPU_DEVICE_CAPABILITY配置`torch_npu.npu.get_device_capability()`的返回值，仅用于兼容原生PyTorch，不代表NPU硬件实际能力

</div>

### torch.cuda.get_device_name

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.get_device_name](https://pytorch.org/docs/2.14/generated/torch.cuda.get_device_name.html)

**NPU 形式名称**：torch_npu.npu.get_device_name

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id27 -->

</div>

### torch.cuda.get_device_properties

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.get_device_properties](https://pytorch.org/docs/2.14/generated/torch.cuda.get_device_properties.html)

**NPU 形式名称**：torch_npu.npu.get_device_properties

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id30 -->

**限制与说明**： 仅支持name、total_memory、L2_cache_size、cube_core_num和vector_core_num属性，原CUDA上支持的其余属性均返回空字段

</div>

### torch.cuda.get_sync_debug_mode

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.get_sync_debug_mode](https://pytorch.org/docs/2.14/generated/torch.cuda.get_sync_debug_mode.html)

**NPU 形式名称**：torch_npu.npu.get_sync_debug_mode

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id33 -->

</div>

### torch.cuda.init

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.init](https://pytorch.org/docs/2.14/generated/torch.cuda.init.html)

**NPU 形式名称**：torch_npu.npu.init

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id36 -->

</div>

### torch.cuda.ipc_collect

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.ipc_collect](https://pytorch.org/docs/2.14/generated/torch.cuda.ipc_collect.html)

**NPU 形式名称**：torch_npu.npu.ipc_collect

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id39 -->

</div>

### torch.cuda.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.is_available](https://pytorch.org/docs/2.14/generated/torch.cuda.is_available.html)

**NPU 形式名称**：torch_npu.npu.is_available

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id42 -->

</div>

### torch.cuda.is_initialized

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.is_initialized](https://pytorch.org/docs/2.14/generated/torch.cuda.is_initialized.html)

**NPU 形式名称**：torch_npu.npu.is_initialized

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id45 -->

</div>

### torch.cuda.memory_usage

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory_usage](https://pytorch.org/docs/2.14/generated/torch.cuda.memory_usage.html)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id48 -->

</div>

### torch.cuda.set_device

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.set_device](https://pytorch.org/docs/2.14/generated/torch.cuda.set_device.html)

**NPU 形式名称**：torch_npu.npu.set_device

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id51 -->

</div>

### torch.cuda.set_stream

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.set_stream](https://pytorch.org/docs/2.14/generated/torch.cuda.set_stream.html)

**NPU 形式名称**：torch_npu.npu.set_stream

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id54 -->

</div>

### torch.cuda.set_sync_debug_mode

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.set_sync_debug_mode](https://pytorch.org/docs/2.14/generated/torch.cuda.set_sync_debug_mode.html)

**NPU 形式名称**：torch_npu.npu.set_sync_debug_mode

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id57 -->

</div>

### torch.cuda.stream

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.stream](https://pytorch.org/docs/2.14/cuda.html#torch.cuda.stream)

**NPU 形式名称**：torch_npu.npu.stream

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id60 -->

</div>

### torch.cuda.synchronize

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.synchronize](https://pytorch.org/docs/2.14/generated/torch.cuda.synchronize.html)

**NPU 形式名称**：torch_npu.npu.synchronize

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id63 -->

</div>

### torch.cuda.utilization

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.utilization](https://pytorch.org/docs/2.14/generated/torch.cuda.utilization.html)

**NPU 形式名称**：torch_npu.npu.utilization

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id66 -->

</div>

### torch.cuda._sanitizer.enable_cuda_sanitizer

<div style="margin-left: 2em">

**原生文档**：[torch.cuda._sanitizer.enable_cuda_sanitizer](https://pytorch.org/docs/2.14/cuda._sanitizer.html#torch.cuda._sanitizer.enable_cuda_sanitizer)

**NPU 形式名称**：torch_npu.npu._sanitizer.enable_npu_sanitizer

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id69 -->

</div>

### torch.cuda.current_device

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.current_device](https://pytorch.org/docs/2.14/generated/torch.cuda.current_device.html)

**NPU 形式名称**：torch_npu.npu.current_device

**产品支持情况**：

<!-- npu="910b" id70 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id72 -->

</div>

### torch.cuda.device

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.device](https://pytorch.org/docs/2.14/generated/torch.cuda.device.html)

**NPU 形式名称**：torch_npu.npu.device

**产品支持情况**：

<!-- npu="910b" id73 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id75 -->

</div>

## Random Number Generator

### torch.cuda.get_rng_state

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.get_rng_state](https://pytorch.org/docs/2.14/generated/torch.cuda.get_rng_state.html)

**NPU 形式名称**：torch_npu.npu.get_rng_state

**产品支持情况**：

<!-- npu="910b" id76 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id78 -->

</div>

### torch.cuda.set_rng_state

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.set_rng_state](https://pytorch.org/docs/2.14/generated/torch.cuda.set_rng_state.html)

**NPU 形式名称**：torch_npu.npu.set_rng_state

**产品支持情况**：

<!-- npu="910b" id79 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id81 -->

</div>

### torch.cuda.set_rng_state_all

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.set_rng_state_all](https://pytorch.org/docs/2.14/generated/torch.cuda.set_rng_state_all.html)

**NPU 形式名称**：torch_npu.npu.set_rng_state_all

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id84 -->

</div>

### torch.cuda.manual_seed

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.manual_seed](https://pytorch.org/docs/2.14/generated/torch.cuda.manual_seed.html)

**NPU 形式名称**：torch_npu.npu.manual_seed

**产品支持情况**：

<!-- npu="910b" id85 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id87 -->

</div>

### torch.cuda.manual_seed_all

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.manual_seed_all](https://pytorch.org/docs/2.14/generated/torch.cuda.manual_seed_all.html)

**NPU 形式名称**：torch_npu.npu.manual_seed_all

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id90 -->

</div>

### torch.cuda.seed

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.seed](https://pytorch.org/docs/2.14/generated/torch.cuda.seed.html)

**NPU 形式名称**：torch_npu.npu.seed

**产品支持情况**：

<!-- npu="910b" id91 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id93 -->

</div>

### torch.cuda.seed_all

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.seed_all](https://pytorch.org/docs/2.14/generated/torch.cuda.seed_all.html)

**NPU 形式名称**：torch_npu.npu.seed_all

**产品支持情况**：

<!-- npu="910b" id94 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id96 -->

</div>

### torch.cuda.initial_seed

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.initial_seed](https://pytorch.org/docs/2.14/generated/torch.cuda.initial_seed.html)

**NPU 形式名称**：torch_npu.npu.initial_seed

**产品支持情况**：

<!-- npu="910b" id97 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id99 -->

</div>

## Communication collectives

### torch.cuda.comm.scatter

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.comm.scatter](https://pytorch.org/docs/2.14/generated/torch.cuda.comm.scatter.html)

**产品支持情况**：

<!-- npu="910b" id100 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id102 -->

</div>

### torch.cuda.comm.gather

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.comm.gather](https://pytorch.org/docs/2.14/generated/torch.cuda.comm.gather.html)

**产品支持情况**：

<!-- npu="910b" id103 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id105 -->

</div>

## Streams and events

### <code><i>class</i></code> torch.cuda.Stream

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Stream](https://pytorch.org/docs/2.14/cuda.html#torch.cuda.Stream)

**NPU 形式名称**：torch_npu.npu.Stream

**产品支持情况**：

<!-- npu="910b" id106 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id108 -->

> <font size="3">wait_stream()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Stream.wait_stream](https://pytorch.org/docs/2.14/generated/torch.cuda.Stream_class.html#torch.cuda.Stream.wait_stream)

**产品支持情况**：

<!-- npu="910b" id109 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id111 -->

</div>

</div>

### <code><i>class</i></code> torch.cuda.Event

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Event](https://pytorch.org/docs/2.14/generated/torch.cuda.Event.html)

**NPU 形式名称**：torch_npu.npu.Event

**产品支持情况**：

<!-- npu="910b" id112 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id114 -->

> <font size="3">elapsed_time()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Event.elapsed_time](https://pytorch.org/docs/2.14/generated/torch.cuda.Event.html#torch.cuda.Event.elapsed_time)

**产品支持情况**：

<!-- npu="910b" id115 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id117 -->

</div>

> <font size="3">from_ipc_handle()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Event.from_ipc_handle](https://pytorch.org/docs/2.14/generated/torch.cuda.Event.html#torch.cuda.Event.from_ipc_handle)

**产品支持情况**：

<!-- npu="910b" id118 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id120 -->

</div>

> <font size="3">ipc_handle()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Event.ipc_handle](https://pytorch.org/docs/2.14/generated/torch.cuda.Event.html#torch.cuda.Event.ipc_handle)

**产品支持情况**：

<!-- npu="910b" id121 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id123 -->

</div>

> <font size="3">query()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Event.query](https://pytorch.org/docs/2.14/generated/torch.cuda.Event.html#torch.cuda.Event.query)

**产品支持情况**：

<!-- npu="910b" id124 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id126 -->

</div>

> <font size="3">wait()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Event.wait](https://pytorch.org/docs/2.14/generated/torch.cuda.Event.html#torch.cuda.Event.wait)

**产品支持情况**：

<!-- npu="910b" id127 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id129 -->

</div>

</div>

## Graphs (beta)

### torch.cuda.is_current_stream_capturing

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.is_current_stream_capturing](https://pytorch.org/docs/2.14/generated/torch.cuda.is_current_stream_capturing.html)

**NPU 形式名称**：torch.npu.is_current_stream_capturing

**产品支持情况**：

<!-- npu="910b" id130 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id132 -->

</div>

### torch.cuda.graph_pool_handle

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.graph_pool_handle](https://pytorch.org/docs/2.14/generated/torch.cuda.graph_pool_handle.html)

**NPU 形式名称**：torch.npu.graph_pool_handle

**产品支持情况**：

<!-- npu="910b" id133 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id135 -->

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

### <code><i>class</i></code> torch.cuda.CUDAGraph

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph](https://pytorch.org/docs/2.14/generated/torch.cuda.CUDAGraph.html)

**NPU 形式名称**：torch.npu.NPUGraph

**产品支持情况**：

<!-- npu="910b" id136 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id138 -->

**限制与说明**： 当前仅支持推理场景，不支持训练场景

> <font size="3">capture_begin()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph.capture_begin](https://pytorch.org/docs/2.14/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.capture_begin)

**产品支持情况**：

<!-- npu="910b" id139 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id141 -->

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

> <font size="3">capture_end()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph.capture_end](https://pytorch.org/docs/2.14/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.capture_end)

**产品支持情况**：

<!-- npu="910b" id142 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id143 -->
<!-- npu="950" id144 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id144 -->

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

> <font size="3">debug_dump()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph.debug_dump](https://pytorch.org/docs/2.14/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.debug_dump)

**产品支持情况**：

<!-- npu="910b" id145 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id147 -->

**限制与说明**：

- 当前仅支持推理场景，不支持训练场景
- 导出文件内容为json格式

</div>

> <font size="3">pool()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph.pool](https://pytorch.org/docs/2.14/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.pool)

**产品支持情况**：

<!-- npu="910b" id148 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id150 -->

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

> <font size="3">replay()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph.replay](https://pytorch.org/docs/2.14/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.replay)

**产品支持情况**：

<!-- npu="910b" id151 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id153 -->

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

> <font size="3">reset()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph.reset](https://pytorch.org/docs/2.14/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.reset)

**产品支持情况**：

<!-- npu="910b" id154 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id156 -->

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

</div>

### torch.cuda.graph

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.graph](https://pytorch.org/docs/2.14/generated/torch.cuda.graph.html)

**NPU 形式名称**：torch.npu.graph

**产品支持情况**：

<!-- npu="910b" id157 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id159 -->

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

### torch.cuda.make_graphed_callables

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.make_graphed_callables](https://pytorch.org/docs/2.14/generated/torch.cuda.make_graphed_callables.html)

**NPU 形式名称**：torch.npu.make_graphed_callables

**产品支持情况**：

<!-- npu="910b" id160 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id162 -->

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

## Memory management

### torch.cuda.empty_cache

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.empty_cache](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.empty_cache.html)

**NPU 形式名称**：torch_npu.npu.empty_cache

**产品支持情况**：

<!-- npu="910b" id163 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id165 -->

</div>

### torch.cuda.mem_get_info

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.mem_get_info](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.mem_get_info.html)

**NPU 形式名称**：torch_npu.npu.mem_get_info

**产品支持情况**：

<!-- npu="910b" id166 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id168 -->

</div>

### torch.cuda.memory_stats

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory_stats](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.memory_stats.html)

**NPU 形式名称**：torch_npu.npu.memory_stats

**产品支持情况**：

<!-- npu="910b" id169 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id171 -->

</div>

### torch.cuda.memory_summary

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory_summary](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.memory_summary.html)

**NPU 形式名称**：torch_npu.npu.memory_summary

**产品支持情况**：

<!-- npu="910b" id172 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id174 -->

</div>

### torch.cuda.memory_allocated

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory_allocated](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.memory_allocated.html)

**NPU 形式名称**：torch_npu.npu.memory_allocated

**产品支持情况**：

<!-- npu="910b" id175 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id177 -->

</div>

### torch.cuda.max_memory_allocated

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.max_memory_allocated](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_allocated.html)

**NPU 形式名称**：torch_npu.npu.max_memory_allocated

**产品支持情况**：

<!-- npu="910b" id178 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id180 -->

</div>

### torch.cuda.reset_max_memory_allocated

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.reset_max_memory_allocated](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.reset_max_memory_allocated.html)

**NPU 形式名称**：torch_npu.npu.reset_max_memory_allocated

**产品支持情况**：

<!-- npu="910b" id181 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id183 -->

</div>

### torch.cuda.memory_reserved

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory_reserved](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.memory_reserved.html)

**NPU 形式名称**：torch_npu.npu.memory_reserved

**产品支持情况**：

<!-- npu="910b" id184 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id186 -->

</div>

### torch.cuda.max_memory_reserved

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.max_memory_reserved](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_reserved.html)

**NPU 形式名称**：torch_npu.npu.max_memory_reserved

**产品支持情况**：

<!-- npu="910b" id187 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id189 -->

</div>

### torch.cuda.set_per_process_memory_fraction

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.set_per_process_memory_fraction](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.set_per_process_memory_fraction.html)

**NPU 形式名称**：torch_npu.npu.set_per_process_memory_fraction

**产品支持情况**：

<!-- npu="910b" id190 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id192 -->

</div>

### torch.cuda.memory_cached

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory_cached](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.memory_cached.html)

**NPU 形式名称**：torch_npu.npu.memory_cached

**产品支持情况**：

<!-- npu="910b" id193 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id195 -->

</div>

### torch.cuda.max_memory_cached

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.max_memory_cached](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_cached.html)

**NPU 形式名称**：torch_npu.npu.max_memory_cached

**产品支持情况**：

<!-- npu="910b" id196 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id197 -->
<!-- npu="950" id198 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id198 -->

</div>

### torch.cuda.reset_max_memory_cached

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.reset_max_memory_cached](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.reset_max_memory_cached.html)

**NPU 形式名称**：torch_npu.npu.reset_max_memory_cached

**产品支持情况**：

<!-- npu="910b" id199 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id200 -->
<!-- npu="950" id201 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id201 -->

</div>

### torch.cuda.reset_peak_memory_stats

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.reset_peak_memory_stats](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.reset_peak_memory_stats.html)

**NPU 形式名称**：torch_npu.npu.reset_peak_memory_stats

**产品支持情况**：

<!-- npu="910b" id202 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id203 -->
<!-- npu="950" id204 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id204 -->

</div>

### torch.cuda.caching_allocator_alloc

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.caching_allocator_alloc](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.caching_allocator_alloc.html)

**NPU 形式名称**：torch_npu.npu.caching_allocator_alloc

**产品支持情况**：

<!-- npu="910b" id205 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id206 -->
<!-- npu="950" id207 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id207 -->

</div>

### torch.cuda.caching_allocator_delete

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.caching_allocator_delete](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.caching_allocator_delete.html)

**NPU 形式名称**：torch_npu.npu.caching_allocator_delete

**产品支持情况**：

<!-- npu="910b" id208 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id209 -->
<!-- npu="950" id210 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id210 -->

</div>

### torch.cuda.get_allocator_backend

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.get_allocator_backend](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.get_allocator_backend.html)

**NPU 形式名称**：torch_npu.npu.get_allocator_backend

**产品支持情况**：

<!-- npu="910b" id211 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id212 -->
<!-- npu="950" id213 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id213 -->

</div>

### <code><i>class</i></code> torch.cuda.CUDAPluggableAllocator

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAPluggableAllocator](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.CUDAPluggableAllocator.html)

**NPU 形式名称**：torch_npu.npu.NPUPluggableAllocator

**产品支持情况**：

<!-- npu="910b" id214 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id215 -->
<!-- npu="950" id216 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id216 -->

**限制与说明**： 该接口涉及高危操作，使用请参考《自定义API》中的“[torch_npu.npu.NPUPluggableAllocator](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-npu/torch-npu-npu-NPUPluggableAllocator.md)”章节。

</div>

### torch.cuda.change_current_allocator

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.change_current_allocator](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.change_current_allocator.html)

**NPU 形式名称**：torch_npu.npu.change_current_allocator

**产品支持情况**：

<!-- npu="910b" id217 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id218 -->
<!-- npu="950" id219 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id219 -->

**限制与说明**： 该接口涉及高危操作，使用请参考《自定义API》中的“[torch_npu.npu.change_current_allocator](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-npu/torch-npu-npu-change_current_allocator.md)”章节。

</div>

### <code><i>class</i></code> torch.cuda.memory.MemPool

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.MemPool](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.MemPool.html)

**NPU 形式名称**：torch.npu.memory.MemPool

**产品支持情况**：

<!-- npu="910b" id220 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id220 -->
<!-- npu="A3" id221 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id221 -->
<!-- npu="950" id222 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id222 -->

**限制与说明**：

- `torch.npu.memory.MemPool`、`torch.npu.MemPool`和`torch_npu.npu.MemPool`功能一致。
- 函数原型为`MemPool(allocator=None, use_on_oom=False, no_split=False)`。`allocator`用于指定内存池使用的NPU内存分配器；`use_on_oom=True`表示内存池外的内存申请发生OOM时，可以将该内存池作为最后的内存分配来源；`no_split=True`表示不拆分该内存池中的内存段。
- `id`属性返回类型为`Tuple[int, int]`的内存池唯一标识。对于用户通过本接口创建的内存池，第一个`int`固定为0，用于与NPUGraph内部创建的内存池进行区分；第二个`int`是用户内存池的递增唯一编号，每创建一个新的用户内存池，该编号递增。
- `use_count()`返回内存池当前的引用计数，返回值类型为`int`。`MemPool`对象本身持有一个引用；进入`torch.npu.use_mem_pool`上下文后引用计数加1，退出上下文后引用计数减1。
- `snapshot()`返回根据当前内存池ID过滤后的NPU内存分配器状态快照，返回值类型为`list`。与原生PyTorch接口不同，当前NPU接口不支持`include_traces`参数。
- `torch.npu.use_mem_pool`仅将当前线程中的内存申请路由到指定内存池，在上下文中创建的新线程不会自动使用该内存池。回收内存池前，需要退出`torch.npu.use_mem_pool`上下文，并释放使用该内存池的Tensor。
- Ascend 950DT系列产品支持使用默认NPU缓存分配器创建内存池，不支持通过`allocator`参数指定`NPUPluggableAllocator`。

</div>

### torch.cuda.reset_accumulated_host_memory_stats

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.reset_accumulated_host_memory_stats](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.reset_accumulated_host_memory_stats.html)

**NPU 形式名称**：torch_npu.npu.reset_accumulated_host_memory_stats

**产品支持情况**：

<!-- npu="910b" id223 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id223 -->
<!-- npu="A3" id224 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id224 -->
<!-- npu="950" id225 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id225 -->

</div>

### torch.cuda.reset_peak_host_memory_stats

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.reset_peak_host_memory_stats](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.reset_peak_host_memory_stats.html)

**NPU 形式名称**：torch_npu.npu.reset_peak_host_memory_stats

**产品支持情况**：

<!-- npu="910b" id226 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id226 -->
<!-- npu="A3" id227 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id227 -->
<!-- npu="950" id228 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id228 -->

</div>

### torch.cuda.host_memory_stats_as_nested_dict

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.host_memory_stats_as_nested_dict](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.host_memory_stats_as_nested_dict.html)

**NPU 形式名称**：torch_npu.npu.host_memory_stats_as_nested_dict

**产品支持情况**：

<!-- npu="910b" id229 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id229 -->
<!-- npu="A3" id230 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id230 -->
<!-- npu="950" id231 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id231 -->

</div>

### torch.cuda.host_memory_stats

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.host_memory_stats](https://pytorch.org/docs/2.14/generated/torch.cuda.memory.host_memory_stats.html)

**NPU 形式名称**：torch_npu.npu.host_memory_stats

**产品支持情况**：

<!-- npu="910b" id232 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id232 -->
<!-- npu="A3" id233 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id233 -->
<!-- npu="950" id234 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id234 -->

</div>
