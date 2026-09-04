# torch.cuda

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.11/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 在使用支持的CUDA接口时，需要将API名称中的CUDA替换为NPU形式才能使用：将torch.cuda.替换为torch_npu.npu.或torch.npu.。torch_npu.npu.和torch.npu.两种调用方式，功能一致。举例如下：
>
>   `torch.cuda.current_device` --> `torch_npu.npu.current_device`<br>
>   `torch.cuda.current_device` --> `torch.npu.current_device`
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.11/cuda.html)。

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

**原生文档**：[torch.cuda.StreamContext](https://pytorch.org/docs/2.11/generated/torch.cuda.StreamContext.html)

**NPU 形式名称**：torch.npu.StreamContext

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.can_device_access_peer

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.can_device_access_peer](https://pytorch.org/docs/2.11/generated/torch.cuda.can_device_access_peer.html)

**NPU 形式名称**：torch_npu.npu.can_device_access_peer

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.current_blas_handle

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.current_blas_handle](https://pytorch.org/docs/2.11/generated/torch.cuda.current_blas_handle.html)

**NPU 形式名称**：torch_npu.npu.current_blas_handle

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.current_stream

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.current_stream](https://pytorch.org/docs/2.11/generated/torch.cuda.current_stream.html)

**NPU 形式名称**：torch_npu.npu.current_stream

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**：未设置`device`时，调用该接口会隐式地初始化当前`device`（默认0卡）

</div>

### torch.cuda.default_stream

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.default_stream](https://pytorch.org/docs/2.11/generated/torch.cuda.default_stream.html)

**NPU 形式名称**：torch_npu.npu.default_stream

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**：未设置`device`时，调用该接口会隐式地初始化当前`device`（默认0卡）

</div>

### torch.cuda.device_count

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.device_count](https://pytorch.org/docs/2.11/generated/torch.cuda.device_count.html)

**NPU 形式名称**：torch_npu.npu.device_count

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.device_of

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.device_of](https://pytorch.org/docs/2.11/generated/torch.cuda.device_of.html)

**NPU 形式名称**：torch_npu.npu.device_of

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.get_device_capability

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.get_device_capability](https://pytorch.org/docs/2.11/generated/torch.cuda.get_device_capability.html)

**NPU 形式名称**：torch_npu.npu.get_device_capability

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： 通过环境变量TORCH_NPU_DEVICE_CAPABILITY配置`torch_npu.npu.get_device_capability()`的返回值，仅用于兼容原生PyTorch，不代表NPU硬件实际能力

</div>

### torch.cuda.get_device_name

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.get_device_name](https://pytorch.org/docs/2.11/generated/torch.cuda.get_device_name.html)

**NPU 形式名称**：torch_npu.npu.get_device_name

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.get_device_properties

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.get_device_properties](https://pytorch.org/docs/2.11/generated/torch.cuda.get_device_properties.html)

**NPU 形式名称**：torch_npu.npu.get_device_properties

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： 仅支持name、total_memory、L2_cache_size、cube_core_num和vector_core_num属性，原cuda上支持的其余属性均返回空字段

</div>

### torch.cuda.get_sync_debug_mode

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.get_sync_debug_mode](https://pytorch.org/docs/2.11/generated/torch.cuda.get_sync_debug_mode.html)

**NPU 形式名称**：torch_npu.npu.get_sync_debug_mode

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.cuda.init

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.init](https://pytorch.org/docs/2.11/generated/torch.cuda.init.html)

**NPU 形式名称**：torch_npu.npu.init

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.ipc_collect

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.ipc_collect](https://pytorch.org/docs/2.11/generated/torch.cuda.ipc_collect.html)

**NPU 形式名称**：torch_npu.npu.ipc_collect

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.is_available](https://pytorch.org/docs/2.11/generated/torch.cuda.is_available.html)

**NPU 形式名称**：torch_npu.npu.is_available

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.is_initialized

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.is_initialized](https://pytorch.org/docs/2.11/generated/torch.cuda.is_initialized.html)

**NPU 形式名称**：torch_npu.npu.is_initialized

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.cuda.memory_usage

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory_usage](https://pytorch.org/docs/2.11/generated/torch.cuda.memory_usage.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.cuda.set_device

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.set_device](https://pytorch.org/docs/2.11/generated/torch.cuda.set_device.html)

**NPU 形式名称**：torch_npu.npu.set_device

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.current_device

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.current_device](https://pytorch.org/docs/2.11/generated/torch.cuda.current_device.html)

**NPU 形式名称**：torch_npu.npu.current_device

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.set_stream

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.set_stream](https://pytorch.org/docs/2.11/generated/torch.cuda.set_stream.html)

**NPU 形式名称**：torch_npu.npu.set_stream

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.set_sync_debug_mode

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.set_sync_debug_mode](https://pytorch.org/docs/2.11/generated/torch.cuda.set_sync_debug_mode.html)

**NPU 形式名称**：torch_npu.npu.set_sync_debug_mode

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.cuda.stream

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.stream](https://pytorch.org/docs/2.11/cuda.html#torch.cuda.stream)

**NPU 形式名称**：torch_npu.npu.stream

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.synchronize

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.synchronize](https://pytorch.org/docs/2.11/generated/torch.cuda.synchronize.html)

**NPU 形式名称**：torch_npu.npu.synchronize

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.utilization

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.utilization](https://pytorch.org/docs/2.11/generated/torch.cuda.utilization.html)

**NPU 形式名称**：torch_npu.npu.utilization

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda._sanitizer.enable_cuda_sanitizer

<div style="margin-left: 2em">

**原生文档**：[torch.cuda._sanitizer.enable_cuda_sanitizer](https://pytorch.org/docs/2.11/cuda._sanitizer.html#torch.cuda._sanitizer.enable_cuda_sanitizer)

**NPU 形式名称**：torch_npu.npu._sanitizer.enable_npu_sanitizer

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

## Random Number Generator

### torch.cuda.get_rng_state

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.get_rng_state](https://pytorch.org/docs/2.11/generated/torch.cuda.get_rng_state.html)

**NPU 形式名称**：torch_npu.npu.get_rng_state

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.set_rng_state

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.set_rng_state](https://pytorch.org/docs/2.11/generated/torch.cuda.set_rng_state.html)

**NPU 形式名称**：torch_npu.npu.set_rng_state

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.set_rng_state_all

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.set_rng_state_all](https://pytorch.org/docs/2.11/generated/torch.cuda.set_rng_state_all.html)

**NPU 形式名称**：torch_npu.npu.set_rng_state_all

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.manual_seed

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.manual_seed](https://pytorch.org/docs/2.11/generated/torch.cuda.manual_seed.html)

**NPU 形式名称**：torch_npu.npu.manual_seed

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.manual_seed_all

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.manual_seed_all](https://pytorch.org/docs/2.11/generated/torch.cuda.manual_seed_all.html)

**NPU 形式名称**：torch_npu.npu.manual_seed_all

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.seed

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.seed](https://pytorch.org/docs/2.11/generated/torch.cuda.seed.html)

**NPU 形式名称**：torch_npu.npu.seed

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.seed_all

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.seed_all](https://pytorch.org/docs/2.11/generated/torch.cuda.seed_all.html)

**NPU 形式名称**：torch_npu.npu.seed_all

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.initial_seed

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.initial_seed](https://pytorch.org/docs/2.11/generated/torch.cuda.initial_seed.html)

**NPU 形式名称**：torch_npu.npu.initial_seed

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

## Communication collectives

### torch.cuda.comm.scatter

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.comm.scatter](https://pytorch.org/docs/2.11/generated/torch.cuda.comm.scatter.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.cuda.comm.gather

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.comm.gather](https://pytorch.org/docs/2.11/generated/torch.cuda.comm.gather.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

## Streams and events

### <code><i>class</i></code> torch.cuda.Stream

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Stream](https://pytorch.org/docs/2.11/cuda.html#torch.cuda.Stream)

**NPU 形式名称**：torch_npu.npu.Stream

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

> <font size="3">wait_stream()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Stream.wait_stream](https://pytorch.org/docs/2.11/generated/torch.cuda.Stream_class.html#torch.cuda.Stream.wait_stream)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.cuda.Event

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Event](https://pytorch.org/docs/2.11/generated/torch.cuda.Event.html)

**NPU 形式名称**：torch_npu.npu.Event

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

> <font size="3">elapsed_time()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Event.elapsed_time](https://pytorch.org/docs/2.11/generated/torch.cuda.Event.html#torch.cuda.Event.elapsed_time)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">from_ipc_handle()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Event.from_ipc_handle](https://pytorch.org/docs/2.11/generated/torch.cuda.Event.html#torch.cuda.Event.from_ipc_handle)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">ipc_handle()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Event.ipc_handle](https://pytorch.org/docs/2.11/generated/torch.cuda.Event.html#torch.cuda.Event.ipc_handle)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">query()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Event.query](https://pytorch.org/docs/2.11/generated/torch.cuda.Event.html#torch.cuda.Event.query)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

> <font size="3">wait()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.Event.wait](https://pytorch.org/docs/2.11/generated/torch.cuda.Event.html#torch.cuda.Event.wait)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

</div>

## Graphs (beta)

### torch.cuda.is_current_stream_capturing

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.is_current_stream_capturing](https://pytorch.org/docs/2.11/generated/torch.cuda.is_current_stream_capturing.html)

**NPU 形式名称**：torch.npu.is_current_stream_capturing

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.graph_pool_handle

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.graph_pool_handle](https://pytorch.org/docs/2.11/generated/torch.cuda.graph_pool_handle.html)

**NPU 形式名称**：torch.npu.graph_pool_handle

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

### <code><i>class</i></code> torch.cuda.CUDAGraph

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph](https://pytorch.org/docs/2.11/generated/torch.cuda.CUDAGraph.html)

**NPU 形式名称**：torch.npu.NPUGraph

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： 当前仅支持推理场景，不支持训练场景

> <font size="3">capture_begin()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph.capture_begin](https://pytorch.org/docs/2.11/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.capture_begin)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

> <font size="3">capture_end()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph.capture_end](https://pytorch.org/docs/2.11/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.capture_end)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

> <font size="3">debug_dump()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph.debug_dump](https://pytorch.org/docs/2.11/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.debug_dump)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：

- 当前仅支持推理场景，不支持训练场景
- 导出文件内容为json格式

</div>

> <font size="3">pool()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph.pool](https://pytorch.org/docs/2.11/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.pool)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

> <font size="3">replay()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph.replay](https://pytorch.org/docs/2.11/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.replay)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

> <font size="3">reset()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAGraph.reset](https://pytorch.org/docs/2.11/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.reset)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

</div>

### torch.cuda.graph

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.graph](https://pytorch.org/docs/2.11/generated/torch.cuda.graph.html)

**NPU 形式名称**：torch.npu.graph

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

### torch.cuda.make_graphed_callables

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.make_graphed_callables](https://pytorch.org/docs/2.11/generated/torch.cuda.make_graphed_callables.html)

**NPU 形式名称**：torch.npu.make_graphed_callables

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： 当前仅支持推理场景，不支持训练场景

</div>

## Memory management

### torch.cuda.device

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.device](https://pytorch.org/docs/2.11/generated/torch.cuda.device.html)

**NPU 形式名称**：torch_npu.npu.device

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.empty_cache

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.empty_cache](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.empty_cache.html)

**NPU 形式名称**：torch_npu.npu.empty_cache

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.mem_get_info

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.mem_get_info](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.mem_get_info.html)

**NPU 形式名称**：torch_npu.npu.mem_get_info

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.memory_stats

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory_stats](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.memory_stats.html)

**NPU 形式名称**：torch_npu.npu.memory_stats

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.memory_summary

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory_summary](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.memory_summary.html)

**NPU 形式名称**：torch_npu.npu.memory_summary

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.cuda.memory_allocated

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory_allocated](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.memory_allocated.html)

**NPU 形式名称**：torch_npu.npu.memory_allocated

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.max_memory_allocated

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.max_memory_allocated](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.max_memory_allocated.html)

**NPU 形式名称**：torch_npu.npu.max_memory_allocated

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.reset_max_memory_allocated

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.reset_max_memory_allocated](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.reset_max_memory_allocated.html)

**NPU 形式名称**：torch_npu.npu.reset_max_memory_allocated

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.memory_reserved

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory_reserved](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.memory_reserved.html)

**NPU 形式名称**：torch_npu.npu.memory_reserved

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.max_memory_reserved

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.max_memory_reserved](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.max_memory_reserved.html)

**NPU 形式名称**：torch_npu.npu.max_memory_reserved

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.set_per_process_memory_fraction

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.set_per_process_memory_fraction](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.set_per_process_memory_fraction.html)

**NPU 形式名称**：torch_npu.npu.set_per_process_memory_fraction

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.cuda.memory_cached

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory_cached](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.memory_cached.html)

**NPU 形式名称**：torch_npu.npu.memory_cached

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.max_memory_cached

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.max_memory_cached](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.max_memory_cached.html)

**NPU 形式名称**：torch_npu.npu.max_memory_cached

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.reset_max_memory_cached

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.reset_max_memory_cached](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.reset_max_memory_cached.html)

**NPU 形式名称**：torch_npu.npu.reset_max_memory_cached

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.reset_peak_memory_stats

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.reset_peak_memory_stats](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.reset_peak_memory_stats.html)

**NPU 形式名称**：torch_npu.npu.reset_peak_memory_stats

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.caching_allocator_alloc

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.caching_allocator_alloc](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.caching_allocator_alloc.html)

**NPU 形式名称**：torch_npu.npu.caching_allocator_alloc

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.cuda.caching_allocator_delete

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.caching_allocator_delete](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.caching_allocator_delete.html)

**NPU 形式名称**：torch_npu.npu.caching_allocator_delete

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.cuda.get_allocator_backend

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.get_allocator_backend](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.get_allocator_backend.html)

**NPU 形式名称**：torch_npu.npu.get_allocator_backend

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### <code><i>class</i></code> torch.cuda.CUDAPluggableAllocator

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.CUDAPluggableAllocator](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.CUDAPluggableAllocator.html)

**NPU 形式名称**：torch_npu.npu.NPUPluggableAllocator

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**： 该接口涉及高危操作，使用请参考《自定义API》中的“[torch_npu.npu.NPUPluggableAllocator](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-npu/torch-npu-npu-NPUPluggableAllocator.md)”章节。

</div>

### torch.cuda.change_current_allocator

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.change_current_allocator](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.change_current_allocator.html)

**NPU 形式名称**：torch_npu.npu.change_current_allocator

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**： 该接口涉及高危操作，使用请参考《自定义API》中的“[torch_npu.npu.change_current_allocator](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-npu/torch-npu-npu-change_current_allocator.md)”章节。

</div>

### <code><i>class</i></code> torch.cuda.memory.MemPool

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.memory.MemPool](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.MemPool.html)

**NPU 形式名称**：torch.npu.memory.MemPool

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**：

- `torch.npu.memory.MemPool`、`torch.npu.MemPool`和`torch_npu.npu.MemPool`功能一致。
- 函数原型为`MemPool(allocator=None, use_on_oom=False, no_split=False)`。`allocator`用于指定内存池使用的NPU内存分配器；`use_on_oom=True`表示内存池外的内存申请发生OOM时，可以将该内存池作为最后的内存分配来源；`no_split=True`表示不拆分该内存池中的内存段。
- `id`属性返回类型为`Tuple[int, int]`的内存池唯一标识。对于用户通过本接口创建的内存池，第一个`int`固定为0，用于与NPUGraph内部创建的内存池进行区分；第二个`int`是用户内存池的递增唯一编号，每创建一个新的用户内存池，该编号递增。
- `use_count()`返回内存池当前的引用计数，返回值类型为`int`。`MemPool`对象本身持有一个引用；进入`torch.npu.use_mem_pool`上下文后引用计数加1，退出上下文后引用计数减1。
- `snapshot()`返回根据当前内存池ID过滤后的NPU内存分配器状态快照，返回值类型为`list`。与原生PyTorch接口不同，当前NPU接口不支持`include_traces`参数。
- `torch.npu.use_mem_pool`仅将当前线程中的内存申请路由到指定内存池，在上下文中创建的新线程不会自动使用该内存池。回收内存池前，需要退出`torch.npu.use_mem_pool`上下文，并释放使用该内存池的Tensor。
- Ascend 950DT支持使用默认NPU缓存分配器创建内存池，不支持通过`allocator`参数指定`NPUPluggableAllocator`。

</div>

### torch.cuda.reset_accumulated_host_memory_stats

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.reset_accumulated_host_memory_stats](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.reset_accumulated_host_memory_stats.html)

**NPU 形式名称**：torch_npu.npu.reset_accumulated_host_memory_stats

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.cuda.reset_peak_host_memory_stats

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.reset_peak_host_memory_stats](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.reset_peak_host_memory_stats.html)

**NPU 形式名称**：torch_npu.npu.reset_peak_host_memory_stats

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.cuda.host_memory_stats_as_nested_dict

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.host_memory_stats_as_nested_dict](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.host_memory_stats_as_nested_dict.html)

**NPU 形式名称**：torch_npu.npu.host_memory_stats_as_nested_dict

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.cuda.host_memory_stats

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.host_memory_stats](https://pytorch.org/docs/2.11/generated/torch.cuda.memory.host_memory_stats.html)

**NPU 形式名称**：torch_npu.npu.host_memory_stats

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>
