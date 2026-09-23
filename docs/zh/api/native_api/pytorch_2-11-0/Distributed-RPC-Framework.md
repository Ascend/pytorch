# Distributed RPC Framework

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.11/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.11/rpc.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [RPC](#rpc)
- [RRef](#rref)
- [RemoteModule](#remotemodule)
- [Distributed Autograd Framework](#distributed-autograd-framework)

</div>

<div style="display:none;">

## &#8203;Distributed RPC Framework

</div>

## RPC

### torch.distributed.rpc.shutdown

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.shutdown](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.shutdown)

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

### torch.distributed.rpc.functions.async_execution

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.functions.async_execution](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.functions.async_execution)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id6 -->

</div>

### <code><i>class</i></code> torch.distributed.rpc.TensorPipeRpcBackendOptions

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.TensorPipeRpcBackendOptions](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions)

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

**限制与说明**：建议使用已适配的`torch_npu.distributed.rpc.options.NPUTensorPipeRpcBackendOptions`

> <font size="3">set_device_map()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.TensorPipeRpcBackendOptions.set_device_map](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions.set_device_map)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id12 -->

**限制与说明**：建议使用已适配的`torch_npu.distributed.rpc.options.NPUTensorPipeRpcBackendOptions.set_device_map`

</div>

> <font size="3">set_devices()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.TensorPipeRpcBackendOptions.set_devices](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions.set_devices)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id15 -->

**限制与说明**：建议使用已适配的`torch_npu.distributed.rpc.options.NPUTensorPipeRpcBackendOptions.set_devices`

</div>

</div>

### torch.distributed.rpc.init_rpc

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.init_rpc](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.init_rpc)

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

**限制与说明**：

- NPU设备启用rpc时需要在`init_rpc`中进行特定的设置：`backend`绑定`rpc.backend_registry.BackendType.NPU_TENSORPIPE`；
- options绑定`NPUTensorPipeRpcBackendOptions`，需要`from torch_npu.distributed.rpc.options import NPUTensorPipeRpcBackendOptions`并设置option选项，参数格式和原版`TensorPipeRpcBackendOptions`相同。

</div>

### torch.distributed.rpc.get_worker_info

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.get_worker_info](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.get_worker_info)

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

### <code><i>class</i></code> torch.distributed.rpc.WorkerInfo

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.WorkerInfo](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.WorkerInfo)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id24 -->

</div>

### <code><i>class</i></code> torch.distributed.rpc.BackendType

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.BackendType](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.BackendType)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id27 -->

</div>

### <code><i>class</i></code> torch.distributed.rpc.RpcBackendOptions

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.RpcBackendOptions](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.RpcBackendOptions)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id30 -->

</div>

### torch.distributed.rpc.rpc_sync

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.rpc_sync](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.rpc_sync)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id33 -->

</div>

### torch.distributed.rpc.rpc_async

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.rpc_async](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.rpc_async)

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

### torch.distributed.rpc.remote

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.remote](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.remote)

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

## RRef

### <code><i>class</i></code> torch.distributed.rpc.PyRRef

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.PyRRef](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.PyRRef)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id42 -->

</div>

## RemoteModule

### <code><i>class</i></code> torch.distributed.nn.api.remote_module.RemoteModule

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.nn.api.remote_module.RemoteModule](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.nn.api.remote_module.RemoteModule)

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

> <font size="3">get_module_rref()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.nn.api.remote_module.RemoteModule.get_module_rref](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.nn.api.remote_module.RemoteModule.get_module_rref)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id48 -->

</div>

> <font size="3">remote_parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.nn.api.remote_module.RemoteModule.remote_parameters](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.nn.api.remote_module.RemoteModule.remote_parameters)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id51 -->

</div>

</div>

## Distributed Autograd Framework

### torch.distributed.autograd.backward

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.autograd.backward](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.autograd.backward)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id54 -->

</div>

### torch.distributed.autograd.context

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.autograd.context](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.autograd.context)

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

### torch.distributed.autograd.get_gradients

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.autograd.get_gradients](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.autograd.get_gradients)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id60 -->

</div>
