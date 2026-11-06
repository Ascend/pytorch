# Distributed RPC Framework

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 本支持清单的范围以[PyTorch社区文档](https://pytorch.org/docs/2.9/)为准，社区文档中未列出的接口不在本清单范围内；已在社区文档中存在但未收录于本清单的API尚未验证，请谨慎使用，我们将在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.9/rpc.html)。

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

### torch.distributed.rpc.init_rpc

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.init_rpc](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.init_rpc)

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
- options绑定`NPUTensorPipeRpcBackendOptions`，需要`from torch_npu.distributed.rpc.options import NPUTensorPipeRpcBackendOptions`并设置options选项，参数格式和原生`TensorPipeRpcBackendOptions`相同。

</div>

### torch.distributed.rpc.rpc_sync

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.rpc_sync](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.rpc_sync)

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

**原生文档**：[torch.distributed.rpc.rpc_async](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.rpc_async)

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

**原生文档**：[torch.distributed.rpc.remote](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.remote)

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

### torch.distributed.rpc.get_worker_info

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.get_worker_info](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.get_worker_info)

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

### torch.distributed.rpc.shutdown

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.shutdown](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.shutdown)

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

### <code><i>class</i></code> torch.distributed.rpc.WorkerInfo

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.WorkerInfo](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.WorkerInfo)

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

> <font size="3">id</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.WorkerInfo.id](https://docs.pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.WorkerInfo.id)

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

> <font size="3">name</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.WorkerInfo.name](https://docs.pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.WorkerInfo.name)

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

</div>

### torch.distributed.rpc.functions.async_execution

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.functions.async_execution](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.functions.async_execution)

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

### <code><i>class</i></code> torch.distributed.rpc.BackendType

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.BackendType](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.BackendType)

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

### <code><i>class</i></code> torch.distributed.rpc.RpcBackendOptions

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.RpcBackendOptions](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.RpcBackendOptions)

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

> <font size="3">init_method</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.RpcBackendOptions.init\_method](https://docs.pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.RpcBackendOptions.init_method)

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id69 -->

</div>

> <font size="3">rpc_timeout</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.RpcBackendOptions.rpc\_timeout](https://docs.pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.RpcBackendOptions.rpc_timeout)

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

</div>

### <code><i>class</i></code> torch.distributed.rpc.TensorPipeRpcBackendOptions

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.TensorPipeRpcBackendOptions](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions)

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

> <font size="3">device_maps</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.TensorPipeRpcBackendOptions.device\_maps](https://docs.pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions.device_maps)

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

> <font size="3">devices</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.TensorPipeRpcBackendOptions.devices](https://docs.pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions.devices)

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

> <font size="3">num_worker_threads</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.TensorPipeRpcBackendOptions.num\_worker\_threads](https://docs.pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions.num_worker_threads)

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

> <font size="3">set_device_map()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.TensorPipeRpcBackendOptions.set_device_map](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions.set_device_map)

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

**限制与说明**：建议使用已适配的`torch_npu.distributed.rpc.options.NPUTensorPipeRpcBackendOptions.set_device_map`

</div>

> <font size="3">set_devices()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.TensorPipeRpcBackendOptions.set_devices](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions.set_devices)

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

**限制与说明**：建议使用已适配的`torch_npu.distributed.rpc.options.NPUTensorPipeRpcBackendOptions.set_devices`

</div>

</div>

## RRef

### <code><i>class</i></code> torch.distributed.rpc.PyRRef

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.PyRRef](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.rpc.PyRRef)

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

## RemoteModule

### <code><i>class</i></code> torch.distributed.nn.api.remote_module.RemoteModule

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.nn.api.remote_module.RemoteModule](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.nn.api.remote_module.RemoteModule)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id45 -->

> <font size="3">get_module_rref()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.nn.api.remote_module.RemoteModule.get_module_rref](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.nn.api.remote_module.RemoteModule.get_module_rref)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id48 -->

</div>

> <font size="3">remote_parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.nn.api.remote_module.RemoteModule.remote_parameters](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.nn.api.remote_module.RemoteModule.remote_parameters)

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

</div>

## Distributed Autograd Framework

### torch.distributed.autograd.backward

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.autograd.backward](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.autograd.backward)

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

### torch.distributed.autograd.context

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.autograd.context](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.autograd.context)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id57 -->

</div>

### torch.distributed.autograd.get_gradients

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.autograd.get_gradients](https://pytorch.org/docs/2.9/rpc.html#torch.distributed.autograd.get_gradients)

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
