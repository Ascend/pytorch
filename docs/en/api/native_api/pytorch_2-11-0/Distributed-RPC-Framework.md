# Distributed RPC Framework

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.11/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.11/rpc.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

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

**Native Documentation**: [torch.distributed.rpc.shutdown](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.shutdown)

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

### torch.distributed.rpc.functions.async_execution

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.rpc.functions.async_execution](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.functions.async_execution)

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

### <code><i>class</i></code> torch.distributed.rpc.TensorPipeRpcBackendOptions

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.rpc.TensorPipeRpcBackendOptions](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions)

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

**Constraints & Notes**: It is recommended to use the adapted torch_npu.distributed.rpc.options.NPUTensorPipeRpcBackendOptions.

> <font size="3">set_device_map()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.rpc.TensorPipeRpcBackendOptions.set_device_map](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions.set_device_map)

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

**Constraints & Notes**: It is recommended to use the adapted torch_npu.distributed.rpc.options.NPUTensorPipeRpcBackendOptions.set_device_map.

</div>

> <font size="3">set_devices()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.rpc.TensorPipeRpcBackendOptions.set_devices](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions.set_devices)

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

**Constraints & Notes**: It is recommended to use the adapted torch_npu.distributed.rpc.options.NPUTensorPipeRpcBackendOptions.set_devices.

</div>

</div>

### torch.distributed.rpc.init_rpc

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.rpc.init_rpc](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.init_rpc)

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

**Constraints & Notes**:

- When RPC is enabled on an NPU device, specific settings need to be configured in `init_rpc`: bind `backend` to rpc.backend_registry.BackendType.NPU_TENSORPIPE.
- `options` is bound to `NPUTensorPipeRpcBackendOptions`. You need to import `NPUTensorPipeRpcBackendOptions` from torch_npu.distributed.rpc.options and set the option. The parameter format is the same as that of the original `TensorPipeRpcBackendOptions`.

</div>

### torch.distributed.rpc.get_worker_info

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.rpc.get_worker_info](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.get_worker_info)

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

### <code><i>class</i></code> torch.distributed.rpc.WorkerInfo

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.rpc.WorkerInfo](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.WorkerInfo)

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

</div>

### <code><i>class</i></code> torch.distributed.rpc.BackendType

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.rpc.BackendType](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.BackendType)

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

### <code><i>class</i></code> torch.distributed.rpc.RpcBackendOptions

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.rpc.RpcBackendOptions](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.RpcBackendOptions)

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

</div>

### torch.distributed.rpc.rpc_sync

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.rpc.rpc_sync](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.rpc_sync)

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

### torch.distributed.rpc.rpc_async

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.rpc.rpc_async](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.rpc_async)

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

### torch.distributed.rpc.remote

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.rpc.remote](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.remote)

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

## RRef

### <code><i>class</i></code> torch.distributed.rpc.PyRRef

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.rpc.PyRRef](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.PyRRef)

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

## RemoteModule

### <code><i>class</i></code> torch.distributed.nn.api.remote_module.RemoteModule

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.nn.api.remote_module.RemoteModule](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.nn.api.remote_module.RemoteModule)

**Support Status**:

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id45 -->

> <font size="3">get_module_rref()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.nn.api.remote_module.RemoteModule.get_module_rref](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.nn.api.remote_module.RemoteModule.get_module_rref)

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

> <font size="3">remote_parameters()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.nn.api.remote_module.RemoteModule.remote_parameters](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.nn.api.remote_module.RemoteModule.remote_parameters)

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

</div>

## Distributed Autograd Framework

### torch.distributed.autograd.backward

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.autograd.backward](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.autograd.backward)

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

### torch.distributed.autograd.context

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.autograd.context](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.autograd.context)

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

### torch.distributed.autograd.get_gradients

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.autograd.get_gradients](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.autograd.get_gradients)

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
