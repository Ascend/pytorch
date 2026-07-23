# Distributed RPC Framework

> [!NOTE]  
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)
- [RPC](#rpc)
- [Basics](#basics)
- [Distributed Autograd Framework](#distributed-autograd-framework)
- [RRef](#rref)

## base API

### torch.distributed.rpc.shutdown

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.shutdown](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.shutdown)

**是否支持**：是

</div>

### torch.distributed.rpc.functions.async_execution

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.functions.async_execution](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.functions.async_execution)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.distributed.rpc.TensorPipeRpcBackendOptions

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.TensorPipeRpcBackendOptions](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions)

**是否支持**：是

**限制与说明**： 建议使用已适配的torch_npu.distributed.rpc.options.NPUTensorPipeRpcBackendOptions

> <font size="3">set_device_map()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.TensorPipeRpcBackendOptions.set_device_map](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions.set_device_map)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 建议使用已适配的torch_npu.distributed.rpc.options.NPUTensorPipeRpcBackendOptions.set_device_map

</div>

> <font size="3">set_devices()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.TensorPipeRpcBackendOptions.set_devices](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions.set_devices)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 建议使用已适配的torch_npu.distributed.rpc.options.NPUTensorPipeRpcBackendOptions.set_devices

</div>

</div>

### _`class`_ torch.distributed.rpc.PyRRef

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.PyRRef](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.PyRRef)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.distributed.nn.api.remote_module.RemoteModule

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.nn.api.remote_module.RemoteModule](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.nn.api.remote_module.RemoteModule)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">get_module_rref()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.nn.api.remote_module.RemoteModule.get_module_rref](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.nn.api.remote_module.RemoteModule.get_module_rref)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">remote_parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.nn.api.remote_module.RemoteModule.remote_parameters](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.nn.api.remote_module.RemoteModule.remote_parameters)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

## RPC

### torch.distributed.rpc.init_rpc

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.init_rpc](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.init_rpc)

**是否支持**：是

**限制与说明**：

- NPU设备启用rpc时需要在init_rpc中进行特定的设置：backend绑定rpc.backend_registry.BackendType.NPU_TENSORPIPE；
- options绑定NPUTensorPipeRpcBackendOptions，需要from torch_npu.distributed.rpc.options import NPUTensorPipeRpcBackendOptions并设置option选项，参数格式和原版TensorPipeRpcBackendOptions相同。

</div>

### torch.distributed.rpc.get_worker_info

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.get_worker_info](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.get_worker_info)

**是否支持**：是

</div>

### _`class`_ torch.distributed.rpc.WorkerInfo

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.WorkerInfo](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.WorkerInfo)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.distributed.rpc.BackendType

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.BackendType](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.BackendType)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.distributed.rpc.RpcBackendOptions

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.RpcBackendOptions](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.RpcBackendOptions)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## Basics

### torch.distributed.rpc.rpc_sync

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.rpc_sync](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.rpc_sync)

**是否支持**：是

</div>

### torch.distributed.rpc.rpc_async

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.rpc_async](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.rpc_async)

**是否支持**：是

</div>

### torch.distributed.rpc.remote

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.rpc.remote](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.rpc.remote)

**是否支持**：是

</div>

## Distributed Autograd Framework

### torch.distributed.autograd.backward

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.autograd.backward](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.autograd.backward)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.distributed.autograd.context

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.autograd.context](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.autograd.context)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## RRef

### torch.distributed.autograd.get_gradients

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.autograd.get_gradients](https://pytorch.org/docs/2.11/rpc.html#torch.distributed.autograd.get_gradients)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>
