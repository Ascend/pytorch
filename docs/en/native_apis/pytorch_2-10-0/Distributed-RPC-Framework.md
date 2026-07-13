# Distributed RPC Framework

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-14T07:53:12.726Z pushedAt=2026-06-14T09:16:34.705Z -->

> [!NOTE] Note
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.distributed.rpc.init_rpc|Yes|When enabling rpc on NPU devices, specific settings are required in init_rpc: bind backend to rpc.backend_registry.BackendType.NPU_TENSORPIPE;<br>bind options to NPUTensorPipeRpcBackendOptions, which requires from torch_npu.distributed.rpc.options import NPUTensorPipeRpcBackendOptions and setting the option parameters. The parameter format is the same as the original TensorPipeRpcBackendOptions.|
|torch.distributed.rpc.rpc_sync|Yes|-|
|torch.distributed.rpc.rpc_async|Yes|-|
|torch.distributed.rpc.remote|Yes|-|
|torch.distributed.rpc.get_worker_info|Yes|-|
|torch.distributed.rpc.shutdown|Yes|-|
|torch.distributed.rpc.WorkerInfo|Yes|-|
|torch.distributed.rpc.functions.async_execution|Yes|-|
|torch.distributed.rpc.BackendType|Yes|-|
|torch.distributed.rpc.RpcBackendOptions|Yes|-|
|torch.distributed.rpc.TensorPipeRpcBackendOptions|Yes|It is recommended to use the adapted torch.distributed.rpc.NPUTensorPipeRpcBackendOptions|
|torch.distributed.rpc.TensorPipeRpcBackendOptions.set_device_map|Yes|It is recommended to use the adapted torch.distributed.rpc.NPUTensorPipeRpcBackendOptions.set_device_map|
|torch.distributed.rpc.TensorPipeRpcBackendOptions.set_devices|Yes|It is recommended to use the adapted torch.distributed.rpc.NPUTensorPipeRpcBackendOptions.set_devices|
|torch.distributed.rpc.PyRRef|Yes|-|
|torch.distributed.nn.api.remote_module.RemoteModule|Yes|-|
|torch.distributed.nn.api.remote_module.RemoteModule.get_module_rref|Yes|-|
|torch.distributed.nn.api.remote_module.RemoteModule.remote_parameters|Yes|-|
|torch.distributed.autograd.backward|Yes|-|
|torch.distributed.autograd.context|Yes|-|
|torch.distributed.autograd.get_gradients|Yes|-|
