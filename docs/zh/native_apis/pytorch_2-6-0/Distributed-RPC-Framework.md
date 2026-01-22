# Distributed RPC Framework

> [!NOTE]  
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.distributed.rpc.init_rpc|是|NPU设备启用rpc时需要在init_rpc中进行特定的设置：backend绑定rpc.backend_registry.BackendType.NPU_TENSORPIPE；<br>options绑定NPUTensorPipeRpcBackendOptions，需要from torch_npu.distributed.rpc.options import NPUTensorPipeRpcBackendOptions并设置option选项，参数格式和原版TensorPipeRpcBackendOptions相同。|
|torch.distributed.rpc.rpc_sync|是|-|
|torch.distributed.rpc.rpc_async|是|-|
|torch.distributed.rpc.remote|是|-|
|torch.distributed.rpc.get_worker_info|是|-|
|torch.distributed.rpc.shutdown|是|-|
|torch.distributed.rpc.WorkerInfo|是|-|
|torch.distributed.rpc.functions.async_execution|是|-|
|torch.distributed.rpc.BackendType|是|-|
|torch.distributed.rpc.RpcBackendOptions|是|-|
|torch.distributed.rpc.TensorPipeRpcBackendOptions|是|建议使用已适配的torch.distributed.rpc.NPUTensorPipeRpcBackendOptions|
|torch.distributed.rpc.TensorPipeRpcBackendOptions.set_device_map|是|建议使用已适配的torch.distributed.rpc.NPUTensorPipeRpcBackendOptions.set_device_map|
|torch.distributed.rpc.TensorPipeRpcBackendOptions.set_devices|是|建议使用已适配的torch.distributed.rpc.NPUTensorPipeRpcBackendOptions.set_devices|
|torch.distributed.rpc.PyRRef|是|-|
|torch.distributed.nn.api.remote_module.RemoteModule|是|-|
|torch.distributed.nn.api.remote_module.RemoteModule.get_module_rref|是|-|
|torch.distributed.nn.api.remote_module.RemoteModule.remote_parameters|是|-|
|torch.distributed.autograd.backward|是|-|
|torch.distributed.autograd.context|是|-|
|torch.distributed.autograd.get_gradients|是|-|


