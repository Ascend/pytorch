# Distributed RPC Framework

> [!NOTE]
>
> - API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.9/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。

## 目录

- [base API](#base-api)
- [RPC](#rpc)
- [Basics](#basics)
- [Distributed Autograd Framework](#distributed-autograd-framework)
- [RRef](#rref)

## base API

### torch.distributed.rpc.shutdown

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### torch.distributed.rpc.functions.async_execution

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### _`class`_ torch.distributed.rpc.TensorPipeRpcBackendOptions

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**：建议使用已适配的torch_npu.distributed.rpc.options.NPUTensorPipeRpcBackendOptions

> <font size="3">set_device_map()</font>

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：建议使用已适配的torch_npu.distributed.rpc.options.NPUTensorPipeRpcBackendOptions.set_device_map

</div>

> <font size="3">set_devices()</font>

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：建议使用已适配的torch_npu.distributed.rpc.options.NPUTensorPipeRpcBackendOptions.set_devices

</div>

</div>

### _`class`_ torch.distributed.rpc.PyRRef

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### _`class`_ torch.distributed.nn.api.remote_module.RemoteModule

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

> <font size="3">get_module_rref()</font>

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">remote_parameters()</font>

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

</div>

## RPC

### torch.distributed.rpc.init_rpc

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**：

- NPU设备启用rpc时需要在`init_rpc`中进行特定的设置：`backend`绑定rpc.backend_registry.BackendType.NPU_TENSORPIPE；
- options绑定`NPUTensorPipeRpcBackendOptions`，需要from torch_npu.distributed.rpc.options import `NPUTensorPipeRpcBackendOptions`并设置option选项，参数格式和原版`TensorPipeRpcBackendOptions`相同。

</div>

### torch.distributed.rpc.get_worker_info

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### _`class`_ torch.distributed.rpc.WorkerInfo

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### _`class`_ torch.distributed.rpc.BackendType

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### _`class`_ torch.distributed.rpc.RpcBackendOptions

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

## Basics

### torch.distributed.rpc.rpc_sync

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### torch.distributed.rpc.rpc_async

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### torch.distributed.rpc.remote

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

## Distributed Autograd Framework

### torch.distributed.autograd.backward

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.distributed.autograd.context

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

## RRef

### torch.distributed.autograd.get_gradients

<div style="margin-left: 2em">

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>
