# torch.distributed

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.12/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.12/distributed.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Initialization](#initialization)
- [Post-Initialization](#post-initialization)
- [Groups](#groups)
- [DeviceMesh](#devicemesh)
- [Point-to-point communication](#point-to-point-communication)
- [Collective functions](#collective-functions)
- [Distributed Key-Value Store](#distributed-key-value-store)
- [Logging](#logging)

</div>

<div style="display:none;">

## &#8203;torch.distributed

</div>

## Initialization

### torch.distributed.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_available](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.is_available)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.distributed.is_mpi_available

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_mpi_available](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.is_mpi_available)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### torch.distributed.is_nccl_available

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_nccl_available](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.is_nccl_available)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.distributed.ProcessGroupNCCL

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.ProcessGroupNCCL](https://pytorch.org/docs/2.12/distributed.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：该接口仅在调用 `torch_npu.contrib.transfer_to_npu` 后支持。调用后，NCCL 进程组接口会映射到 `ProcessGroupHCCL`，实际通信由 HCCL 执行；未调用 `torch_npu.contrib.transfer_to_npu` 时，该接口不支持。

</div>

### torch.distributed.is_gloo_available

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_gloo_available](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.is_gloo_available)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### torch.distributed.is_torchelastic_launched

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_torchelastic_launched](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.is_torchelastic_launched)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.distributed.init_process_group

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.init_process_group](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.init_process_group)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 当`pg_options`参数传入类型为`torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()`时，可通过配置该变量的`hccl_config`属性控制HCCL通信域缓冲区大小。具体示例可参考《PyTorch 训练模型迁移调优指南》的“[hccl_buffer_size](https://www.hiascend.com/document/detail/zh/ModelZoo/traditional_model_train/PyTorch/docs/zh/performance_tuning/performance_tuning_methods/communication_basics_overview.md#hccl_buffer_size)”章节。可通过配置变量`hccl_config`的`group_name`字段设置HCCL通信域的通信组自定义名称，取值为长度不超过32的字符串。

</div>

### torch.distributed.is_initialized

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_initialized](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.is_initialized)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

## Post-Initialization

### torch.distributed.get_backend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_backend](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.get_backend)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### torch.distributed.get_rank

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_rank](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.get_rank)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### torch.distributed.get_world_size

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_world_size](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.get_world_size)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.distributed.Backend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Backend](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.Backend)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">register_backend()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Backend.register_backend](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.Backend.register_backend)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## Groups

### torch.distributed.get_group_rank

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_group_rank](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.get_group_rank)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### torch.distributed.get_global_rank

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_global_rank](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.get_global_rank)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### torch.distributed.get_process_group_ranks

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_process_group_ranks](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.get_process_group_ranks)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### torch.distributed.new_group

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.new_group](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.new_group)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 当`pg_options`参数传入类型为`torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()`时，可通过配置该变量的`hccl_config`属性控制HCCL通信域缓冲区大小。具体示例可参考《PyTorch 训练模型迁移调优指南》的“[hccl_buffer_size](https://www.hiascend.com/document/detail/zh/ModelZoo/traditional_model_train/PyTorch/docs/zh/performance_tuning/performance_tuning_methods/communication_basics_overview.md#hccl_buffer_size)”章节。可通过配置变量`hccl_config`的`group_name`字段设置HCCL通信域的通信组自定义名称，取值为长度不超过32的字符串。

</div>

## DeviceMesh

### <code><i>class</i></code> torch.distributed.device_mesh.DeviceMesh

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.device_mesh.DeviceMesh)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">from_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.from_group](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.device_mesh.DeviceMesh.from_group)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">get_all_groups()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_all_groups](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_all_groups)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">get_coordinate()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_coordinate](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_coordinate)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">get_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_group](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_group)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">get_local_rank()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_local_rank](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_local_rank)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">get_rank()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_rank](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_rank)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## Point-to-point communication

### torch.distributed.batch_isend_irecv

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.batch_isend_irecv](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.batch_isend_irecv)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### <code><i>class</i></code> torch.distributed.P2POp

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.P2POp](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.P2POp)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.send

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.send](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.send)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.recv

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.recv](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.recv)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.isend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.isend](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.isend)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.irecv

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.irecv](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.irecv)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

## Collective functions

### torch.distributed.all_gather_into_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_gather_into_tensor](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.all_gather_into_tensor)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `output_tensor`仅支持bf16，fp16，fp32，int8，int32，bool
- `world_size`不支持3，5，6，7

</div>

### torch.distributed.reduce_scatter

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.reduce_scatter](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.reduce_scatter)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `output`仅支持bf16，fp16，fp32，int8，int32

</div>

### torch.distributed.reduce_scatter_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.reduce_scatter_tensor](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.reduce_scatter_tensor)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `output`仅支持bf16，fp16，fp32，int8，int32
- `world_size`不支持3，5，6，7
- 针对<term>Atlas A2 训练系列产品</term>，当前版本“prod”操作不支持int16、bf16数据类型

</div>

### torch.distributed.all_to_all_single

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_to_all_single](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.all_to_all_single)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `output`仅支持fp32

</div>

### torch.distributed.all_to_all

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_to_all](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.all_to_all)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `input`仅支持fp32
- 通过设置`torch_npu.npu.use_compatible_impl(True)`，`torch.distributed.all_to_all`切换为与原生实现保持一致，例如：

  ```python
  import torch_npu
  torch_npu.npu.use_compatible_impl(True)
  ```

</div>

### torch.distributed.reduce_op

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.reduce_op](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.reduce_op)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int32，int64

</div>

### torch.distributed.broadcast

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.broadcast](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.broadcast)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.broadcast_object_list

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.broadcast_object_list](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.broadcast_object_list)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### torch.distributed.reduce

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.reduce](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.reduce)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，uint8，int8，int32，int64，bool

</div>

### torch.distributed.all_gather

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_gather](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.all_gather)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，int8，int32，bool

</div>

### torch.distributed.all_gather_object

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_gather_object](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.all_gather_object)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### torch.distributed.gather

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.gather](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.gather)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `tensor`仅支持bf16，fp16，fp32，int8，int32，bool
- 通过设置`torch_npu.npu.use_compatible_impl(True)`，`torch.distributed.gather`切换为与原生实现保持一致，例如：

  ```python
  import torch_npu
  torch_npu.npu.use_compatible_impl(True)
  ```

</div>

### torch.distributed.gather_object

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.gather_object](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.gather_object)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 支持的输入类型为Python Object

</div>

### torch.distributed.scatter

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.scatter](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.scatter)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- 通过设置`torch_npu.npu.use_compatible_impl(True)`，`torch.distributed.scatter`切换为与原生实现保持一致，例如：

  ```python
  import torch_npu
  torch_npu.npu.use_compatible_impl(True)
  ```

</div>

### torch.distributed.scatter_object_list

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.scatter_object_list](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.scatter_object_list)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 不涉及`dtype`参数

</div>

### <code><i>class</i></code> torch.distributed.ReduceOp

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.ReduceOp](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.ReduceOp)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int32，int64，bool

</div>

### torch.distributed.all_reduce

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_reduce](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.all_reduce)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，int32，int64，bool

</div>

### torch.distributed.barrier

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.barrier](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.barrier)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### torch.distributed.monitored_barrier

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.monitored_barrier](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.monitored_barrier)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

## Distributed Key-Value Store

### <code><i>class</i></code> torch.distributed.PrefixStore

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.PrefixStore](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.PrefixStore)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.PrefixStore.\_\_init\_\_](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.PrefixStore.\_\_init\_\_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">underlying_store()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.PrefixStore.underlying_store](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.PrefixStore.underlying_store)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.distributed.Store

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.Store)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.\_\_init\_\_](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.Store.\_\_init\_\_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">set()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.set](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.Store.set)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.get](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.Store.get)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">add()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.add](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.Store.add)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">compare_set()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.compare_set](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.Store.compare_set)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">wait()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.wait](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.Store.wait)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">num_keys()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.num_keys](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.Store.num_keys)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">delete_key()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.delete_key](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.Store.delete_key)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">set_timeout()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.set_timeout](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.Store.set_timeout)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">append()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.append](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.Store.append)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">check()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.check](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.Store.check)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_extended_api()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.has_extended_api](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.Store.has_extended_api)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">multi_set()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.multi_set](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.Store.multi_set)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">multi_get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.multi_get](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.Store.multi_get)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">timeout()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.timeout](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.Store.timeout)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.distributed.TCPStore

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.TCPStore)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore.\_\_init\_\_](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.TCPStore.__init__)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">host()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore.host](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.TCPStore.host)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">libuvBackend()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore.libuvBackend](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.TCPStore.libuvBackend)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">port()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore.port](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.TCPStore.port)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.distributed.HashStore

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.HashStore](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.HashStore)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.HashStore.\_\_init\_\_](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.HashStore.__init__)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.distributed.FileStore

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.FileStore](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.FileStore)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.FileStore.\_\_init\_\_](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.FileStore.__init__)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">path()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.FileStore.path](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.FileStore.path)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## Logging

### <code><i>class</i></code> torch.distributed.DistBackendError

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.DistBackendError](https://pytorch.org/docs/2.12/distributed.html#torch.distributed.DistBackendError)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>
