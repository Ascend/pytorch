# torch.distributed

> [!NOTE]  
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)
- [Backends](#backends)
- [Post-Initialization](#post-initialization)
- [Initialization](#initialization)
- [Distributed Key-Value Store](#distributed-key-value-store)
- [Groups](#groups)
- [Point-to-point communication](#point-to-point-communication)
- [Collective functions](#collective-functions)
- [Debuggingtorch.distributedapplications](#debuggingtorchdistributedapplications)

## base API

### torch.distributed.is_available

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_available](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.is_available)

**是否支持**：是

</div>

### torch.distributed.is_mpi_available

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_mpi_available](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.is_mpi_available)

**是否支持**：是

</div>

### torch.distributed.is_nccl_available

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_nccl_available](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.is_nccl_available)

**是否支持**：是

</div>

### torch.distributed.is_gloo_available

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_gloo_available](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.is_gloo_available)

**是否支持**：是

</div>

### torch.distributed.is_torchelastic_launched

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_torchelastic_launched](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.is_torchelastic_launched)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.distributed.get_backend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_backend](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.get_backend)

**是否支持**：是

</div>

### torch.distributed.get_rank

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_rank](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.get_rank)

**是否支持**：是

</div>

### torch.distributed.get_world_size

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_world_size](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.get_world_size)

**是否支持**：是

</div>

### _`class`_ torch.distributed.PrefixStore

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.PrefixStore](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.PrefixStore)

**是否支持**：是

> <font size="3">__init__()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.PrefixStore.__init__](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.PrefixStore.__init__)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">underlying_store()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.PrefixStore.underlying_store](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.PrefixStore.underlying_store)

**是否支持**：是

</div>

</div>

### torch.distributed.get_group_rank

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_group_rank](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.get_group_rank)

**是否支持**：是

</div>

### torch.distributed.get_global_rank

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_global_rank](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.get_global_rank)

**是否支持**：是

</div>

### torch.distributed.get_process_group_ranks

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_process_group_ranks](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.get_process_group_ranks)

**是否支持**：是

</div>

### torch.distributed.batch_isend_irecv

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.batch_isend_irecv](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.batch_isend_irecv)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### _`class`_ torch.distributed.P2POp

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.P2POp](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.P2POp)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.all_gather_into_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_gather_into_tensor](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.all_gather_into_tensor)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，int8，int32，bool
- world size不支持3，5，6，7

</div>

### torch.distributed.reduce_scatter

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.reduce_scatter](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.reduce_scatter)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int8，int32

</div>

### torch.distributed.reduce_scatter_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.reduce_scatter_tensor](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.reduce_scatter_tensor)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，int8，int32
- world size不支持3，5，6，7
- 针对<term>Atlas A2 训练系列产品</term>，当前版本“prod”操作不支持int16、bf16数据类型

</div>

### torch.distributed.all_to_all_single

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_to_all_single](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.all_to_all_single)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

### torch.distributed.all_to_all

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_to_all](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.all_to_all)

**是否支持**：是

**限制与说明**：

- 支持fp32
- 通过设置torch_npu.npu.use_compatible_impl(True)，torch.distributed.all_to_all切换为与原生实现保持一致

</div>

### torch.distributed.reduce_op

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.reduce_op](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.reduce_op)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int32，int64

</div>

### _`class`_ torch.distributed.DistBackendError

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.DistBackendError](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.DistBackendError)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## Backends

### torch.distributed.init_process_group

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.init_process_group](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.init_process_group)

**是否支持**：是

**限制与说明**： 当pg_options参数传入类型为torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()时，配置该变量属性hccl_config可控制HCCL通信域缓冲区大小。具体示例可参考《PyTorch 训练模型迁移调优指南》的“[hccl_buffer_size](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/hccl_buffer_size.md)”章节。配置变量属性hccl_config的group_name字段可以设置HCCL通信域的通信组自定义名称，取值为长度不超过32的字符串。

</div>

## Post-Initialization

### torch.distributed.is_initialized

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_initialized](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.is_initialized)

**是否支持**：是

</div>

## Initialization

### _`class`_ torch.distributed.Backend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Backend](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.Backend)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">register_backend()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Backend.register_backend](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.Backend.register_backend)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.distributed.Store

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.Store)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">__init__()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.__init__](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.Store.__init__)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.set](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.Store.set)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.get](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.Store.get)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">add()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.add](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.Store.add)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">compare_set()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.compare_set](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.Store.compare_set)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">wait()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.wait](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.Store.wait)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">num_keys()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.num_keys](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.Store.num_keys)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">delete_key()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.delete_key](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.Store.delete_key)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_timeout()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.set_timeout](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.Store.set_timeout)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">append()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.append](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.Store.append)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">check()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.check](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.Store.check)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_extended_api()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.has_extended_api](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.Store.has_extended_api)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">multi_set()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.multi_set](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.Store.multi_set)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">multi_get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.multi_get](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.Store.multi_get)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">timeout()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.timeout](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.Store.timeout)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.distributed.device_mesh.DeviceMesh

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.device_mesh.DeviceMesh)

**是否支持**：是

> <font size="3">from_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.from_group](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.device_mesh.DeviceMesh.from_group)

**是否支持**：是

</div>

> <font size="3">get_all_groups()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_all_groups](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_all_groups)

**是否支持**：是

</div>

> <font size="3">get_coordinate()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_coordinate](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_coordinate)

**是否支持**：是

</div>

> <font size="3">get_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_group](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_group)

**是否支持**：是

</div>

> <font size="3">get_local_rank()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_local_rank](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_local_rank)

**是否支持**：是

</div>

> <font size="3">get_rank()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_rank](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_rank)

**是否支持**：是

</div>

</div>

## Distributed Key-Value Store

### _`class`_ torch.distributed.TCPStore

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.TCPStore)

**是否支持**：是

> <font size="3">__init__()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore.__init__](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.TCPStore.__init__)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">host()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore.host](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.TCPStore.host)

**是否支持**：是

</div>

> <font size="3">libuvBackend()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore.libuvBackend](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.TCPStore.libuvBackend)

**是否支持**：是

</div>

> <font size="3">port()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore.port](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.TCPStore.port)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributed.HashStore

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.HashStore](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.HashStore)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">__init__()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.HashStore.__init__](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.HashStore.__init__)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.distributed.FileStore

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.FileStore](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.FileStore)

**是否支持**：是

> <font size="3">__init__()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.FileStore.__init__](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.FileStore.__init__)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">path()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.FileStore.path](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.FileStore.path)

**是否支持**：是

</div>

</div>

## Groups

### torch.distributed.new_group

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.new_group](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.new_group)

**是否支持**：是

**限制与说明**： 当pg_options参数传入类型为torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()时，配置该变量属性hccl_config可控制HCCL通信域缓冲区大小。具体示例可参考《PyTorch 训练模型迁移调优指南》的“[hccl_buffer_size](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/hccl_buffer_size.md)”章节。配置变量属性hccl_config的group_name字段可以设置HCCL通信域的通信组自定义名称，取值为长度不超过32的字符串。

</div>

## Point-to-point communication

### torch.distributed.send

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.send](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.send)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.recv

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.recv](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.recv)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.isend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.isend](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.isend)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.irecv

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.irecv](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.irecv)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

## Collective functions

### torch.distributed.broadcast

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.broadcast](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.broadcast)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.broadcast_object_list

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.broadcast_object_list](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.broadcast_object_list)

**是否支持**：是

</div>

### torch.distributed.reduce

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.reduce](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.reduce)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int32，int64，bool

</div>

### torch.distributed.all_gather

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_gather](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.all_gather)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，int8，int32，bool

</div>

### torch.distributed.all_gather_object

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_gather_object](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.all_gather_object)

**是否支持**：是

</div>

### torch.distributed.gather

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.gather](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.gather)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，int8，int32，bool
- 通过设置torch_npu.npu.use_compatible_impl(True)，torch.distributed.gather切换为与原生实现保持一致

</div>

### torch.distributed.gather_object

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.gather_object](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.gather_object)

**是否支持**：是

**限制与说明**： 支持的输入类型为Python Object对象

</div>

### torch.distributed.scatter

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.scatter](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.scatter)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool
- 通过设置torch_npu.npu.use_compatible_impl(True)，torch.distributed.scatter切换为与原生实现保持一致

</div>

### torch.distributed.scatter_object_list

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.scatter_object_list](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.scatter_object_list)

**是否支持**：是

**限制与说明**： 不涉及dtype参数

</div>

### _`class`_ torch.distributed.ReduceOp

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.ReduceOp](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.ReduceOp)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32，uint8，int8，int32，int64，bool

</div>

## Debuggingtorch.distributedapplications

### torch.distributed.all_reduce

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_reduce](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.all_reduce)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，int32，int64，bool

</div>

### torch.distributed.barrier

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.barrier](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.barrier)

**是否支持**：是

</div>

### torch.distributed.monitored_barrier

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.monitored_barrier](https://pytorch.org/docs/2.11/distributed.html#torch.distributed.monitored_barrier)

**是否支持**：是

</div>
