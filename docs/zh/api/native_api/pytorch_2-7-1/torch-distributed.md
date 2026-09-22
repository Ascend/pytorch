# torch.distributed

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.7/distributed.html)。

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

**原生文档**：[torch.distributed.is_available](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.is_available)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id3 -->

</div>

### torch.distributed.is_mpi_available

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_mpi_available](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.is_mpi_available)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT</term>：支持
<!-- end id6 -->

</div>

### torch.distributed.is_nccl_available

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_nccl_available](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.is_nccl_available)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT</term>：支持
<!-- end id9 -->

</div>

### <code><i>class</i></code> torch.distributed.ProcessGroupNCCL

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.ProcessGroupNCCL](https://pytorch.org/docs/2.7/distributed.html)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT</term>：支持
<!-- end id12 -->

**限制与说明**：该接口仅在调用 `torch_npu.contrib.transfer_to_npu` 后支持。调用后，NCCL 进程组接口会映射到 `ProcessGroupHCCL`，实际通信由 HCCL 执行；未调用 `torch_npu.contrib.transfer_to_npu` 时，该接口不支持。

</div>

### torch.distributed.is_gloo_available

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_gloo_available](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.is_gloo_available)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT</term>：支持
<!-- end id15 -->

</div>

### torch.distributed.is_torchelastic_launched

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_torchelastic_launched](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.is_torchelastic_launched)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id18 -->

</div>

### torch.distributed.init_process_group

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.init_process_group](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.init_process_group)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT</term>：支持
<!-- end id21 -->

**限制与说明**： 当`pg_options`参数传入类型为`torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()`时，可通过配置该变量的`hccl_config`属性控制HCCL通信域缓冲区大小。具体示例可参考[通过pg_options配置HCCL通信域参数](../../../developer_notes/distributed/parameter_configuration/setting_HCCL_communicator_parameter.md)中的`hccl_buffer_size`配置说明。可通过配置变量`hccl_config`的`group_name`字段设置HCCL通信域的通信组自定义名称，取值为长度不超过32的字符串。

</div>

### torch.distributed.is_initialized

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_initialized](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.is_initialized)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id24 -->

</div>

## Post-Initialization

### torch.distributed.get_backend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_backend](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.get_backend)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT</term>：支持
<!-- end id27 -->

</div>

### torch.distributed.get_rank

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_rank](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.get_rank)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT</term>：支持
<!-- end id30 -->

</div>

### torch.distributed.get_world_size

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_world_size](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.get_world_size)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT</term>：支持
<!-- end id33 -->

</div>

### <code><i>class</i></code> torch.distributed.Backend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Backend](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Backend)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id36 -->

> <font size="3">register_backend()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Backend.register_backend](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Backend.register_backend)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id39 -->

</div>

</div>

## Groups

### torch.distributed.get_group_rank

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_group_rank](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.get_group_rank)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT</term>：支持
<!-- end id42 -->

</div>

### torch.distributed.get_global_rank

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_global_rank](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.get_global_rank)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT</term>：支持
<!-- end id45 -->

</div>

### torch.distributed.get_process_group_ranks

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_process_group_ranks](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.get_process_group_ranks)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT</term>：支持
<!-- end id48 -->

</div>

### torch.distributed.new_group

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.new_group](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.new_group)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT</term>：支持
<!-- end id51 -->

**限制与说明**： 当`pg_options`参数传入类型为`torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()`时，可通过配置该变量的`hccl_config`属性控制HCCL通信域缓冲区大小。具体示例可参考[通过pg_options配置HCCL通信域参数](../../../developer_notes/distributed/parameter_configuration/setting_HCCL_communicator_parameter.md)中的`hccl_buffer_size`配置说明。可通过配置变量`hccl_config`的`group_name`字段设置HCCL通信域的通信组自定义名称，取值为长度不超过32的字符串。

</div>

## DeviceMesh

### <code><i>class</i></code> torch.distributed.device_mesh.DeviceMesh

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.device_mesh.DeviceMesh)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT</term>：支持
<!-- end id54 -->

> <font size="3">from_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.from_group](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.device_mesh.DeviceMesh.from_group)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT</term>：支持
<!-- end id57 -->

</div>

> <font size="3">get_all_groups()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_all_groups](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_all_groups)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT</term>：支持
<!-- end id60 -->

</div>

> <font size="3">get_coordinate()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_coordinate](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_coordinate)

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT</term>：支持
<!-- end id63 -->

</div>

> <font size="3">get_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_group](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_group)

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT</term>：支持
<!-- end id66 -->

</div>

> <font size="3">get_local_rank()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_local_rank](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_local_rank)

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT</term>：支持
<!-- end id69 -->

</div>

> <font size="3">get_rank()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_rank](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_rank)

**产品支持情况**：

<!-- npu="910b" id70 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT</term>：支持
<!-- end id72 -->

</div>

</div>

## Point-to-point communication

### torch.distributed.batch_isend_irecv

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.batch_isend_irecv](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.batch_isend_irecv)

**产品支持情况**：

<!-- npu="910b" id73 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT</term>：支持
<!-- end id75 -->

**限制与说明**： `p2p_op_list`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.send

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.send](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.send)

**产品支持情况**：

<!-- npu="910b" id76 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT</term>：支持
<!-- end id78 -->

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.recv

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.recv](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.recv)

**产品支持情况**：

<!-- npu="910b" id79 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT</term>：支持
<!-- end id81 -->

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.isend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.isend](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.isend)

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT</term>：支持
<!-- end id84 -->

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.irecv

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.irecv](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.irecv)

**产品支持情况**：

<!-- npu="910b" id85 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT</term>：支持
<!-- end id87 -->

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### <code><i>class</i></code> torch.distributed.P2POp

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.P2POp](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.P2POp)

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT</term>：支持
<!-- end id90 -->

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

## Collective functions

### torch.distributed.all_gather_into_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_gather_into_tensor](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.all_gather_into_tensor)

**产品支持情况**：

<!-- npu="910b" id91 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT</term>：支持
<!-- end id93 -->

**限制与说明**：

- `output_tensor`仅支持bf16，fp16，fp32，int8，int32，bool
- `world_size`不支持3，5，6，7

</div>

### torch.distributed.reduce_scatter

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.reduce_scatter](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.reduce_scatter)

**产品支持情况**：

<!-- npu="910b" id94 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT</term>：支持
<!-- end id96 -->

**限制与说明**： `input_list`仅支持bf16，fp16，fp32，int8，int32

</div>

### torch.distributed.reduce_scatter_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.reduce_scatter_tensor](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.reduce_scatter_tensor)

**产品支持情况**：

<!-- npu="910b" id97 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT</term>：支持
<!-- end id99 -->

**限制与说明**：

- `output`仅支持bf16，fp16，fp32，int8，int32
- `world_size`不支持3，5，6，7

<!-- npu="910b" id100 -->
- 针对<term>Atlas A2 训练系列产品</term>，当前版本“prod”操作不支持int16、bf16数据类型
<!-- end id100 -->

</div>

### torch.distributed.all_to_all_single

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_to_all_single](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.all_to_all_single)

**产品支持情况**：

<!-- npu="910b" id101 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id101 -->
<!-- npu="A3" id102 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id102 -->
<!-- npu="950" id103 -->
- <term>Ascend 950DT</term>：支持
<!-- end id103 -->

**限制与说明**： `input`仅支持fp32

</div>

### torch.distributed.all_to_all

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_to_all](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.all_to_all)

**产品支持情况**：

<!-- npu="910b" id104 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id104 -->
<!-- npu="A3" id105 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id105 -->
<!-- npu="950" id106 -->
- <term>Ascend 950DT</term>：支持
<!-- end id106 -->

**限制与说明**：

- `input_tensor_list`仅支持fp32
- 通过设置`torch_npu.npu.use_compatible_impl(True)`，`torch.distributed.all_to_all`切换为与原生实现保持一致，例如：

  ```python
  import torch_npu
  torch_npu.npu.use_compatible_impl(True)
  ```

</div>

### torch.distributed.reduce_op

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.reduce_op](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.reduce_op)

**产品支持情况**：

<!-- npu="910b" id107 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id107 -->
<!-- npu="A3" id108 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id108 -->
<!-- npu="950" id109 -->
- <term>Ascend 950DT</term>：支持
<!-- end id109 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int32，int64

</div>

### torch.distributed.broadcast

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.broadcast](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.broadcast)

**产品支持情况**：

<!-- npu="910b" id110 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id110 -->
<!-- npu="A3" id111 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id111 -->
<!-- npu="950" id112 -->
- <term>Ascend 950DT</term>：支持
<!-- end id112 -->

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.broadcast_object_list

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.broadcast_object_list](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.broadcast_object_list)

**产品支持情况**：

<!-- npu="910b" id113 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id113 -->
<!-- npu="A3" id114 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id114 -->
<!-- npu="950" id115 -->
- <term>Ascend 950DT</term>：支持
<!-- end id115 -->

</div>

### torch.distributed.reduce

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.reduce](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.reduce)

**产品支持情况**：

<!-- npu="910b" id116 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id116 -->
<!-- npu="A3" id117 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id117 -->
<!-- npu="950" id118 -->
- <term>Ascend 950DT</term>：支持
<!-- end id118 -->

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，uint8，int8，int32，int64，bool

</div>

### torch.distributed.all_gather

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_gather](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.all_gather)

**产品支持情况**：

<!-- npu="910b" id119 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id119 -->
<!-- npu="A3" id120 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id120 -->
<!-- npu="950" id121 -->
- <term>Ascend 950DT</term>：支持
<!-- end id121 -->

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，int8，int32，bool

</div>

### torch.distributed.all_gather_object

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_gather_object](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.all_gather_object)

**产品支持情况**：

<!-- npu="910b" id122 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id122 -->
<!-- npu="A3" id123 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id123 -->
<!-- npu="950" id124 -->
- <term>Ascend 950DT</term>：支持
<!-- end id124 -->

</div>

### torch.distributed.gather

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.gather](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.gather)

**产品支持情况**：

<!-- npu="910b" id125 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id125 -->
<!-- npu="A3" id126 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id126 -->
<!-- npu="950" id127 -->
- <term>Ascend 950DT</term>：支持
<!-- end id127 -->

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

**原生文档**：[torch.distributed.gather_object](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.gather_object)

**产品支持情况**：

<!-- npu="910b" id128 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id128 -->
<!-- npu="A3" id129 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id129 -->
<!-- npu="950" id130 -->
- <term>Ascend 950DT</term>：支持
<!-- end id130 -->

**限制与说明**： 支持的输入类型为Python Object

</div>

### torch.distributed.scatter

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.scatter](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.scatter)

**产品支持情况**：

<!-- npu="910b" id131 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id131 -->
<!-- npu="A3" id132 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id132 -->
<!-- npu="950" id133 -->
- <term>Ascend 950DT</term>：支持
<!-- end id133 -->

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

**原生文档**：[torch.distributed.scatter_object_list](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.scatter_object_list)

**产品支持情况**：

<!-- npu="910b" id134 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id134 -->
<!-- npu="A3" id135 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id135 -->
<!-- npu="950" id136 -->
- <term>Ascend 950DT</term>：支持
<!-- end id136 -->

**限制与说明**： 不涉及`dtype`参数

</div>

### <code><i>class</i></code> torch.distributed.ReduceOp

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.ReduceOp](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.ReduceOp)

**产品支持情况**：

<!-- npu="910b" id137 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id137 -->
<!-- npu="A3" id138 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id138 -->
<!-- npu="950" id139 -->
- <term>Ascend 950DT</term>：支持
<!-- end id139 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int32，int64，bool

</div>

### torch.distributed.all_reduce

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_reduce](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.all_reduce)

**产品支持情况**：

<!-- npu="910b" id140 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id140 -->
<!-- npu="A3" id141 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id141 -->
<!-- npu="950" id142 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id142 -->

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，int32，int64，bool

</div>

### torch.distributed.barrier

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.barrier](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.barrier)

**产品支持情况**：

<!-- npu="910b" id143 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id143 -->
<!-- npu="A3" id144 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id144 -->
<!-- npu="950" id145 -->
- <term>Ascend 950DT</term>：支持
<!-- end id145 -->

</div>

### torch.distributed.monitored_barrier

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.monitored_barrier](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.monitored_barrier)

**产品支持情况**：

<!-- npu="910b" id146 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id146 -->
<!-- npu="A3" id147 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id147 -->
<!-- npu="950" id148 -->
- <term>Ascend 950DT</term>：支持
<!-- end id148 -->

</div>

## Distributed Key-Value Store

### <code><i>class</i></code> torch.distributed.PrefixStore

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.PrefixStore](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.PrefixStore)

**产品支持情况**：

<!-- npu="910b" id149 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id149 -->
<!-- npu="A3" id150 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id150 -->
<!-- npu="950" id151 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id151 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.PrefixStore.\_\_init\_\_](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.PrefixStore.__init__)

**产品支持情况**：

<!-- npu="910b" id152 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id152 -->
<!-- npu="A3" id153 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id153 -->
<!-- npu="950" id154 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id154 -->

</div>

> <font size="3">underlying_store()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.PrefixStore.underlying_store](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.PrefixStore.underlying_store)

**产品支持情况**：

<!-- npu="910b" id155 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id155 -->
<!-- npu="A3" id156 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id156 -->
<!-- npu="950" id157 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id157 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.Store

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store)

**产品支持情况**：

<!-- npu="910b" id158 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id158 -->
<!-- npu="A3" id159 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id159 -->
<!-- npu="950" id160 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id160 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.\_\_init\_\_](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.__init__)

**产品支持情况**：

<!-- npu="910b" id161 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id161 -->
<!-- npu="A3" id162 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id162 -->
<!-- npu="950" id163 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id163 -->

</div>

> <font size="3">set()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.set](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.set)

**产品支持情况**：

<!-- npu="910b" id164 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id164 -->
<!-- npu="A3" id165 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id165 -->
<!-- npu="950" id166 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id166 -->

</div>

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.get](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.get)

**产品支持情况**：

<!-- npu="910b" id167 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id167 -->
<!-- npu="A3" id168 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id168 -->
<!-- npu="950" id169 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id169 -->

</div>

> <font size="3">add()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.add](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.add)

**产品支持情况**：

<!-- npu="910b" id170 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id170 -->
<!-- npu="A3" id171 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id171 -->
<!-- npu="950" id172 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id172 -->

</div>

> <font size="3">compare_set()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.compare_set](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.compare_set)

**产品支持情况**：

<!-- npu="910b" id173 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id173 -->
<!-- npu="A3" id174 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id174 -->
<!-- npu="950" id175 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id175 -->

</div>

> <font size="3">wait()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.wait](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.wait)

**产品支持情况**：

<!-- npu="910b" id176 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id176 -->
<!-- npu="A3" id177 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id177 -->
<!-- npu="950" id178 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id178 -->

</div>

> <font size="3">num_keys()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.num_keys](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.num_keys)

**产品支持情况**：

<!-- npu="910b" id179 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id179 -->
<!-- npu="A3" id180 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id180 -->
<!-- npu="950" id181 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id181 -->

</div>

> <font size="3">delete_key()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.delete_key](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.delete_key)

**产品支持情况**：

<!-- npu="910b" id182 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id182 -->
<!-- npu="A3" id183 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id183 -->
<!-- npu="950" id184 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id184 -->

</div>

> <font size="3">set_timeout()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.set_timeout](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.set_timeout)

**产品支持情况**：

<!-- npu="910b" id185 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id185 -->
<!-- npu="A3" id186 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id186 -->
<!-- npu="950" id187 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id187 -->

</div>

> <font size="3">append()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.append](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.append)

**产品支持情况**：

<!-- npu="910b" id188 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id188 -->
<!-- npu="A3" id189 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id189 -->
<!-- npu="950" id190 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id190 -->

</div>

> <font size="3">check()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.check](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.check)

**产品支持情况**：

<!-- npu="910b" id191 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id191 -->
<!-- npu="A3" id192 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id192 -->
<!-- npu="950" id193 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id193 -->

</div>

> <font size="3">has_extended_api()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.has_extended_api](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.has_extended_api)

**产品支持情况**：

<!-- npu="910b" id194 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id194 -->
<!-- npu="A3" id195 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id195 -->
<!-- npu="950" id196 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id196 -->

</div>

> <font size="3">multi_set()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.multi_set](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.multi_set)

**产品支持情况**：

<!-- npu="910b" id197 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id197 -->
<!-- npu="A3" id198 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id198 -->
<!-- npu="950" id199 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id199 -->

</div>

> <font size="3">multi_get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.multi_get](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.multi_get)

**产品支持情况**：

<!-- npu="910b" id200 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id200 -->
<!-- npu="A3" id201 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id201 -->
<!-- npu="950" id202 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id202 -->

</div>

> <font size="3">timeout()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.timeout](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.timeout)

**产品支持情况**：

<!-- npu="910b" id203 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id203 -->
<!-- npu="A3" id204 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id204 -->
<!-- npu="950" id205 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id205 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.TCPStore

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.TCPStore)

**产品支持情况**：

<!-- npu="910b" id206 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id206 -->
<!-- npu="A3" id207 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id207 -->
<!-- npu="950" id208 -->
- <term>Ascend 950DT</term>：支持
<!-- end id208 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore.\_\_init\_\_](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.TCPStore.__init__)

**产品支持情况**：

<!-- npu="910b" id209 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id209 -->
<!-- npu="A3" id210 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id210 -->
<!-- npu="950" id211 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id211 -->

</div>

> <font size="3">host()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore.host](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.TCPStore.host)

**产品支持情况**：

<!-- npu="910b" id212 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id212 -->
<!-- npu="A3" id213 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id213 -->
<!-- npu="950" id214 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id214 -->

</div>

> <font size="3">libuvBackend()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore.libuvBackend](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.TCPStore.libuvBackend)

**产品支持情况**：

<!-- npu="910b" id215 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id215 -->
<!-- npu="A3" id216 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id216 -->
<!-- npu="950" id217 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id217 -->

</div>

> <font size="3">port()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore.port](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.TCPStore.port)

**产品支持情况**：

<!-- npu="910b" id218 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id218 -->
<!-- npu="A3" id219 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id219 -->
<!-- npu="950" id220 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id220 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.HashStore

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.HashStore](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.HashStore)

**产品支持情况**：

<!-- npu="910b" id221 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id221 -->
<!-- npu="A3" id222 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id222 -->
<!-- npu="950" id223 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id223 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.HashStore.\_\_init\_\_](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.HashStore.__init__)

**产品支持情况**：

<!-- npu="910b" id224 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id224 -->
<!-- npu="A3" id225 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id225 -->
<!-- npu="950" id226 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id226 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.FileStore

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.FileStore](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.FileStore)

**产品支持情况**：

<!-- npu="910b" id227 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id227 -->
<!-- npu="A3" id228 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id228 -->
<!-- npu="950" id229 -->
- <term>Ascend 950DT</term>：支持
<!-- end id229 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.FileStore.\_\_init\_\_](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.FileStore.__init__)

**产品支持情况**：

<!-- npu="910b" id230 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id230 -->
<!-- npu="A3" id231 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id231 -->
<!-- npu="950" id232 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id232 -->

</div>

> <font size="3">path()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.FileStore.path](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.FileStore.path)

**产品支持情况**：

<!-- npu="910b" id233 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id233 -->
<!-- npu="A3" id234 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id234 -->
<!-- npu="950" id235 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id235 -->

</div>

</div>

## Logging

### <code><i>class</i></code> torch.distributed.DistBackendError

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.DistBackendError](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.DistBackendError)

**产品支持情况**：

<!-- npu="910b" id236 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id236 -->
<!-- npu="A3" id237 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id237 -->
<!-- npu="950" id238 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id238 -->

</div>
