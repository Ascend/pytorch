# torch.distributed

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.13/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.13/distributed.html)。

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

**原生文档**：[torch.distributed.is_available](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.is_available)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id3 -->

</div>

### torch.distributed.is_mpi_available

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_mpi_available](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.is_mpi_available)

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

### torch.distributed.is_nccl_available

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_nccl_available](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.is_nccl_available)

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

</div>

### <code><i>class</i></code> torch.distributed.ProcessGroupNCCL

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.ProcessGroupNCCL](https://pytorch.org/docs/2.13/distributed.html)

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

**限制与说明**：该接口仅在调用 `torch_npu.contrib.transfer_to_npu` 后支持。调用后，NCCL 进程组接口会映射到 `ProcessGroupHCCL`，实际通信由 HCCL 执行；未调用 `torch_npu.contrib.transfer_to_npu` 时，该接口不支持。

</div>

### torch.distributed.is_gloo_available

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_gloo_available](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.is_gloo_available)

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

</div>

### torch.distributed.is_torchelastic_launched

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_torchelastic_launched](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.is_torchelastic_launched)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id18 -->

</div>

### torch.distributed.init_process_group

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.init_process_group](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.init_process_group)

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

**限制与说明**： 当`pg_options`参数传入类型为`torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()`时，可通过配置该变量的`hccl_config`属性控制HCCL通信域缓冲区大小。具体示例可参考[通过pg_options配置HCCL通信域参数](../../../developer_notes/distributed/parameter_configuration/setting_HCCL_communicator_parameter.md)中的`hccl_buffer_size`配置说明。可通过配置变量`hccl_config`的`group_name`字段设置HCCL通信域的通信组自定义名称，取值为长度不超过32的字符串。

</div>

### torch.distributed.is_initialized

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.is_initialized](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.is_initialized)

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

## Post-Initialization

### torch.distributed.get_backend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_backend](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.get_backend)

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

### torch.distributed.get_debug_level

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_debug_level](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.get_debug_level)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.distributed.get_rank

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_rank](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.get_rank)

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

</div>

### torch.distributed.get_world_size

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_world_size](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.get_world_size)

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

### <code><i>class</i></code> torch.distributed.Backend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Backend](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Backend)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id36 -->

> <font size="3">register_backend()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Backend.register_backend](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Backend.register_backend)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id39 -->

</div>

</div>

## Groups

### torch.distributed.get_group_rank

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_group_rank](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.get_group_rank)

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

### torch.distributed.get_global_rank

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_global_rank](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.get_global_rank)

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

</div>

### torch.distributed.get_process_group_ranks

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.get_process_group_ranks](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.get_process_group_ranks)

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

### torch.distributed.new_group

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.new_group](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.new_group)

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

**限制与说明**： 当`pg_options`参数传入类型为`torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()`时，可通过配置该变量的`hccl_config`属性控制HCCL通信域缓冲区大小。具体示例可参考[通过pg_options配置HCCL通信域参数](../../../developer_notes/distributed/parameter_configuration/setting_HCCL_communicator_parameter.md)中的`hccl_buffer_size`配置说明。可通过配置变量`hccl_config`的`group_name`字段设置HCCL通信域的通信组自定义名称，取值为长度不超过32的字符串。

</div>

## DeviceMesh

### <code><i>class</i></code> torch.distributed.device_mesh.DeviceMesh

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.device_mesh.DeviceMesh)

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

> <font size="3">from_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.from_group](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.device_mesh.DeviceMesh.from_group)

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

> <font size="3">get_all_groups()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_all_groups](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_all_groups)

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

> <font size="3">get_coordinate()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_coordinate](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_coordinate)

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

> <font size="3">get_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_group](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_group)

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

> <font size="3">get_local_rank()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_local_rank](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_local_rank)

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

> <font size="3">get_rank()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.device_mesh.DeviceMesh.get_rank](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_rank)

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

## Point-to-point communication

### torch.distributed.batch_isend_irecv

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.batch_isend_irecv](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.batch_isend_irecv)

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

**限制与说明**： `p2p_op_list`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### <code><i>class</i></code> torch.distributed.P2POp

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.P2POp](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.P2POp)

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

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.send

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.send](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.send)

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

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.recv

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.recv](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.recv)

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id84 -->

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.isend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.isend](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.isend)

**产品支持情况**：

<!-- npu="910b" id85 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id87 -->

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.irecv

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.irecv](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.irecv)

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id90 -->

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

## Collective functions

### torch.distributed.reduce_scatter

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.reduce_scatter](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.reduce_scatter)

**产品支持情况**：

<!-- npu="910b" id91 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id93 -->

**限制与说明**： `input_list`仅支持bf16，fp16，fp32，int8，int32

</div>

### torch.distributed.all_to_all_single

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_to_all_single](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.all_to_all_single)

**产品支持情况**：

<!-- npu="910b" id94 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id96 -->

**限制与说明**： `input`仅支持fp32

</div>

### torch.distributed.all_to_all

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_to_all](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.all_to_all)

**产品支持情况**：

<!-- npu="910b" id97 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id99 -->

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

**原生文档**：[torch.distributed.reduce_op](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.reduce_op)

**产品支持情况**：

<!-- npu="910b" id100 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id102 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int32，int64

</div>

### torch.distributed.broadcast

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.broadcast](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.broadcast)

**产品支持情况**：

<!-- npu="910b" id103 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id105 -->

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

### torch.distributed.broadcast_object_list

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.broadcast_object_list](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.broadcast_object_list)

**产品支持情况**：

<!-- npu="910b" id106 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id108 -->

</div>

### torch.distributed.reduce

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.reduce](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.reduce)

**产品支持情况**：

<!-- npu="910b" id109 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id111 -->

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，uint8，int8，int32，int64，bool

</div>

### torch.distributed.all_gather

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_gather](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.all_gather)

**产品支持情况**：

<!-- npu="910b" id112 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id114 -->

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，int8，int32，bool

</div>

### torch.distributed.all_gather_object

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_gather_object](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.all_gather_object)

**产品支持情况**：

<!-- npu="910b" id115 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id117 -->

</div>

### torch.distributed.gather

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.gather](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.gather)

**产品支持情况**：

<!-- npu="910b" id118 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id120 -->

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

**原生文档**：[torch.distributed.gather_object](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.gather_object)

**产品支持情况**：

<!-- npu="910b" id121 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id123 -->

**限制与说明**： 支持的输入类型为Python Object

</div>

### torch.distributed.scatter

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.scatter](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.scatter)

**产品支持情况**：

<!-- npu="910b" id124 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id126 -->

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

**原生文档**：[torch.distributed.scatter_object_list](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.scatter_object_list)

**产品支持情况**：

<!-- npu="910b" id127 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id129 -->

**限制与说明**： 不涉及`dtype`参数

</div>

### <code><i>class</i></code> torch.distributed.ReduceOp

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.ReduceOp](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.ReduceOp)

**产品支持情况**：

<!-- npu="910b" id130 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id132 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32，uint8，int8，int32，int64，bool

</div>

### torch.distributed.all_reduce

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.all_reduce](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.all_reduce)

**产品支持情况**：

<!-- npu="910b" id133 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id135 -->

**限制与说明**： `tensor`仅支持bf16，fp16，fp32，int32，int64，bool

</div>

### torch.distributed.barrier

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.barrier](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.barrier)

**产品支持情况**：

<!-- npu="910b" id136 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id138 -->

</div>

### torch.distributed.monitored_barrier

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.monitored_barrier](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.monitored_barrier)

**产品支持情况**：

<!-- npu="910b" id139 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id141 -->

</div>

## Distributed Key-Value Store

### <code><i>class</i></code> torch.distributed.PrefixStore

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.PrefixStore](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.PrefixStore)

**产品支持情况**：

<!-- npu="910b" id142 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id143 -->
<!-- npu="950" id144 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id144 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.PrefixStore.\_\_init\_\_](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.PrefixStore.__init__)

**产品支持情况**：

<!-- npu="910b" id145 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id147 -->

</div>

> <font size="3">underlying_store()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.PrefixStore.underlying_store](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.PrefixStore.underlying_store)

**产品支持情况**：

<!-- npu="910b" id148 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id150 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.Store

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store)

**产品支持情况**：

<!-- npu="910b" id151 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id153 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.\_\_init\_\_](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.__init__)

**产品支持情况**：

<!-- npu="910b" id154 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id156 -->

</div>

> <font size="3">set()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.set](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.set)

**产品支持情况**：

<!-- npu="910b" id157 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id159 -->

</div>

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.get](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.get)

**产品支持情况**：

<!-- npu="910b" id160 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id162 -->

</div>

> <font size="3">add()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.add](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.add)

**产品支持情况**：

<!-- npu="910b" id163 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id165 -->

</div>

> <font size="3">compare_set()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.compare_set](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.compare_set)

**产品支持情况**：

<!-- npu="910b" id166 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id168 -->

</div>

> <font size="3">wait()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.wait](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.wait)

**产品支持情况**：

<!-- npu="910b" id169 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id171 -->

</div>

> <font size="3">num_keys()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.num_keys](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.num_keys)

**产品支持情况**：

<!-- npu="910b" id172 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id174 -->

</div>

> <font size="3">delete_key()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.delete_key](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.delete_key)

**产品支持情况**：

<!-- npu="910b" id175 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id177 -->

</div>

> <font size="3">set_timeout()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.set_timeout](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.set_timeout)

**产品支持情况**：

<!-- npu="910b" id178 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id180 -->

</div>

> <font size="3">append()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.append](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.append)

**产品支持情况**：

<!-- npu="910b" id181 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id183 -->

</div>

> <font size="3">check()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.check](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.check)

**产品支持情况**：

<!-- npu="910b" id184 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id186 -->

</div>

> <font size="3">has_extended_api()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.has_extended_api](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.has_extended_api)

**产品支持情况**：

<!-- npu="910b" id187 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id189 -->

</div>

> <font size="3">multi_set()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.multi_set](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.multi_set)

**产品支持情况**：

<!-- npu="910b" id190 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id192 -->

</div>

> <font size="3">multi_get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.multi_get](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.multi_get)

**产品支持情况**：

<!-- npu="910b" id193 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id195 -->

</div>

> <font size="3">timeout()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.Store.timeout](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.timeout)

**产品支持情况**：

<!-- npu="910b" id196 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id197 -->
<!-- npu="950" id198 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id198 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.TCPStore

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.TCPStore)

**产品支持情况**：

<!-- npu="910b" id199 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id200 -->
<!-- npu="950" id201 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id201 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore.\_\_init\_\_](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.TCPStore.__init__)

**产品支持情况**：

<!-- npu="910b" id202 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id203 -->
<!-- npu="950" id204 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id204 -->

</div>

> <font size="3">host()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore.host](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.TCPStore.host)

**产品支持情况**：

<!-- npu="910b" id205 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id206 -->
<!-- npu="950" id207 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id207 -->

</div>

> <font size="3">libuvBackend()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore.libuvBackend](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.TCPStore.libuvBackend)

**产品支持情况**：

<!-- npu="910b" id208 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id209 -->
<!-- npu="950" id210 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id210 -->

</div>

> <font size="3">port()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.TCPStore.port](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.TCPStore.port)

**产品支持情况**：

<!-- npu="910b" id211 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id212 -->
<!-- npu="950" id213 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id213 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.HashStore

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.HashStore](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.HashStore)

**产品支持情况**：

<!-- npu="910b" id214 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id215 -->
<!-- npu="950" id216 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id216 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.HashStore.\_\_init\_\_](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.HashStore.__init__)

**产品支持情况**：

<!-- npu="910b" id217 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id218 -->
<!-- npu="950" id219 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id219 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.FileStore

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.FileStore](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.FileStore)

**产品支持情况**：

<!-- npu="910b" id220 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id220 -->
<!-- npu="A3" id221 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id221 -->
<!-- npu="950" id222 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id222 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.FileStore.\_\_init\_\_](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.FileStore.__init__)

**产品支持情况**：

<!-- npu="910b" id223 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id223 -->
<!-- npu="A3" id224 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id224 -->
<!-- npu="950" id225 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id225 -->

</div>

> <font size="3">path()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.FileStore.path](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.FileStore.path)

**产品支持情况**：

<!-- npu="910b" id226 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id226 -->
<!-- npu="A3" id227 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id227 -->
<!-- npu="950" id228 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id228 -->

</div>

</div>

## Logging

### <code><i>class</i></code> torch.distributed.DistBackendError

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.DistBackendError](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.DistBackendError)

**产品支持情况**：

<!-- npu="910b" id229 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id229 -->
<!-- npu="A3" id230 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id230 -->
<!-- npu="950" id231 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id231 -->

</div>
