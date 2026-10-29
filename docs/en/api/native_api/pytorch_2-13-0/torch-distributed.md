# torch.distributed

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.13/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.13/distributed.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

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

**Native Documentation**: [torch.distributed.is_available](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.is_available)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id3 -->

</div>

### torch.distributed.is_mpi_available

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.is_mpi_available](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.is_mpi_available)

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

### torch.distributed.is_nccl_available

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.is_nccl_available](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.is_nccl_available)

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

</div>

### <code><i>class</i></code> torch.distributed.ProcessGroupNCCL

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.ProcessGroupNCCL](https://pytorch.org/docs/2.13/distributed.html)

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

**Constraints & Notes**: This interface is supported only after calling `torch_npu.contrib.transfer_to_npu`. After the call, the NCCL process group interface is mapped to `ProcessGroupHCCL`, and the actual communication is performed by HCCL; when `torch_npu.contrib.transfer_to_npu` is not called, this interface is not supported.

</div>

### torch.distributed.is_gloo_available

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.is_gloo_available](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.is_gloo_available)

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

</div>

### torch.distributed.distributed_c10d.is_ucc_available

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.distributed_c10d.is_ucc_available](https://docs.pytorch.org/docs/2.13/distributed.html#torch.distributed.distributed_c10d.is_ucc_available)

**Support Status**:

<!-- npu="910b" id232 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id232 -->
<!-- npu="A3" id233 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id233 -->
<!-- npu="950" id234 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id234 -->

</div>

### torch.distributed.is_torchelastic_launched

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.is_torchelastic_launched](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.is_torchelastic_launched)

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

</div>

### torch.distributed.init_process_group

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.init_process_group](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.init_process_group)

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

**Constraints & Notes**: When the `pg_options` parameter is passed a value of type `torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()`, you can control the HCCL communicator buffer size by configuring the `hccl_config` attribute of this variable. For specific examples, refer to the `hccl_buffer_size` configuration description in [Configuring HCCL communicator parameters through pg_options](../../../developer_notes/distributed/parameter_configuration/setting_HCCL_communicator_parameter.md). You can set a custom name for the communication group of the HCCL communicator through the `group_name` field of the `hccl_config` variable, whose value must be a string of no more than 32 characters.

</div>

### torch.distributed.is_initialized

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.is_initialized](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.is_initialized)

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id24 -->

</div>

## Post-Initialization

### torch.distributed.get_backend

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.get_backend](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.get_backend)

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

### torch.distributed.get_debug_level

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.get_debug_level](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.get_debug_level)

**Support Status**:

<!-- npu="910b" id235 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id235 -->
<!-- npu="A3" id236 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id236 -->
<!-- npu="950" id237 -->
- <term>Ascend 950DT</term>: Supported
<!-- end id237 -->

</div>

### torch.distributed.get_rank

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.get_rank](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.get_rank)

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

### torch.distributed.get_world_size

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.get_world_size](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.get_world_size)

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

### <code><i>class</i></code> torch.distributed.Backend

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Backend](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Backend)

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id36 -->

> <font size="3">register_backend()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Backend.register_backend](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Backend.register_backend)

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id39 -->

</div>

</div>

## Groups

### torch.distributed.get_group_rank

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.get_group_rank](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.get_group_rank)

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

### torch.distributed.get_global_rank

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.get_global_rank](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.get_global_rank)

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

</div>

### torch.distributed.get_process_group_ranks

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.get_process_group_ranks](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.get_process_group_ranks)

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

### torch.distributed.new_group

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.new_group](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.new_group)

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

**Constraints & Notes**: When the `pg_options` parameter is passed a value of type `torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()`, you can control the HCCL communicator buffer size by configuring the `hccl_config` attribute of this variable. For specific examples, refer to the `hccl_buffer_size` configuration description in [Configuring HCCL communicator parameters through pg_options](../../../developer_notes/distributed/parameter_configuration/setting_HCCL_communicator_parameter.md). You can set a custom name for the communication group of the HCCL communicator through the `group_name` field of the `hccl_config` variable, whose value must be a string of no more than 32 characters.

</div>

## DeviceMesh

### <code><i>class</i></code> torch.distributed.device_mesh.DeviceMesh

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.device_mesh.DeviceMesh](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.device_mesh.DeviceMesh)

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

> <font size="3">from_group()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.device_mesh.DeviceMesh.from_group](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.device_mesh.DeviceMesh.from_group)

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

> <font size="3">get_all_groups()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.device_mesh.DeviceMesh.get_all_groups](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_all_groups)

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

> <font size="3">get_coordinate()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.device_mesh.DeviceMesh.get_coordinate](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_coordinate)

**Support Status**:

<!-- npu="910b" id61 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id63 -->

</div>

> <font size="3">get_group()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.device_mesh.DeviceMesh.get_group](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_group)

**Support Status**:

<!-- npu="910b" id64 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id66 -->

</div>

> <font size="3">get_local_rank()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.device_mesh.DeviceMesh.get_local_rank](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_local_rank)

**Support Status**:

<!-- npu="910b" id67 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id69 -->

</div>

> <font size="3">get_rank()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.device_mesh.DeviceMesh.get_rank](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_rank)

**Support Status**:

<!-- npu="910b" id70 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id72 -->

</div>

</div>

## Point-to-point communication

### torch.distributed.batch_isend_irecv

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.batch_isend_irecv](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.batch_isend_irecv)

**Support Status**:

<!-- npu="910b" id73 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id75 -->

**Constraints & Notes**: `p2p_op_list` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### <code><i>class</i></code> torch.distributed.P2POp

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.P2POp](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.P2POp)

**Support Status**:

<!-- npu="910b" id76 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id78 -->

**Constraints & Notes**: `tensor` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.distributed.send

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.send](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.send)

**Support Status**:

<!-- npu="910b" id79 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id81 -->

**Constraints & Notes**: `tensor` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.distributed.recv

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.recv](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.recv)

**Support Status**:

<!-- npu="910b" id82 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id84 -->

**Constraints & Notes**: `tensor` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.distributed.isend

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.isend](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.isend)

**Support Status**:

<!-- npu="910b" id85 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id87 -->

**Constraints & Notes**: `tensor` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.distributed.irecv

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.irecv](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.irecv)

**Support Status**:

<!-- npu="910b" id88 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id90 -->

**Constraints & Notes**: `tensor` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

## Collective functions

### torch.distributed.reduce_scatter

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.reduce_scatter](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.reduce_scatter)

**Support Status**:

<!-- npu="910b" id91 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id93 -->

**Constraints & Notes**: `input_list` only supports bf16, fp16, fp32, int8, int32

</div>

### torch.distributed.all_to_all_single

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.all_to_all_single](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.all_to_all_single)

**Support Status**:

<!-- npu="910b" id94 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id96 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.distributed.all_to_all

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.all_to_all](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.all_to_all)

**Support Status**:

<!-- npu="910b" id97 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id99 -->

**Constraints & Notes**:

- `input_tensor_list` only supports fp32
- By setting `torch_npu.npu.use_compatible_impl(True)`, `torch.distributed.all_to_all` switches to be consistent with the native implementation, for example:

  ```python
  import torch_npu
  torch_npu.npu.use_compatible_impl(True)
  ```

</div>

### torch.distributed.reduce_op

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.reduce_op](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.reduce_op)

**Support Status**:

<!-- npu="910b" id100 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id102 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int32, int64

</div>

### torch.distributed.broadcast

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.broadcast](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.broadcast)

**Support Status**:

<!-- npu="910b" id103 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id105 -->

**Constraints & Notes**: `tensor` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.distributed.broadcast_object_list

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.broadcast_object_list](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.broadcast_object_list)

**Support Status**:

<!-- npu="910b" id106 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id108 -->

</div>

### torch.distributed.reduce

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.reduce](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.reduce)

**Support Status**:

<!-- npu="910b" id109 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id111 -->

**Constraints & Notes**: `tensor` only supports bf16, fp16, fp32, uint8, int8, int32, int64, bool

</div>

### torch.distributed.all_gather

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.all_gather](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.all_gather)

**Support Status**:

<!-- npu="910b" id112 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id114 -->

**Constraints & Notes**: `tensor` only supports bf16, fp16, fp32, int8, int32, bool

</div>

### torch.distributed.all_gather_object

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.all_gather_object](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.all_gather_object)

**Support Status**:

<!-- npu="910b" id115 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id117 -->

</div>

### torch.distributed.gather

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.gather](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.gather)

**Support Status**:

<!-- npu="910b" id118 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id120 -->

**Constraints & Notes**:

- `tensor` only supports bf16, fp16, fp32, int8, int32, bool
- By setting `torch_npu.npu.use_compatible_impl(True)`, `torch.distributed.gather` switches to be consistent with the native implementation, for example:

  ```python
  import torch_npu
  torch_npu.npu.use_compatible_impl(True)
  ```

</div>

### torch.distributed.gather_object

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.gather_object](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.gather_object)

**Support Status**:

<!-- npu="910b" id121 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id123 -->

**Constraints & Notes**: The supported input type is a Python Object

</div>

### torch.distributed.scatter

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.scatter](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.scatter)

**Support Status**:

<!-- npu="910b" id124 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id126 -->

**Constraints & Notes**:

- `tensor` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool
- By setting `torch_npu.npu.use_compatible_impl(True)`, `torch.distributed.scatter` switches to be consistent with the native implementation, for example:

  ```python
  import torch_npu
  torch_npu.npu.use_compatible_impl(True)
  ```

</div>

### torch.distributed.scatter_object_list

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.scatter_object_list](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.scatter_object_list)

**Support Status**:

<!-- npu="910b" id127 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id129 -->

**Constraints & Notes**: The `dtype` parameter is not involved.

</div>

### <code><i>class</i></code> torch.distributed.ReduceOp

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.ReduceOp](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.ReduceOp)

**Support Status**:

<!-- npu="910b" id130 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id132 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int32, int64, bool

</div>

### torch.distributed.all_reduce

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.all_reduce](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.all_reduce)

**Support Status**:

<!-- npu="910b" id133 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id135 -->

**Constraints & Notes**: `tensor` only supports bf16, fp16, fp32, int32, int64, bool

</div>

### torch.distributed.barrier

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.barrier](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.barrier)

**Support Status**:

<!-- npu="910b" id136 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id138 -->

</div>

### torch.distributed.monitored_barrier

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.monitored_barrier](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.monitored_barrier)

**Support Status**:

<!-- npu="910b" id139 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id141 -->

</div>

## Distributed Key-Value Store

### <code><i>class</i></code> torch.distributed.PrefixStore

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.PrefixStore](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.PrefixStore)

**Support Status**:

<!-- npu="910b" id142 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id143 -->
<!-- npu="950" id144 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id144 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.PrefixStore.\_\_init\_\_](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.PrefixStore.__init__)

**Support Status**:

<!-- npu="910b" id145 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id147 -->

</div>

> <font size="3">underlying_store()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.PrefixStore.underlying_store](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.PrefixStore.underlying_store)

**Support Status**:

<!-- npu="910b" id148 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id150 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.Store

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store)

**Support Status**:

<!-- npu="910b" id151 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id153 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.\_\_init\_\_](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.__init__)

**Support Status**:

<!-- npu="910b" id154 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id156 -->

</div>

> <font size="3">set()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.set](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.set)

**Support Status**:

<!-- npu="910b" id157 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id159 -->

</div>

> <font size="3">get()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.get](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.get)

**Support Status**:

<!-- npu="910b" id160 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id162 -->

</div>

> <font size="3">add()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.add](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.add)

**Support Status**:

<!-- npu="910b" id163 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id165 -->

</div>

> <font size="3">compare_set()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.compare_set](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.compare_set)

**Support Status**:

<!-- npu="910b" id166 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id168 -->

</div>

> <font size="3">wait()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.wait](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.wait)

**Support Status**:

<!-- npu="910b" id169 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id171 -->

</div>

> <font size="3">num_keys()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.num_keys](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.num_keys)

**Support Status**:

<!-- npu="910b" id172 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id174 -->

</div>

> <font size="3">delete_key()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.delete_key](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.delete_key)

**Support Status**:

<!-- npu="910b" id175 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id177 -->

</div>

> <font size="3">set_timeout()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.set_timeout](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.set_timeout)

**Support Status**:

<!-- npu="910b" id178 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id180 -->

</div>

> <font size="3">append()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.append](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.append)

**Support Status**:

<!-- npu="910b" id181 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id183 -->

</div>

> <font size="3">check()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.check](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.check)

**Support Status**:

<!-- npu="910b" id184 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id186 -->

</div>

> <font size="3">has_extended_api()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.has_extended_api](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.has_extended_api)

**Support Status**:

<!-- npu="910b" id187 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id189 -->

</div>

> <font size="3">multi_set()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.multi_set](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.multi_set)

**Support Status**:

<!-- npu="910b" id190 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id192 -->

</div>

> <font size="3">multi_get()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.multi_get](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.multi_get)

**Support Status**:

<!-- npu="910b" id193 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id195 -->

</div>

> <font size="3">timeout()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.timeout](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.Store.timeout)

**Support Status**:

<!-- npu="910b" id196 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id197 -->
<!-- npu="950" id198 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id198 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.TCPStore

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.TCPStore](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.TCPStore)

**Support Status**:

<!-- npu="910b" id199 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id200 -->
<!-- npu="950" id201 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id201 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.TCPStore.\_\_init\_\_](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.TCPStore.__init__)

**Support Status**:

<!-- npu="910b" id202 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id203 -->
<!-- npu="950" id204 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id204 -->

</div>

> <font size="3">host()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.TCPStore.host](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.TCPStore.host)

**Support Status**:

<!-- npu="910b" id205 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id206 -->
<!-- npu="950" id207 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id207 -->

</div>

> <font size="3">libuvBackend()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.TCPStore.libuvBackend](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.TCPStore.libuvBackend)

**Support Status**:

<!-- npu="910b" id208 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id209 -->
<!-- npu="950" id210 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id210 -->

</div>

> <font size="3">port()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.TCPStore.port](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.TCPStore.port)

**Support Status**:

<!-- npu="910b" id211 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id212 -->
<!-- npu="950" id213 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id213 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.HashStore

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.HashStore](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.HashStore)

**Support Status**:

<!-- npu="910b" id214 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id215 -->
<!-- npu="950" id216 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id216 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.HashStore.\_\_init\_\_](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.HashStore.__init__)

**Support Status**:

<!-- npu="910b" id217 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id218 -->
<!-- npu="950" id219 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id219 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.FileStore

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.FileStore](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.FileStore)

**Support Status**:

<!-- npu="910b" id220 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id220 -->
<!-- npu="A3" id221 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id221 -->
<!-- npu="950" id222 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id222 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.FileStore.\_\_init\_\_](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.FileStore.__init__)

**Support Status**:

<!-- npu="910b" id223 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id223 -->
<!-- npu="A3" id224 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id224 -->
<!-- npu="950" id225 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id225 -->

</div>

> <font size="3">path()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.FileStore.path](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.FileStore.path)

**Support Status**:

<!-- npu="910b" id226 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id226 -->
<!-- npu="A3" id227 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id227 -->
<!-- npu="950" id228 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id228 -->

</div>

</div>

## Logging

### <code><i>class</i></code> torch.distributed.DistError

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.DistError](https://docs.pytorch.org/docs/2.13/distributed.html#torch.distributed.DistError)

**Support Status**:

<!-- npu="910b" id238 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id238 -->
<!-- npu="A3" id239 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id239 -->
<!-- npu="950" id240 -->
- <term>Ascend 950DT</term>: Supported
<!-- end id240 -->

</div>

### <code><i>class</i></code> torch.distributed.DistBackendError

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.DistBackendError](https://pytorch.org/docs/2.13/distributed.html#torch.distributed.DistBackendError)

**Support Status**:

<!-- npu="910b" id229 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id229 -->
<!-- npu="A3" id230 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id230 -->
<!-- npu="950" id231 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id231 -->

</div>

### <code><i>class</i></code> torch.distributed.DistNetworkError

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.DistNetworkError](https://docs.pytorch.org/docs/2.13/distributed.html#torch.distributed.DistNetworkError)

**Support Status**:

<!-- npu="910b" id241 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id241 -->
<!-- npu="A3" id242 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id242 -->
<!-- npu="950" id243 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id243 -->

</div>
