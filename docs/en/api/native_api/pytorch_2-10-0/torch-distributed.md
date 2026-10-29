# torch.distributed

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.10/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.10/distributed.html).

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

**Native Documentation**: [torch.distributed.is_available](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.is_available)

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

**Native Documentation**: [torch.distributed.is_mpi_available](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.is_mpi_available)

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

**Native Documentation**: [torch.distributed.is_nccl_available](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.is_nccl_available)

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

**Native Documentation**: [torch.distributed.ProcessGroupNCCL](https://pytorch.org/docs/2.10/distributed.html)

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

**Constraints & Notes**: This API is supported only after `torch_npu.contrib.transfer_to_npu` is called. After the call, the NCCL process group APIs are mapped to `ProcessGroupHCCL`, and the actual communication is performed by HCCL. When `torch_npu.contrib.transfer_to_npu` is not called, this API is not supported.

</div>

### torch.distributed.is_gloo_available

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.is_gloo_available](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.is_gloo_available)

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

**Native Documentation**: [torch.distributed.distributed_c10d.is_ucc_available](https://docs.pytorch.org/docs/2.10/distributed.html#torch.distributed.distributed_c10d.is_ucc_available)

**Support Status**:

<!-- npu="910b" id239 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id239 -->
<!-- npu="A3" id240 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id240 -->
<!-- npu="950" id241 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id241 -->

</div>

### torch.distributed.is_torchelastic_launched

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.is_torchelastic_launched](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.is_torchelastic_launched)

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

**Native Documentation**: [torch.distributed.init_process_group](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.init_process_group)

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

**Constraints & Notes**: When the `pg_options` parameter is passed in with the type `torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()`, the HCCL communicator buffer size can be controlled by configuring the `hccl_config` attribute of the variable. For specific examples, refer to the `hccl_buffer_size` configuration description in [Configuring HCCL Communicator Parameters Through pg_options](../../../developer_notes/distributed/parameter_configuration/setting_HCCL_communicator_parameter.md). A custom name for the communication group of the HCCL communicator can be set by configuring the `group_name` field of the `hccl_config` variable, whose value is a string with a length not exceeding 32.

</div>

### torch.distributed.is_initialized

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.is_initialized](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.is_initialized)

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

**Native Documentation**: [torch.distributed.get_backend](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.get_backend)

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

### torch.distributed.get_rank

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.get_rank](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.get_rank)

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

**Native Documentation**: [torch.distributed.get_world_size](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.get_world_size)

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

**Native Documentation**: [torch.distributed.Backend](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.Backend)

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

**Native Documentation**: [torch.distributed.Backend.register_backend](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.Backend.register_backend)

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

**Native Documentation**: [torch.distributed.get_group_rank](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.get_group_rank)

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

**Native Documentation**: [torch.distributed.get_global_rank](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.get_global_rank)

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

**Native Documentation**: [torch.distributed.get_process_group_ranks](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.get_process_group_ranks)

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

**Native Documentation**: [torch.distributed.new_group](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.new_group)

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

**Constraints & Notes**: When the `pg_options` parameter is passed in with the type `torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()`, the HCCL communicator buffer size can be controlled by configuring the `hccl_config` attribute of the variable. For specific examples, refer to the `hccl_buffer_size` configuration description in [Configuring HCCL Communicator Parameters Through pg_options](../../../developer_notes/distributed/parameter_configuration/setting_HCCL_communicator_parameter.md). A custom name for the communication group of the HCCL communicator can be set by configuring the `group_name` field of the `hccl_config` variable, whose value is a string with a length not exceeding 32.

</div>

## DeviceMesh

### <code><i>class</i></code> torch.distributed.device_mesh.DeviceMesh

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.device_mesh.DeviceMesh](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.device_mesh.DeviceMesh)

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

**Native Documentation**: [torch.distributed.device_mesh.DeviceMesh.from_group](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.device_mesh.DeviceMesh.from_group)

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

**Native Documentation**: [torch.distributed.device_mesh.DeviceMesh.get_all_groups](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_all_groups)

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

**Native Documentation**: [torch.distributed.device_mesh.DeviceMesh.get_coordinate](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_coordinate)

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

**Native Documentation**: [torch.distributed.device_mesh.DeviceMesh.get_group](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_group)

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

**Native Documentation**: [torch.distributed.device_mesh.DeviceMesh.get_local_rank](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_local_rank)

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

**Native Documentation**: [torch.distributed.device_mesh.DeviceMesh.get_rank](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_rank)

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

**Native Documentation**: [torch.distributed.batch_isend_irecv](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.batch_isend_irecv)

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

### torch.distributed.send

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.send](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.send)

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

### torch.distributed.recv

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.recv](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.recv)

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

### torch.distributed.isend

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.isend](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.isend)

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

### torch.distributed.irecv

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.irecv](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.irecv)

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

### <code><i>class</i></code> torch.distributed.P2POp

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.P2POp](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.P2POp)

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

### torch.distributed.all_gather_into_tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.all_gather_into_tensor](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.all_gather_into_tensor)

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

**Constraints & Notes**:

- `output_tensor` only supports bf16, fp16, fp32, int8, int32, bool
- `world_size` does not support 3, 5, 6, 7

</div>

### torch.distributed.reduce_scatter

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.reduce_scatter](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.reduce_scatter)

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

**Constraints & Notes**: `input_list` only supports bf16, fp16, fp32, int8, int32

</div>

### torch.distributed.reduce_scatter_tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.reduce_scatter_tensor](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.reduce_scatter_tensor)

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

- `output` only supports bf16, fp16, fp32, int8, int32
- `world_size` does not support 3, 5, 6, 7

<!-- npu="910b" id100 -->
- For <term>Atlas A2 training products</term>, the "prod" operation in the current version does not support the int16 and bf16 data types.
<!-- end id100 -->

</div>

### torch.distributed.all_to_all_single

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.all_to_all_single](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.all_to_all_single)

**Support Status**:

<!-- npu="910b" id101 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id101 -->
<!-- npu="A3" id102 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id102 -->
<!-- npu="950" id103 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id103 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.distributed.all_to_all

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.all_to_all](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.all_to_all)

**Support Status**:

<!-- npu="910b" id104 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id104 -->
<!-- npu="A3" id105 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id105 -->
<!-- npu="950" id106 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id106 -->

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

**Native Documentation**: [torch.distributed.reduce_op](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.reduce_op)

**Support Status**:

<!-- npu="910b" id107 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id107 -->
<!-- npu="A3" id108 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id108 -->
<!-- npu="950" id109 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id109 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int32, int64

</div>

### torch.distributed.broadcast

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.broadcast](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.broadcast)

**Support Status**:

<!-- npu="910b" id110 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id110 -->
<!-- npu="A3" id111 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id111 -->
<!-- npu="950" id112 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id112 -->

**Constraints & Notes**: `tensor` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

### torch.distributed.broadcast_object_list

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.broadcast_object_list](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.broadcast_object_list)

**Support Status**:

<!-- npu="910b" id113 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id113 -->
<!-- npu="A3" id114 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id114 -->
<!-- npu="950" id115 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id115 -->

</div>

### torch.distributed.reduce

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.reduce](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.reduce)

**Support Status**:

<!-- npu="910b" id116 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id116 -->
<!-- npu="A3" id117 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id117 -->
<!-- npu="950" id118 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id118 -->

**Constraints & Notes**: `tensor` only supports bf16, fp16, fp32, uint8, int8, int32, int64, bool

</div>

### torch.distributed.all_gather

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.all_gather](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.all_gather)

**Support Status**:

<!-- npu="910b" id119 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id119 -->
<!-- npu="A3" id120 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id120 -->
<!-- npu="950" id121 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id121 -->

**Constraints & Notes**: `tensor` only supports bf16, fp16, fp32, int8, int32, bool

</div>

### torch.distributed.all_gather_object

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.all_gather_object](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.all_gather_object)

**Support Status**:

<!-- npu="910b" id122 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id122 -->
<!-- npu="A3" id123 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id123 -->
<!-- npu="950" id124 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id124 -->

</div>

### torch.distributed.gather

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.gather](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.gather)

**Support Status**:

<!-- npu="910b" id125 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id125 -->
<!-- npu="A3" id126 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id126 -->
<!-- npu="950" id127 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id127 -->

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

**Native Documentation**: [torch.distributed.gather_object](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.gather_object)

**Support Status**:

<!-- npu="910b" id128 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id128 -->
<!-- npu="A3" id129 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id129 -->
<!-- npu="950" id130 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id130 -->

**Constraints & Notes**: The supported input type is a Python object.

</div>

### torch.distributed.scatter

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.scatter](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.scatter)

**Support Status**:

<!-- npu="910b" id131 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id131 -->
<!-- npu="A3" id132 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id132 -->
<!-- npu="950" id133 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id133 -->

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

**Native Documentation**: [torch.distributed.scatter_object_list](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.scatter_object_list)

**Support Status**:

<!-- npu="910b" id134 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id134 -->
<!-- npu="A3" id135 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id135 -->
<!-- npu="950" id136 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id136 -->

**Constraints & Notes**: The `dtype` parameter is not involved.

</div>

### <code><i>class</i></code> torch.distributed.ReduceOp

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.ReduceOp](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.ReduceOp)

**Support Status**:

<!-- npu="910b" id137 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id137 -->
<!-- npu="A3" id138 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id138 -->
<!-- npu="950" id139 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id139 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, uint8, int8, int32, int64, bool

</div>

### torch.distributed.all_reduce

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.all_reduce](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.all_reduce)

**Support Status**:

<!-- npu="910b" id140 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id140 -->
<!-- npu="A3" id141 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id141 -->
<!-- npu="950" id142 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id142 -->

**Constraints & Notes**: `tensor` only supports bf16, fp16, fp32, int32, int64, bool

</div>

### torch.distributed.barrier

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.barrier](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.barrier)

**Support Status**:

<!-- npu="910b" id143 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id143 -->
<!-- npu="A3" id144 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id144 -->
<!-- npu="950" id145 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id145 -->

</div>

### torch.distributed.monitored_barrier

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.monitored_barrier](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.monitored_barrier)

**Support Status**:

<!-- npu="910b" id146 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id146 -->
<!-- npu="A3" id147 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id147 -->
<!-- npu="950" id148 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id148 -->

</div>

## Distributed Key-Value Store

### <code><i>class</i></code> torch.distributed.PrefixStore

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.PrefixStore](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.PrefixStore)

**Support Status**:

<!-- npu="910b" id149 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id149 -->
<!-- npu="A3" id150 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id150 -->
<!-- npu="950" id151 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id151 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.PrefixStore.\_\_init\_\_](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.PrefixStore.__init__)

**Support Status**:

<!-- npu="910b" id152 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id152 -->
<!-- npu="A3" id153 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id153 -->
<!-- npu="950" id154 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id154 -->

</div>

> <font size="3">underlying_store()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.PrefixStore.underlying_store](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.PrefixStore.underlying_store)

**Support Status**:

<!-- npu="910b" id155 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id155 -->
<!-- npu="A3" id156 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id156 -->
<!-- npu="950" id157 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id157 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.Store

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.Store)

**Support Status**:

<!-- npu="910b" id158 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id158 -->
<!-- npu="A3" id159 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id159 -->
<!-- npu="950" id160 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id160 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.\_\_init\_\_](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.Store.__init__)

**Support Status**:

<!-- npu="910b" id161 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id161 -->
<!-- npu="A3" id162 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id162 -->
<!-- npu="950" id163 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id163 -->

</div>

> <font size="3">set()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.set](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.Store.set)

**Support Status**:

<!-- npu="910b" id164 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id164 -->
<!-- npu="A3" id165 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id165 -->
<!-- npu="950" id166 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id166 -->

</div>

> <font size="3">get()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.get](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.Store.get)

**Support Status**:

<!-- npu="910b" id167 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id167 -->
<!-- npu="A3" id168 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id168 -->
<!-- npu="950" id169 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id169 -->

</div>

> <font size="3">add()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.add](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.Store.add)

**Support Status**:

<!-- npu="910b" id170 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id170 -->
<!-- npu="A3" id171 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id171 -->
<!-- npu="950" id172 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id172 -->

</div>

> <font size="3">compare_set()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.compare_set](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.Store.compare_set)

**Support Status**:

<!-- npu="910b" id173 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id173 -->
<!-- npu="A3" id174 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id174 -->
<!-- npu="950" id175 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id175 -->

</div>

> <font size="3">wait()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.wait](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.Store.wait)

**Support Status**:

<!-- npu="910b" id176 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id176 -->
<!-- npu="A3" id177 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id177 -->
<!-- npu="950" id178 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id178 -->

</div>

> <font size="3">num_keys()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.num_keys](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.Store.num_keys)

**Support Status**:

<!-- npu="910b" id179 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id179 -->
<!-- npu="A3" id180 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id180 -->
<!-- npu="950" id181 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id181 -->

</div>

> <font size="3">delete_key()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.delete_key](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.Store.delete_key)

**Support Status**:

<!-- npu="910b" id182 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id182 -->
<!-- npu="A3" id183 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id183 -->
<!-- npu="950" id184 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id184 -->

</div>

> <font size="3">set_timeout()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.set_timeout](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.Store.set_timeout)

**Support Status**:

<!-- npu="910b" id185 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id185 -->
<!-- npu="A3" id186 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id186 -->
<!-- npu="950" id187 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id187 -->

</div>

> <font size="3">append()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.append](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.Store.append)

**Support Status**:

<!-- npu="910b" id188 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id188 -->
<!-- npu="A3" id189 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id189 -->
<!-- npu="950" id190 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id190 -->

</div>

> <font size="3">check()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.check](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.Store.check)

**Support Status**:

<!-- npu="910b" id191 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id191 -->
<!-- npu="A3" id192 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id192 -->
<!-- npu="950" id193 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id193 -->

</div>

> <font size="3">has_extended_api()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.has_extended_api](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.Store.has_extended_api)

**Support Status**:

<!-- npu="910b" id194 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id194 -->
<!-- npu="A3" id195 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id195 -->
<!-- npu="950" id196 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id196 -->

</div>

> <font size="3">multi_set()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.multi_set](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.Store.multi_set)

**Support Status**:

<!-- npu="910b" id197 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id197 -->
<!-- npu="A3" id198 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id198 -->
<!-- npu="950" id199 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id199 -->

</div>

> <font size="3">multi_get()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.multi_get](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.Store.multi_get)

**Support Status**:

<!-- npu="910b" id200 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id200 -->
<!-- npu="A3" id201 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id201 -->
<!-- npu="950" id202 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id202 -->

</div>

> <font size="3">timeout()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.Store.timeout](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.Store.timeout)

**Support Status**:

<!-- npu="910b" id203 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id203 -->
<!-- npu="A3" id204 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id204 -->
<!-- npu="950" id205 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id205 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.TCPStore

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.TCPStore](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.TCPStore)

**Support Status**:

<!-- npu="910b" id206 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id206 -->
<!-- npu="A3" id207 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id207 -->
<!-- npu="950" id208 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id208 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.TCPStore.\_\_init\_\_](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.TCPStore.__init__)

**Support Status**:

<!-- npu="910b" id209 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id209 -->
<!-- npu="A3" id210 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id210 -->
<!-- npu="950" id211 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id211 -->

</div>

> <font size="3">host()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.TCPStore.host](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.TCPStore.host)

**Support Status**:

<!-- npu="910b" id212 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id212 -->
<!-- npu="A3" id213 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id213 -->
<!-- npu="950" id214 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id214 -->

</div>

> <font size="3">libuvBackend()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.TCPStore.libuvBackend](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.TCPStore.libuvBackend)

**Support Status**:

<!-- npu="910b" id215 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id215 -->
<!-- npu="A3" id216 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id216 -->
<!-- npu="950" id217 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id217 -->

</div>

> <font size="3">port()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.TCPStore.port](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.TCPStore.port)

**Support Status**:

<!-- npu="910b" id218 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id218 -->
<!-- npu="A3" id219 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id219 -->
<!-- npu="950" id220 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id220 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.HashStore

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.HashStore](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.HashStore)

**Support Status**:

<!-- npu="910b" id221 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id221 -->
<!-- npu="A3" id222 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id222 -->
<!-- npu="950" id223 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id223 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.HashStore.\_\_init\_\_](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.HashStore.__init__)

**Support Status**:

<!-- npu="910b" id224 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id224 -->
<!-- npu="A3" id225 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id225 -->
<!-- npu="950" id226 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id226 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.FileStore

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.FileStore](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.FileStore)

**Support Status**:

<!-- npu="910b" id227 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id227 -->
<!-- npu="A3" id228 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id228 -->
<!-- npu="950" id229 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id229 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.FileStore.\_\_init\_\_](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.FileStore.__init__)

**Support Status**:

<!-- npu="910b" id230 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id230 -->
<!-- npu="A3" id231 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id231 -->
<!-- npu="950" id232 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id232 -->

</div>

> <font size="3">path()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.FileStore.path](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.FileStore.path)

**Support Status**:

<!-- npu="910b" id233 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id233 -->
<!-- npu="A3" id234 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id234 -->
<!-- npu="950" id235 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id235 -->

</div>

</div>

## Logging

### <code><i>class</i></code> torch.distributed.DistBackendError

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.DistBackendError](https://pytorch.org/docs/2.10/distributed.html#torch.distributed.DistBackendError)

**Support Status**:

<!-- npu="910b" id236 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id236 -->
<!-- npu="A3" id237 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id237 -->
<!-- npu="950" id238 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id238 -->

</div>

### <code><i>class</i></code> torch.distributed.DistNetworkError

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.DistNetworkError](https://docs.pytorch.org/docs/2.10/distributed.html#torch.distributed.DistNetworkError)

**Support Status**:

<!-- npu="910b" id242 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id242 -->
<!-- npu="A3" id243 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id243 -->
<!-- npu="950" id244 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id244 -->

</div>
