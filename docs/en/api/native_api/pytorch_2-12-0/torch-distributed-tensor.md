# torch.distributed.tensor

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.12/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.12/distributed.tensor.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [PyTorch DTensor (Distributed Tensor)](#pytorch-dtensor-distributed-tensor)
- [Different ways to create a DTensor](#different-ways-to-create-a-dtensor)
- [Experimental Features](#experimental-features)

</div>

<div style="display:none;">

## &#8203;torch.distributed.tensor

</div>

## PyTorch DTensor (Distributed Tensor)

### <code><i>class</i></code> torch.distributed.tensor.DTensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.tensor.DTensor](https://pytorch.org/docs/2.12/distributed.tensor.html#torch.distributed.tensor.DTensor)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id3 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

> <font size="3">from_local()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.tensor.DTensor.from_local](https://pytorch.org/docs/2.12/distributed.tensor.html#torch.distributed.tensor.DTensor.from_local)

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

**Constraints & Notes**: `local_tensor` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">redistribute()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.tensor.DTensor.redistribute](https://pytorch.org/docs/2.12/distributed.tensor.html#torch.distributed.tensor.DTensor.redistribute)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool

</div>

</div>

### <code><i>class</i></code> torch.distributed.tensor.placement_types.Shard

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.tensor.placement_types.Shard](https://pytorch.org/docs/2.12/distributed.tensor.html#torch.distributed.tensor.placement_types.Shard)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool, complex64, complex128

</div>

### torch.distributed.tensor.placement_types._StridedShard

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.tensor.placement_types._StridedShard](https://pytorch.org/docs/2.12/distributed.tensor.html#torch.distributed.tensor.placement_types._StridedShard)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool

</div>

### <code><i>class</i></code> torch.distributed.tensor.placement_types.Placement

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.tensor.placement_types.Placement](https://pytorch.org/docs/2.12/distributed.tensor.html#torch.distributed.tensor.placement_types.Placement)

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

> <font size="3">is_replicate()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.tensor.placement_types.Placement.is_replicate](https://pytorch.org/docs/2.12/distributed.tensor.html#torch.distributed.tensor.placement_types.Placement.is_replicate)

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

</div>

</div>

## Different ways to create a DTensor

### torch.distributed.tensor.distribute_module

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.tensor.distribute_module](https://pytorch.org/docs/2.12/distributed.tensor.html#torch.distributed.tensor.distribute_module)

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id24 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.distributed.tensor.distribute_tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.tensor.distribute_tensor](https://pytorch.org/docs/2.12/distributed.tensor.html#torch.distributed.tensor.distribute_tensor)

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

**Constraints & Notes**: `tensor` only supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool

</div>

## Experimental Features

### torch.distributed.tensor.experimental.context_parallel

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.tensor.experimental.context_parallel](https://pytorch.org/docs/2.12/distributed.tensor.html#torch.distributed.tensor.experimental.context_parallel)

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

**Constraints & Notes**: Only supports the NPU fused SDPA path; `q`/`k`/`v` only supports the BNSD layout; does not support `pse`, `padding_mask`, `prefix`, `actual_seq_qlen`, `actual_seq_kvlen`, `sink`, or any non-causal attention mask; causal attention is required when load balance is enabled; compiling to a computation graph via torch.compile is not supported.

</div>
