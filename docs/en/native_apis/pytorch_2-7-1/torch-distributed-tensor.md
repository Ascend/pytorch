# torch.distributed.tensor

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.7/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.7/distributed.tensor.html).

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

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

> <font size="3">from_local()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `local_tensor` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

> <font size="3">redistribute()</font>

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool

</div>

</div>

### <code><i>class</i></code> torch.distributed.tensor.placement_types.Shard

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool, complex64, complex128

</div>

## Different ways to create a DTensor

### torch.distributed.tensor.distribute_module

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.distributed.tensor.distribute_tensor

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Supported

**Constraints & Notes**: `tensor` only supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool

</div>

## Experimental Features

### torch.distributed.tensor.experimental.context_parallel

<div style="margin-left: 2em">

**Support Status**:

- <term>Atlas A2 training products</term>: Supported
- <term>Atlas A3 training products</term>: Supported
- <term>Ascend 950DT</term>: Not Supported

**Constraints & Notes**: Only supports the NPU fused SDPA path; `q`/`k`/`v` only supports the BNSD layout; does not currently support `pse`, `padding_mask`, `prefix`, `actual_seq_qlen`, `actual_seq_kvlen`, `sink`, or any non-causal attention mask; causal attention is required when load balance is enabled; compiling to a computation graph via torch.compile is not currently supported.

</div>
