# DDP Communication Hooks

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.9/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.9/ddp_comm_hooks.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [What Does a Communication Hook Operate On?](#what-does-a-communication-hook-operate-on)
- [Default Communication Hooks](#default-communication-hooks)
- [PowerSGD Communication Hook](#powersgd-communication-hook)
- [Debugging Communication Hooks](#debugging-communication-hooks)

</div>

<div style="display:none;">

## &#8203;DDP Communication Hooks

</div>

## What Does a Communication Hook Operate On?

### <code><i>class</i></code> torch.distributed.GradBucket

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.GradBucket](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket)

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

**Constraints & Notes**: `input` only supports fp32

> <font size="3">index()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.GradBucket.index](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket.index)

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

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">buffer()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.GradBucket.buffer](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket.buffer)

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

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">gradients()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.GradBucket.gradients](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket.gradients)

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

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">is_last()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.GradBucket.is_last](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket.is_last)

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

> <font size="3">set_buffer()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.GradBucket.set_buffer](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket.set_buffer)

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

> <font size="3">parameters()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.GradBucket.parameters](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket.parameters)

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

## Default Communication Hooks

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook)

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

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper)

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id27 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper)

**Support Status**:

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id30 -->

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook)

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

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook)

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

</div>

## PowerSGD Communication Hook

### torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook)

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id39 -->

</div>

### <code><i>class</i></code> torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState)

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

> <font size="3">\_\_getstate\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.ddp\\\_comm_hooks.powerSGD_hook.PowerSGDState._\_getstate__](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState.__getstate__)

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

</div>

### torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook)

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

## Debugging Communication Hooks

### torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook)

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

**Constraints & Notes**: `input` only supports fp32

</div>
