# DDP Communication Hooks

> [!NOTE]  
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)
- [What Does a Communication Hook Operate On?](#what-does-a-communication-hook-operate-on)
- [Default Communication Hooks](#default-communication-hooks)
- [PowerSGD Communication Hook](#powersgd-communication-hook)

## base API

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook)

**是否支持**：是

</div>

### torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

## What Does a Communication Hook Operate On?

### _`class`_ torch.distributed.GradBucket

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

> <font size="3">index()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.index](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket.index)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.buffer](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket.buffer)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">gradients()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.gradients](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket.gradients)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">is_last()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.is_last](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket.is_last)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.set_buffer](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket.set_buffer)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.parameters](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket.parameters)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

## Default Communication Hooks

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## PowerSGD Communication Hook

### _`class`_ torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook)

**是否支持**：是

</div>
