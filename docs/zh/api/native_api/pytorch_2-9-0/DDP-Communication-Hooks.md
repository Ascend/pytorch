# DDP Communication Hooks

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.9/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.9/ddp_comm_hooks.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

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

**原生文档**：[torch.distributed.GradBucket](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket)

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

**限制与说明**：`input`仅支持fp32

> <font size="3">index()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.index](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket.index)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id6 -->

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.buffer](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket.buffer)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id9 -->

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">gradients()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.gradients](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket.gradients)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id12 -->

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">is_last()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.is_last](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket.is_last)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id15 -->

</div>

> <font size="3">set_buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.set_buffer](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket.set_buffer)

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

> <font size="3">parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.parameters](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.GradBucket.parameters)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id21 -->

</div>

</div>

## Default Communication Hooks

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook)

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

**限制与说明**： `input`仅支持fp32

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id27 -->

**限制与说明**： `input`仅支持fp32

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id30 -->

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id33 -->

**限制与说明**：`input`仅支持fp32

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook)

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

</div>

## PowerSGD Communication Hook

### torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id39 -->

</div>

### <code><i>class</i></code> torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState)

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

> <font size="3">__getstate__()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState.__getstate__](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState.__getstate__)

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

</div>

### torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id48 -->

</div>

## Debugging Communication Hooks

### torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook](https://pytorch.org/docs/2.9/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id51 -->

**限制与说明**： `input`仅支持fp32

</div>
