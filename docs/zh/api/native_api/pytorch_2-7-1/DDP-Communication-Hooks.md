# DDP Communication Hooks

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.7/ddp_comm_hooks.html)。

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

**原生文档**：[torch.distributed.GradBucket](https://pytorch.org/docs/2.7/ddp_comm_hooks.html#torch.distributed.GradBucket)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`input`仅支持fp32

> <font size="3">index()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.index](https://pytorch.org/docs/2.7/ddp_comm_hooks.html#torch.distributed.GradBucket.index)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.buffer](https://pytorch.org/docs/2.7/ddp_comm_hooks.html#torch.distributed.GradBucket.buffer)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">gradients()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.gradients](https://pytorch.org/docs/2.7/ddp_comm_hooks.html#torch.distributed.GradBucket.gradients)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">is_last()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.is_last](https://pytorch.org/docs/2.7/ddp_comm_hooks.html#torch.distributed.GradBucket.is_last)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">set_buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.set_buffer](https://pytorch.org/docs/2.7/ddp_comm_hooks.html#torch.distributed.GradBucket.set_buffer)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.parameters](https://pytorch.org/docs/2.7/ddp_comm_hooks.html#torch.distributed.GradBucket.parameters)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## Default Communication Hooks

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook](https://pytorch.org/docs/2.7/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper](https://pytorch.org/docs/2.7/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper](https://pytorch.org/docs/2.7/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook](https://pytorch.org/docs/2.7/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`input`仅支持fp32

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook](https://pytorch.org/docs/2.7/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

## PowerSGD Communication Hook

### torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook](https://pytorch.org/docs/2.7/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState](https://pytorch.org/docs/2.7/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook](https://pytorch.org/docs/2.7/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

## Debugging Communication Hooks

### torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook](https://pytorch.org/docs/2.7/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持fp32

</div>
