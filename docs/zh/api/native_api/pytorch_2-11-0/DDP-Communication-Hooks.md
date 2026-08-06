# DDP Communication Hooks

> [!NOTE]
>
> API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>

## 目录

- [base API](#base-api)

## base API

### _`class`_ torch.distributed.GradBucket

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket](https://pytorch.org/docs/2.11/ddp_comm_hooks.html#torch.distributed.GradBucket)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： `input`仅支持fp32

> <font size="3">index()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.index](https://pytorch.org/docs/2.11/ddp_comm_hooks.html#torch.distributed.GradBucket.index)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.buffer](https://pytorch.org/docs/2.11/ddp_comm_hooks.html#torch.distributed.GradBucket.buffer)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">gradients()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.gradients](https://pytorch.org/docs/2.11/ddp_comm_hooks.html#torch.distributed.GradBucket.gradients)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">is_last()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.is_last](https://pytorch.org/docs/2.11/ddp_comm_hooks.html#torch.distributed.GradBucket.is_last)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">set_buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.set_buffer](https://pytorch.org/docs/2.11/ddp_comm_hooks.html#torch.distributed.GradBucket.set_buffer)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.GradBucket.parameters](https://pytorch.org/docs/2.11/ddp_comm_hooks.html#torch.distributed.GradBucket.parameters)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook](https://pytorch.org/docs/2.11/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： `input`仅支持fp32

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook](https://pytorch.org/docs/2.11/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： `input`仅支持fp32

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook](https://pytorch.org/docs/2.11/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper](https://pytorch.org/docs/2.11/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： `input`仅支持fp32

</div>

### torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper](https://pytorch.org/docs/2.11/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### _`class`_ torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState](https://pytorch.org/docs/2.11/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook](https://pytorch.org/docs/2.11/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook](https://pytorch.org/docs/2.11/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook](https://pytorch.org/docs/2.11/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： `input`仅支持fp32

</div>
