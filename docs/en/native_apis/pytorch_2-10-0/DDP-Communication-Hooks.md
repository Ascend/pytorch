# DDP Communication Hooks

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-14T07:53:13.196Z pushedAt=2026-06-14T09:16:34.706Z -->

> [!NOTE] Note
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.distributed.GradBucket](https://pytorch.org/docs/2.10/ddp_comm_hooks.html#torch.distributed.GradBucket)|Yes|Supports FP32|
|[torch.distributed.GradBucket.index](https://pytorch.org/docs/2.10/ddp_comm_hooks.html#torch.distributed.GradBucket.index)|Yes|Supports FP32|
|[torch.distributed.GradBucket.buffer](https://pytorch.org/docs/2.10/ddp_comm_hooks.html#torch.distributed.GradBucket.buffer)|Yes|Supports FP32|
|[torch.distributed.GradBucket.gradients](https://pytorch.org/docs/2.10/ddp_comm_hooks.html#torch.distributed.GradBucket.gradients)|Yes|Supports FP32|
|[torch.distributed.GradBucket.is_last](https://pytorch.org/docs/2.10/ddp_comm_hooks.html#torch.distributed.GradBucket.is_last)|Yes|-|
|[torch.distributed.GradBucket.set_buffer](https://pytorch.org/docs/2.10/ddp_comm_hooks.html#torch.distributed.GradBucket.set_buffer)|Yes|-|
|[torch.distributed.GradBucket.parameters](https://pytorch.org/docs/2.10/ddp_comm_hooks.html#torch.distributed.GradBucket.parameters)|Yes|-|
|[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook](https://pytorch.org/docs/2.10/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook)|Yes|Supports FP32|
|[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook](https://pytorch.org/docs/2.10/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook)|Yes|Supports FP32|
|[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook](https://pytorch.org/docs/2.10/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook)|Yes|-|
|[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper](https://pytorch.org/docs/2.10/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper)|Yes|Supports FP32|
|[torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper](https://pytorch.org/docs/2.10/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper)|Yes|-|
|[torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState](https://pytorch.org/docs/2.10/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState)|Yes|Supports BF16, FP16, FP32|
|[torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook](https://pytorch.org/docs/2.10/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook)|Yes|-|
|[torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook](https://pytorch.org/docs/2.10/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook)|Yes|-|
|[torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook](https://pytorch.org/docs/2.10/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook)|Yes|Supports FP32|
