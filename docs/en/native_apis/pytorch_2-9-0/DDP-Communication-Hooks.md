# DDP Communication Hooks

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T02:10:14.717Z pushedAt=2026-06-15T03:25:49.137Z -->

> [!NOTE]  
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.distributed.GradBucket|Yes|Supports fp32|
|torch.distributed.GradBucket.index|Yes|Supports fp32|
|torch.distributed.GradBucket.buffer|Yes|Supports fp32|
|torch.distributed.GradBucket.gradients|Yes|Supports fp32|
|torch.distributed.GradBucket.is_last|Yes|-|
|torch.distributed.GradBucket.set_buffer|Yes|-|
|torch.distributed.GradBucket.parameters|Yes|-|
|torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook|Yes|Supports fp32|
|torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook|Yes|Supports fp32|
|torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook|Yes|-|
|torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper|Yes|Supports fp32|
|torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper|Yes|-|
|torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState|Yes|Supports bf16, fp16, fp32|
|torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook|Yes|-|
|torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook|Yes|-|
|torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook|Yes|Supports fp32|
