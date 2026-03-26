# DDP Communication Hooks

> [!NOTE]  
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.distributed.GradBucket|是|支持fp32|
|torch.distributed.GradBucket.index|是|支持fp32|
|torch.distributed.GradBucket.buffer|是|支持fp32|
|torch.distributed.GradBucket.gradients|是|支持fp32|
|torch.distributed.GradBucket.is_last|是|-|
|torch.distributed.GradBucket.set_buffer|是|-|
|torch.distributed.GradBucket.parameters|是|-|
|torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook|是|支持fp32|
|torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook|是|支持fp32|
|torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook|是|-|
|torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper|是|支持fp32|
|torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper|是|-|
|torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState|是|支持bf16，fp16，fp32|
|torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook|是|-|
|torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook|是|-|
|torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook|是|支持fp32|


