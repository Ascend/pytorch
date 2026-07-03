# torch.distributed.tensor

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T02:15:45.771Z pushedAt=2026-06-15T03:25:49.169Z -->

> [!NOTE]  
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.distributed.tensor.distribute_module](https://pytorch.org/docs/2.9/distributed.tensor.html#torch.distributed.tensor.distribute_module)|Yes|Supports bf16, fp16, fp32|
|[torch.distributed.tensor.distribute_tensor](https://pytorch.org/docs/2.9/distributed.tensor.html#torch.distributed.tensor.distribute_tensor)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool|
|[torch.distributed.tensor.DTensor](https://pytorch.org/docs/2.9/distributed.tensor.html#torch.distributed.tensor.DTensor)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.distributed.tensor.DTensor.from_local](https://pytorch.org/docs/2.9/distributed.tensor.html#torch.distributed.tensor.DTensor.from_local)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|[torch.distributed.tensor.DTensor.redistribute](https://pytorch.org/docs/2.9/distributed.tensor.html#torch.distributed.tensor.DTensor.redistribute)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool|
|[torch.distributed.tensor.placement_types.Shard](https://pytorch.org/docs/2.9/distributed.tensor.html#torch.distributed.tensor.placement_types.Shard)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool, complex64, complex128|
|[torch.distributed.tensor.placement_types._StridedShard](https://pytorch.org/docs/2.9/distributed.tensor.html#torch.distributed.tensor.placement_types._StridedShard)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool|
|[torch.distributed.tensor.experimental.context_parallel](https://pytorch.org/docs/2.9/distributed.tensor.html#torch.distributed.tensor.experimental.context_parallel)|Yes|Only supports the NPU fused SDPA path; q/k/v only supports BNSD layout; pse, padding_mask, prefix, actual_seq_qlen, actual_seq_kvlen, sink, and any non-causal attention mask are not supported; load balance requires causal attention|
