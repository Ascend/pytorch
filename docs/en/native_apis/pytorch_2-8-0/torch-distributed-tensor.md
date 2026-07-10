# torch.distributed.tensor

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:30:55.152Z pushedAt=2026-07-09T08:44:08.277Z -->

> [!NOTE]
> If the "Supported" column is "Yes" and "Restrictions and Notes" is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.distributed.tensor.distribute_module|Yes|Supports bf16, fp16, fp32|
|torch.distributed.tensor.distribute_tensor|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool|
|torch.distributed.tensor.DTensor|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.distributed.tensor.DTensor.from_local|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.distributed.tensor.DTensor.redistribute|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool|
|torch.distributed.tensor.placement_types.Shard|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool, complex64, complex128|
|torch.distributed.tensor.placement_types._StridedShard|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool|
