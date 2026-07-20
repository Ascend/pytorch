# Private APIs

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T02:24:58.526Z pushedAt=2026-06-15T03:25:49.245Z -->

> [!NOTE]  
> Unless otherwise specified in "Restrictions and Notes", the API is supported across all PyTorch versions. If it is only supported in some PyTorch versions, this will be indicated in "Restrictions and Notes".

## torch

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch._foreach_maximum_|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch._foreach_pow|Yes|Supports bf16, fp16, fp32, int32|
|torch.\_foreach\_pow\_|Yes|Supports bf16, fp16, fp32, int32|
|torch._foreach_tanh|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.\_foreach\_tanh\_|Yes|Supports bf16, fp16, fp32|
|torch.\_foreach\_copy\_|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch._foreach_addcdiv|Yes|Supports bf16, fp16, fp32|
|torch._foreach_div|Yes|Supports bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch._foreach_norm|Yes|Supports bf16, fp16, fp32|
|torch._chunk_cat|Yes|Supports bf16, fp16, fp32|
|torch.split_with_sizes_copy|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch._foreach_add|Yes|Supports bf16, fp16, fp32, int32, bool|
|torch._foreach_lerp|Yes|Supports bf16, fp16, fp32|
|torch.ops.aten._to_copy.default|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|

## torch.amp

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch._amp_foreach_non_finite_check_and_unscale_|Yes|Supports fp16, fp32|
|torch._amp_update_scale_|Yes|Supports fp32|

## torch.distributed

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.distributed._functional_collectives.reduce_scatter_tensor|Yes|Supports bf16, fp16, fp32, int8, int32, int64|
|torch.distributed._reduce_scatter_base|Yes|Supports fp16, fp32, int8, int32, int64|
|torch.distributed.all_reduce_coalesced|Yes|Supports fp16, fp32, uint8, int8, int32, int64, bool, complex64|
|torch.distributed._functional_collectives.AsyncCollectiveTensor|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool, complex64, complex128|

## torch.distributed.nn

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.distributed.nn.all_reduce|Yes|Supports fp16, fp32, uint8, int8, int32, int64, bool, complex64|

## torch.distributed.tensor

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.distributed.tensor._redistribute.redistribute_local_tensor|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool|
|torch.distributed.tensor.DTensor._local_tensor|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|

## torch.distributed.fsdp.fully_shard

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.distributed.fsdp._fully_shard._fsdp_api.ReduceScatter|Yes|Supports bf16, fp16, fp32, int32, int64<br>Only Supported in PyTorch 2.9.0 or Later Versions|
|torch.distributed.fsdp._fully_shard._fsdp_collectives.DefaultReduceScatter|Yes|Supports bf16, fp16, fp32, int32, int64<br>Only Supported in PyTorch 2.9.0 or Later Versions|
|torch.distributed.fsdp._fully_shard._fsdp_collectives.ProcessGroupAllocReduceScatter|Yes|Supports bf16, fp16, fp32, int32, int64<br>Only Supported in PyTorch 2.9.0 or Later Versions|
|torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_reduce|Yes|Supports bf16, fp16, fp32<br>Only Supported in PyTorch 2.8.0 or Later Versions|
