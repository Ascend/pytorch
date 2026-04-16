# Private APIs

> [!NOTE]  
> 未在“限制与说明”中特殊说明的为全PyTorch版本支持，若仅支持部分PyTorch版本会标识在“限制与说明”中。

## torch

|API名称|是否支持|限制与说明|
|--|--|--|
|torch._foreach_maximum_|是|支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64|
|torch._foreach_pow|是|支持bf16，fp16，fp32，int32|
|torch.\_foreach\_pow\_|是|支持bf16，fp16，fp32，int32|
|torch._foreach_tanh|是|支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool|
|torch.\_foreach\_tanh\_|是|支持bf16，fp16，fp32|
|torch.\_foreach\_copy\_|是|支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128|
|torch._foreach_addcdiv|是|支持bf16，fp16，p32|
|torch._foreach_div|是|支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool|
|torch._foreach_norm|是|支持bf16，fp16，fp32|
|torch._chunk_cat|是|支持bf16，fp16，fp32|
|torch.split_with_sizes_copy|是|支持fp16，fp32，uint8，int8，int16，int32，int64，bool|
|torch._foreach_add|是|支持bf16，fp16，fp32，int32，bool|
|torch._foreach_lerp|是|支持bf16，fp16，fp32|
|torch.ops.aten._to_copy.default|是|支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128|

## torch.amp

|API名称|是否支持|限制与说明|
|--|--|--|
|torch._amp_foreach_non_finite_check_and_unscale_|是|支持fp16，fp32|
|torch.\_amp\_update\_scale\_|是|支持fp32|

## torch.distributed

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.distributed._functional_collectives.reduce_scatter_tensor|是|支持bf16，fp16，fp32，int8，int32，int64|
|torch.distributed._reduce_scatter_base|是|支持fp16，fp32，int8，int32，int64|
|torch.distributed.all_reduce_coalesced |是|支持fp16，fp32，uint8，int8，int32，int64，bool，complex64|
|torch.distributed._functional_collectives.AsyncCollectiveTensor|是|支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64，bool，complex64，complex128|

## torch.distributed.nn

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.distributed.nn.all_reduce|是|支持fp16，fp32，uint8，int8，int32，int64，bool，complex64|

## torch.distributed.tensor

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.distributed.tensor._redistribute.redistribute_local_tensor|是|支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64，bool|
|torch.distributed.tensor.DTensor._local_tensor|是|支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128|

## torch.distributed.fsdp.fully_shard

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.distributed.fsdp._fully_shard._fsdp_api.ReduceScatter|是|支持bf16，fp16，fp32，int32，int64<br>仅支持PyTorch 2.9.0以上版本|
|torch.distributed.fsdp._fully_shard._fsdp_collectives.DefaultReduceScatter|是|支持bf16，fp16，fp32，int32，int64<br>仅支持PyTorch 2.9.0以上版本|
|torch.distributed.fsdp._fully_shard._fsdp_collectives.ProcessGroupAllocReduceScatter|是|支持bf16，fp16，fp32，int32，int64<br>仅支持PyTorch 2.9.0以上版本|
|torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_reduce|是|支持bf16，fp16，fp32<br>仅支持PyTorch 2.8.0以上版本|
