# Private APIs

> [!NOTE]  
> Unless otherwise specified in "Constraints & Notes", the API is supported in all PyTorch versions. If an API is only supported in some PyTorch versions, this will be indicated in "Constraints & Notes".

## torch

|API Name|Supported|Constraints & Notes|
|--|--|--|
|torch._foreach_maximum_|Yes|Supported: bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch._foreach_pow|Yes|Supported: bf16, fp16, fp32, int32|
|torch.\_foreach\_pow\_|Yes<br>Currently not supported on <term>Ascend 950DT</term>|Supported: bf16, fp16, fp32, int32|
|torch._foreach_tanh|Yes|Supported: bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.\_foreach\_tanh\_|Yes<br>Currently not supported on <term>Ascend 950DT</term>|Supported: bf16, fp16, fp32|
|torch.\_foreach\_copy\_|Yes<br>Currently not supported on <term>Ascend 950DT</term>|Supported: bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch._foreach_addcdiv|Yes|Supported: bf16, fp16, fp32|
|torch._foreach_div|Yes|Supported: bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch._foreach_norm|Yes|Supported: bf16, fp16, fp32|
|torch._chunk_cat|Yes|Supported: bf16, fp16, fp32|
|torch.split_with_sizes_copy|Yes|Supported: fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch._foreach_add|Yes|Supported: bf16, fp16, fp32, int32, bool|
|torch._foreach_lerp|Yes|Supported: bf16, fp16, fp32|
|torch.ops.aten._to_copy.default|Yes|Supported: bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch._scaled_mm|Yes<br>Currently not supported on <term>Ascend 950DT</term>|Supported: layouts with ScalingType of tensorwise, rowwise, and BlockWise1x128 in fp8 mode, and with ScalingType of BlockWise1x32 in mxfp8 mode. mxfp8 follows the requirements of [aclnnQuantMatmulV5](https://gitcode.com/cann/ops-nn/blob/9.0.0/matmul/quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md) (for details about scale_a and scale_b, see the constraints)<br>Only supports PyTorch 2.7.1 and later|
|torch._scaled_mm_v2|Yes<br>Currently not supported on <term>Ascend 950DT</term>|Supported: layouts with ScalingType of tensorwise, rowwise, and BlockWise1x128 in fp8 mode, and with ScalingType of BlockWise1x32 in mxfp8 mode. mxfp8 follows the requirements of [aclnnQuantMatmulV5](https://gitcode.com/cann/ops-nn/blob/9.0.0/matmul/quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md) (swizzle_a and swizzle_b must be None; for details about scale_a and scale_b, see the constraints)<br>Only supports PyTorch 2.10.0 and later|
|torch._scaled_grouped_mm|Yes<br>Currently not supported on <term>Ascend 950DT</term>|Supported: layouts with ScalingType of rowwise in fp8 mode and in mxfp8 mode. mxfp8 follows the requirements of [aclnnGroupedMatmulV5](https://gitcode.com/cann/ops-nn/blob/9.0.0/matmul/quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md) (for details about scale_a and scale_b, see the constraints)<br>Only supports PyTorch 2.7.1 and later|
|torch._scaled_grouped_mm_v2|Yes<br>Currently not supported on <term>Ascend 950DT</term>|Supported: layouts with ScalingType of rowwise in fp8 mode and in mxfp8 mode. mxfp8 follows the requirements of [aclnnGroupedMatmulV5](https://gitcode.com/cann/ops-transformer/blob/9.0.0/gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md) (swizzle_a and swizzle_b must be None; for details about scale_a and scale_b, see the constraints)<br>Only supports PyTorch 2.10.0 and later|

## torch.amp

|API Name|Supported|Constraints & Notes|
|--|--|--|
|torch._amp_foreach_non_finite_check_and_unscale_|Yes|Supported: fp16, fp32|
|torch.\_amp\_update\_scale\_|Yes<br>Currently not supported on <term>Ascend 950DT</term>|Supported: fp32|

## torch.cuda

|API Name|NPU Form Name|Supported|Constraints & Notes|
|--|--|--|--|
|torch.cuda.reset_accumulated_host_memory_stats| torch_npu.npu.reset_accumulated_host_memory_stats|Yes<br>Currently not supported on <term>Ascend 950DT</term>|This API has been changed to a public API since PyTorch 2.9.0|
|torch.cuda.host_memory_stats_as_nested_dict|torch_npu.npu.host_memory_stats_as_nested_dict|Yes<br>Currently not supported on <term>Ascend 950DT</term>|This API has been changed to a public API since PyTorch 2.9.0|

## torch.distributed

|API Name|Supported|Constraints & Notes|
|--|--|--|
|torch.distributed._functional_collectives.reduce_scatter_tensor|Yes|Supported: bf16, fp16, fp32, int8, int32, int64|
|torch.distributed._reduce_scatter_base|Yes|Supported: fp16, fp32, int8, int32, int64|
|torch.distributed.all_reduce_coalesced|Yes|Supported: fp16, fp32, uint8, int8, int32, int64, bool, complex64|
|torch.distributed._functional_collectives.AsyncCollectiveTensor|Yes|Supported: bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool, complex64, complex128|

## torch.distributed.nn

|API Name|Supported|Constraints & Notes|
|--|--|--|
|torch.distributed.nn.all_reduce|Yes|Supported: fp16, fp32, uint8, int8, int32, int64, bool, complex64|

## torch.distributed.tensor

|API Name|Supported|Constraints & Notes|
|--|--|--|
|torch.distributed.tensor._redistribute.redistribute_local_tensor|Yes|Supported: bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool|
|torch.distributed.tensor.DTensor._local_tensor|Yes|Supported: bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128|
|torch.distributed.tensor.placement_types._StridedShard|Yes<br>Currently not supported on <term>Ascend 950DT</term>|Supported: bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool<br>This API has been changed to a public API since PyTorch 2.11.0|

## torch.distributed.fsdp.fully_shard

|API Name|Supported|Constraints & Notes|
|--|--|--|
|torch.distributed.fsdp._fully_shard._fsdp_api.ReduceScatter|Yes<br>Currently not supported on <term>Ascend 950DT</term>|Supported: bf16, fp16, fp32, int32, int64<br>Only supports PyTorch 2.9.0 and later|
|torch.distributed.fsdp._fully_shard._fsdp_collectives.DefaultReduceScatter|Yes<br>Currently not supported on <term>Ascend 950DT</term>|Supported: bf16, fp16, fp32, int32, int64<br>Only supports PyTorch 2.9.0 and later|
|torch.distributed.fsdp._fully_shard._fsdp_collectives.ProcessGroupAllocReduceScatter|Yes<br>Currently not supported on <term>Ascend 950DT</term>|Supported: bf16, fp16, fp32, int32, int64<br>Only supports PyTorch 2.9.0 and later|
|torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_reduce|Yes<br>Currently not supported on <term>Ascend 950DT</term>|Supported: bf16, fp16, fp32<br>Only supports PyTorch 2.8.0 and later|

## torch.fx

|API Name|Supported|Constraints & Notes|
|--|--|--|
|torch.fx.proxy.ParameterProxy|Yes<br>Currently not supported on <term>Ascend 950DT</term>|-|
|torch.fx.passes.split_module.split_module|Yes<br>Currently not supported on <term>Ascend 950DT</term>|-|
|torch.fx.passes.regional_inductor.regional_inductor|Yes<br>Currently not supported on <term>Ascend 950DT</term>|Only supports PyTorch 2.10.0 and later<br>This API has been changed to a public API since PyTorch 2.12.0|
