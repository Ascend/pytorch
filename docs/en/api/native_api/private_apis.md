# Private APIs

> [!NOTE]  
> Unless otherwise specified in "Constraints & Notes", the API is supported in all PyTorch versions. If an API is only supported in some PyTorch versions, this will be indicated in "Constraints & Notes".

## torch

### torch.\_foreach_maximum\_

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id3 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64

</div>

### torch._foreach_pow

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id6 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32, int32

</div>

### torch.\_foreach_pow\_

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id9 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32, int32

</div>

### torch._foreach_tanh

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id12 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.\_foreach_tanh\_

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id15 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32

</div>

### torch.\_foreach_copy\_

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id18 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch._foreach_addcdiv

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id21 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32

</div>

### torch._foreach_div

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id24 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch._foreach_norm

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id27 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32

</div>

### torch._chunk_cat

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id30 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32

</div>

### torch.split_with_sizes_copy

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id33 -->

**Constraints & Notes**:

- Supported: fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch._foreach_add

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id36 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32, int32, bool

</div>

### torch._foreach_lerp

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id39 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32

</div>

### torch.ops.aten._to_copy.default

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id42 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch._scaled_mm

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id45 -->

**Constraints & Notes**:

- Supported: layouts with ScalingType of tensorwise, rowwise, and BlockWise1x128 in fp8 mode, and with ScalingType of BlockWise1x32 in mxfp8 mode. mxfp8 follows the requirements of [aclnnQuantMatmulV5](https://gitcode.com/cann/ops-nn/blob/master/matmul/quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md) (for details about scale_a and scale_b, see the constraints)
- Only supports PyTorch 2.7.1 and later

</div>

### torch._scaled_mm_v2

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id46 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id48 -->

**Constraints & Notes**:

- Supported: layouts with ScalingType of tensorwise, rowwise, and BlockWise1x128 in fp8 mode, and with ScalingType of BlockWise1x32 in mxfp8 mode. mxfp8 follows the requirements of [aclnnQuantMatmulV5](https://gitcode.com/cann/ops-nn/blob/master/matmul/quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md) (swizzle_a and swizzle_b must be None; for details about scale_a and scale_b, see the constraints)
- Only supports PyTorch 2.10.0 and later

</div>

### torch._scaled_grouped_mm

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id49 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id51 -->

**Constraints & Notes**:

- Supported: layouts with ScalingType of rowwise in fp8 mode and in mxfp8 mode. mxfp8 follows the requirements of [aclnnGroupedMatmulV5](https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md) (for details about scale_a and scale_b, see the constraints)
- Only supports PyTorch 2.7.1 and later

</div>

### torch._scaled_grouped_mm_v2

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id52 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id54 -->

**Constraints & Notes**:

- Supported: layouts with ScalingType of rowwise in fp8 mode and in mxfp8 mode. mxfp8 follows the requirements of [aclnnGroupedMatmulV5](https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md) (swizzle_a and swizzle_b must be None; for details about scale_a and scale_b, see the constraints)
- Only supports PyTorch 2.10.0 and later

</div>

## torch.amp

### torch.\_amp_foreach_non_finite_check_and_unscale\_

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id55 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id57 -->

**Constraints & Notes**:

- Supported: fp16, fp32

</div>

### torch.\_amp_update_scale\_

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id58 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id60 -->

**Constraints & Notes**:

- Supported: fp32

</div>

## torch.cuda

### torch.cuda.reset_accumulated_host_memory_stats

<div style="margin-left: 2em">

**NPU Form Name**: `torch_npu.npu.reset_accumulated_host_memory_stats`

**Support Status**:

<!-- npu="910b" id61 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id63 -->

**Constraints & Notes**:

- This API has been changed to a public API since PyTorch 2.9.0.

</div>

### torch.cuda.host_memory_stats_as_nested_dict

<div style="margin-left: 2em">

**NPU Form Name**: `torch_npu.npu.host_memory_stats_as_nested_dict`

**Support Status**:

<!-- npu="910b" id64 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id66 -->

**Constraints & Notes**:

- This API has been changed to a public API since PyTorch 2.9.0.

</div>

## torch.distributed

### torch.distributed._functional_collectives.reduce_scatter_tensor

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id67 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id69 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32, int8, int32, int64

</div>

### torch.distributed._reduce_scatter_base

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id70 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id72 -->

**Constraints & Notes**:

- Supported: fp16, fp32, int8, int32, int64

</div>

### torch.distributed.all_reduce_coalesced

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id73 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id75 -->

**Constraints & Notes**:

- Supported: fp16, fp32, uint8, int8, int32, int64, bool, complex64

</div>

### torch.distributed._functional_collectives.AsyncCollectiveTensor

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id76 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id78 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool, complex64, complex128

</div>

## torch.distributed.nn

### torch.distributed.nn.all_reduce

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id79 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id81 -->

**Constraints & Notes**:

- Supported: fp16, fp32, uint8, int8, int32, int64, bool, complex64

</div>

## torch.distributed.tensor

### torch.distributed.tensor._redistribute.redistribute_local_tensor

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id82 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id84 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool

</div>

### torch.distributed.tensor.DTensor._local_tensor

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id85 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id87 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.distributed.tensor.placement_types._StridedShard

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id88 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id90 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32, fp64, uint8, int8, int32, int64, bool
- This API has been changed to a public API since PyTorch 2.11.0.

</div>

## torch.distributed.fsdp.fully_shard

### torch.distributed.fsdp._fully_shard._fsdp_api.ReduceScatter

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id91 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id93 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32, int32, int64
- Only supports PyTorch 2.9.0 and later

</div>

### torch.distributed.fsdp._fully_shard._fsdp_collectives.DefaultReduceScatter

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id94 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id96 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32, int32, int64
- Only supports PyTorch 2.9.0 and later

</div>

### torch.distributed.fsdp._fully_shard._fsdp_collectives.ProcessGroupAllocReduceScatter

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id97 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id99 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32, int32, int64
- Only supports PyTorch 2.9.0 and later

</div>

### torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_reduce

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id100 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id102 -->

**Constraints & Notes**:

- Supported: bf16, fp16, fp32
- Only supports PyTorch 2.8.0 and later

</div>

## torch.fx

### torch.fx.proxy.ParameterProxy

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id103 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id105 -->

**Constraints & Notes**:

- -

</div>

### torch.fx.passes.split_module.split_module

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id106 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id108 -->

**Constraints & Notes**:

- -

</div>

### torch.fx.passes.regional_inductor.regional_inductor

<div style="margin-left: 2em">

**Support Status**:

<!-- npu="910b" id109 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id111 -->

**Constraints & Notes**:

- Only supports PyTorch 2.10.0 and later
- This API has been changed to a public API since PyTorch 2.12.0.

</div>
