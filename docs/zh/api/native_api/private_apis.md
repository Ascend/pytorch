# Private APIs

> [!NOTE]  
> 未在“限制与说明”中特殊说明的为全PyTorch版本支持，若仅支持部分PyTorch版本会标识在“限制与说明”中。

## torch

### torch._foreach_maximum_

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id3 -->

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### torch._foreach_pow

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id6 -->

**限制与说明**：

- 支持bf16，fp16，fp32，int32

</div>

### torch._foreach_pow_

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：暂不支持
<!-- end id9 -->

**限制与说明**：

- 支持bf16，fp16，fp32，int32

</div>

### torch._foreach_tanh

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id12 -->

**限制与说明**：

- 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch._foreach_tanh_

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：暂不支持
<!-- end id15 -->

**限制与说明**：

- 支持bf16，fp16，fp32

</div>

### torch._foreach_copy_

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：暂不支持
<!-- end id18 -->

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch._foreach_addcdiv

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id21 -->

**限制与说明**：

- 支持bf16，fp16，fp32

</div>

### torch._foreach_div

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id24 -->

**限制与说明**：

- 支持bf16，fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch._foreach_norm

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id27 -->

**限制与说明**：

- 支持bf16，fp16，fp32

</div>

### torch._chunk_cat

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id30 -->

**限制与说明**：

- 支持bf16，fp16，fp32

</div>

### torch.split_with_sizes_copy

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id33 -->

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch._foreach_add

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id36 -->

**限制与说明**：

- 支持bf16，fp16，fp32，int32，bool

</div>

### torch._foreach_lerp

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id39 -->

**限制与说明**：

- 支持bf16，fp16，fp32

</div>

### torch.ops.aten._to_copy.default

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id42 -->

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch._scaled_mm

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT系列产品</term>：暂不支持
<!-- end id45 -->

**限制与说明**：

- 支持fp8模式下ScalingType为tensorwise，rowwise和BlockWise1x128，mxfp8模式下ScalingType为BlockWise1x32的排布，mxfp8遵循[aclnnQuantMatmulV5](https://gitcode.com/cann/ops-nn/blob/master/matmul/quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md)要求（scale_a和scale_b详见约束说明）
- 仅支持PyTorch 2.7.1以上版本

</div>

### torch._scaled_mm_v2

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT系列产品</term>：暂不支持
<!-- end id48 -->

**限制与说明**：

- 支持fp8模式下ScalingType为tensorwise，rowwise和BlockWise1x128，mxfp8模式下ScalingType为BlockWise1x32的排布，mxfp8遵循[aclnnQuantMatmulV5](https://gitcode.com/cann/ops-nn/blob/master/matmul/quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md)要求（swizzle_a和swizzle_b必须为None，scale_a和scale_b详见约束说明）
- 仅支持PyTorch 2.10.0以上版本

</div>

### torch._scaled_grouped_mm

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT系列产品</term>：暂不支持
<!-- end id51 -->

**限制与说明**：

- 支持fp8模式下ScalingType为rowwise，mxfp8模式的排布，mxfp8遵循[aclnnGroupedMatmulV5](https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md)要求（scale_a和scale_b详见约束说明）
- 仅支持PyTorch 2.7.1以上版本

</div>

### torch._scaled_grouped_mm_v2

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT系列产品</term>：暂不支持
<!-- end id54 -->

**限制与说明**：

- 支持fp8模式下ScalingType为rowwise，mxfp8模式的排布，mxfp8遵循[aclnnGroupedMatmulV5](https://gitcode.com/cann/ops-transformer/blob/master/gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md)要求（swizzle_a和swizzle_b必须为None，scale_a和scale_b详见约束说明）
- 仅支持PyTorch 2.10.0以上版本

</div>

## torch.amp

### torch._amp_foreach_non_finite_check_and_unscale_

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id57 -->

**限制与说明**：

- 支持fp16，fp32

</div>

### torch._amp_update_scale_

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT系列产品</term>：暂不支持
<!-- end id60 -->

**限制与说明**：

- 支持fp32

</div>

## torch.cuda

### torch.cuda.reset_accumulated_host_memory_stats

<div style="margin-left: 2em">

**NPU形式名称**：`torch_npu.npu.reset_accumulated_host_memory_stats`

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT系列产品</term>：暂不支持
<!-- end id63 -->

**限制与说明**：

- 此接口自PyTorch2.9.0版本开始修改为公开接口

</div>

### torch.cuda.host_memory_stats_as_nested_dict

<div style="margin-left: 2em">

**NPU形式名称**：`torch_npu.npu.host_memory_stats_as_nested_dict`

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT系列产品</term>：暂不支持
<!-- end id66 -->

**限制与说明**：

- 此接口自PyTorch2.9.0版本开始修改为公开接口

</div>

## torch.distributed

### torch.distributed._functional_collectives.reduce_scatter_tensor

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id69 -->

**限制与说明**：

- 支持bf16，fp16，fp32，int8，int32，int64

</div>

### torch.distributed._reduce_scatter_base

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id70 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id72 -->

**限制与说明**：

- 支持fp16，fp32，int8，int32，int64

</div>

### torch.distributed.all_reduce_coalesced

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id73 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id75 -->

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int32，int64，bool，complex64

</div>

### torch.distributed._functional_collectives.AsyncCollectiveTensor

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id76 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id78 -->

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64，bool，complex64，complex128

</div>

## torch.distributed.nn

### torch.distributed.nn.all_reduce

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id79 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id81 -->

**限制与说明**：

- 支持fp16，fp32，uint8，int8，int32，int64，bool，complex64

</div>

## torch.distributed.tensor

### torch.distributed.tensor._redistribute.redistribute_local_tensor

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id84 -->

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64，bool

</div>

### torch.distributed.tensor.DTensor._local_tensor

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id85 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id87 -->

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.distributed.tensor.placement_types._StridedShard

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT系列产品</term>：暂不支持
<!-- end id90 -->

**限制与说明**：

- 支持bf16，fp16，fp32，fp64，uint8，int8，int32，int64，bool
- 此接口自PyTorch2.11.0版本开始修改为公开接口

</div>

## torch.distributed.fsdp.fully_shard

### torch.distributed.fsdp._fully_shard._fsdp_api.ReduceScatter

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id91 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT系列产品</term>：暂不支持
<!-- end id93 -->

**限制与说明**：

- 支持bf16，fp16，fp32，int32，int64
- 仅支持PyTorch 2.9.0以上版本

</div>

### torch.distributed.fsdp._fully_shard._fsdp_collectives.DefaultReduceScatter

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id94 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT系列产品</term>：暂不支持
<!-- end id96 -->

**限制与说明**：

- 支持bf16，fp16，fp32，int32，int64
- 仅支持PyTorch 2.9.0以上版本

</div>

### torch.distributed.fsdp._fully_shard._fsdp_collectives.ProcessGroupAllocReduceScatter

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id97 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT系列产品</term>：暂不支持
<!-- end id99 -->

**限制与说明**：

- 支持bf16，fp16，fp32，int32，int64
- 仅支持PyTorch 2.9.0以上版本

</div>

### torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_reduce

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id100 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT系列产品</term>：暂不支持
<!-- end id102 -->

**限制与说明**：

- 支持bf16，fp16，fp32
- 仅支持PyTorch 2.8.0以上版本

</div>

## torch.fx

### torch.fx.proxy.ParameterProxy

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id103 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT系列产品</term>：暂不支持
<!-- end id105 -->

**限制与说明**：

- -

</div>

### torch.fx.passes.split_module.split_module

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id106 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT系列产品</term>：暂不支持
<!-- end id108 -->

**限制与说明**：

- -

</div>

### torch.fx.passes.regional_inductor.regional_inductor

<div style="margin-left: 2em">

**产品支持情况**：

<!-- npu="910b" id109 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT系列产品</term>：暂不支持
<!-- end id111 -->

**限制与说明**：

- 仅支持PyTorch 2.10.0以上版本
- 此接口自PyTorch2.12.0版本开始修改为公开接口

</div>
