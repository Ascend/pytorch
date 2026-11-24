# torch.distributed.fsdp.fully_shard

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 本支持清单的范围以[PyTorch社区文档](https://pytorch.org/docs/2.13/)为准，社区文档中未列出的接口不在本清单范围内；已在社区文档中存在但未收录于本清单的API尚未验证，请谨慎使用，我们将在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [PyTorch FSDP2 (fully_shard)](#pytorch-fsdp2-fully_shard)
- [PyTorch FSDP2 ( fully_shard )](#pytorch-fsdp2--fully_shard-)

</div>

<div style="display:none;">

## &#8203;torch.distributed.fsdp.fully_shard

</div>

## PyTorch FSDP2 (fully_shard)

### torch.distributed.fsdp.fully_shard

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.fully_shard](https://pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.fully_shard)

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

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

<!-- npu="950" id8 -->

<!-- npu="950" id10 -->
### <code><i>class</i></code> torch.distributed.fsdp.MixedPrecisionPolicy

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.MixedPrecisionPolicy](https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.MixedPrecisionPolicy)

**产品支持情况**：

<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id9 -->

</div>

<!-- end id10 -->

<!-- npu="950" id12 -->
### <code><i>class</i></code> torch.distributed.fsdp.OffloadPolicy

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.OffloadPolicy](https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.OffloadPolicy)

**产品支持情况**：

<!-- npu="950" id11 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id11 -->

</div>

<!-- end id12 -->

### <code><i>class</i></code> torch.distributed.fsdp.CPUOffloadPolicy

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.CPUOffloadPolicy](https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.CPUOffloadPolicy)

**产品支持情况**：

<!-- npu="950" id7 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id7 -->

</div>

<!-- end id8 -->

## PyTorch FSDP2 ( fully_shard )

### <code><i>class</i></code> torch.distributed.fsdp.FSDPModule

<div style="margin-left: 2em">

<!-- npu="950" id14 -->
**原生文档**：[torch.distributed.fsdp.FSDPModule](https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule)

**产品支持情况**：

<!-- npu="950" id13 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id13 -->

<!-- end id14 -->

<!-- npu="950" id16 -->
<a class="method_anchor" id="torch_distributed_fsdp_FSDPModule_reshard"></a>

> <font size="3">reshard()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FSDPModule.reshard](https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.reshard)

**产品支持情况**：

<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id15 -->

</div>

<!-- end id16 -->

<!-- npu="950" id18 -->
<a class="method_anchor" id="torch_distributed_fsdp_FSDPModule_set_all_reduce_hook"></a>

> <font size="3">set_all_reduce_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FSDPModule.set\_all\_reduce\_hook](https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_all_reduce_hook)

**产品支持情况**：

<!-- npu="950" id17 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id17 -->

</div>

<!-- end id18 -->

<!-- npu="950" id20 -->
<a class="method_anchor" id="torch_distributed_fsdp_FSDPModule_set_is_last_backward"></a>

> <font size="3">set_is_last_backward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FSDPModule.set\_is\_last\_backward](https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_is_last_backward)

**产品支持情况**：

<!-- npu="950" id19 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id19 -->

</div>

<!-- end id20 -->

<!-- npu="950" id22 -->
<a class="method_anchor" id="torch_distributed_fsdp_FSDPModule_set_modules_to_backward_prefetch"></a>

> <font size="3">set_modules_to_backward_prefetch()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FSDPModule.set\_modules\_to\_backward\_prefetch](https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_modules_to_backward_prefetch)

**产品支持情况**：

<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id21 -->

</div>

<!-- end id22 -->

<!-- npu="950" id24 -->
<a class="method_anchor" id="torch_distributed_fsdp_FSDPModule_set_modules_to_forward_prefetch"></a>

> <font size="3">set_modules_to_forward_prefetch()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FSDPModule.set\_modules\_to\_forward\_prefetch](https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_modules_to_forward_prefetch)

**产品支持情况**：

<!-- npu="950" id23 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id23 -->

</div>

<!-- end id24 -->

<!-- npu="950" id26 -->
<a class="method_anchor" id="torch_distributed_fsdp_FSDPModule_set_post_optim_event"></a>

> <font size="3">set_post_optim_event()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FSDPModule.set\_post\_optim\_event](https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_post_optim_event)

**产品支持情况**：

<!-- npu="950" id25 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id25 -->

</div>

<!-- end id26 -->

<!-- npu="950" id28 -->
<a class="method_anchor" id="torch_distributed_fsdp_FSDPModule_set_reduce_scatter_divide_factor"></a>

> <font size="3">set_reduce_scatter_divide_factor()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FSDPModule.set\_reduce\_scatter\_divide\_factor](https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_reduce_scatter_divide_factor)

**产品支持情况**：

<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id27 -->

</div>

<!-- end id28 -->

<!-- npu="950" id30 -->
<a class="method_anchor" id="torch_distributed_fsdp_FSDPModule_set_requires_all_reduce"></a>

> <font size="3">set_requires_all_reduce()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FSDPModule.set\_requires\_all\_reduce](https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_requires_all_reduce)

**产品支持情况**：

<!-- npu="950" id29 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id29 -->

</div>

<!-- end id30 -->

<!-- npu="950" id32 -->
<a class="method_anchor" id="torch_distributed_fsdp_FSDPModule_set_requires_gradient_sync"></a>

> <font size="3">set_requires_gradient_sync()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FSDPModule.set\_requires\_gradient\_sync](https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_requires_gradient_sync)

**产品支持情况**：

<!-- npu="950" id31 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id31 -->

</div>

<!-- end id32 -->

<!-- npu="950" id34 -->
<a class="method_anchor" id="torch_distributed_fsdp_FSDPModule_set_reshard_after_backward"></a>

> <font size="3">set_reshard_after_backward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FSDPModule.set\_reshard\_after\_backward](https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_reshard_after_backward)

**产品支持情况**：

<!-- npu="950" id33 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id33 -->

</div>

<!-- end id34 -->

<!-- npu="950" id36 -->
<a class="method_anchor" id="torch_distributed_fsdp_FSDPModule_set_unshard_in_backward"></a>

> <font size="3">set_unshard_in_backward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FSDPModule.set\_unshard\_in\_backward](https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_unshard_in_backward)

**产品支持情况**：

<!-- npu="950" id35 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id35 -->

</div>

<!-- end id36 -->

<a class="method_anchor" id="torch_distributed_fsdp_FSDPModule_unshard"></a>

> <font size="3">unshard()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FSDPModule.unshard](https://docs.pytorch.org/docs/2.13/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.unshard)

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

</div>

</div>
