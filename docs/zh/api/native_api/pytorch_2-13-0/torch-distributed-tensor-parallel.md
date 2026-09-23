# torch.distributed.tensor.parallel

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.13/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.13/distributed.tensor.parallel.html)。

<div style="display:none;">

## &#8203;torch.distributed.tensor.parallel

</div>

### torch.distributed.tensor.parallel.parallelize_module

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.parallelize_module](https://pytorch.org/docs/2.13/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.parallelize_module)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id3 -->

</div>

### <code><i>class</i></code> torch.distributed.tensor.parallel.ColwiseParallel

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.ColwiseParallel](https://pytorch.org/docs/2.13/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.ColwiseParallel)

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

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

### <code><i>class</i></code> torch.distributed.tensor.parallel.RowwiseParallel

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.RowwiseParallel](https://pytorch.org/docs/2.13/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.RowwiseParallel)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id9 -->

</div>

### <code><i>class</i></code> torch.distributed.tensor.parallel.PrepareModuleInput

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.PrepareModuleInput](https://pytorch.org/docs/2.13/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.PrepareModuleInput)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id12 -->

</div>

### <code><i>class</i></code> torch.distributed.tensor.parallel.PrepareModuleOutput

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.PrepareModuleOutput](https://pytorch.org/docs/2.13/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.PrepareModuleOutput)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id15 -->

</div>

### torch.distributed.tensor.parallel.loss_parallel

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.loss_parallel](https://pytorch.org/docs/2.13/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.loss_parallel)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id18 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，int64

<!-- npu="950,A3,910b" id19 -->
- 针对<term>Ascend 950DT系列产品</term>，当logits数值特别大时，精度可能和<term>Atlas A2训练系列产品</term>/<term>Atlas A3训练系列产品</term>存在差异
<!-- end id19 -->

</div>
