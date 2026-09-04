# torch.distributed.tensor.parallel

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.12/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.12/distributed.tensor.parallel.html)。

<div style="display:none;">

## &#8203;torch.distributed.tensor.parallel

</div>

### torch.distributed.tensor.parallel.parallelize_module

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.parallelize_module](https://pytorch.org/docs/2.12/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.parallelize_module)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.tensor.parallel.ColwiseParallel

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.ColwiseParallel](https://pytorch.org/docs/2.12/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.ColwiseParallel)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### <code><i>class</i></code> torch.distributed.tensor.parallel.RowwiseParallel

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.RowwiseParallel](https://pytorch.org/docs/2.12/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.RowwiseParallel)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.tensor.parallel.PrepareModuleInput

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.PrepareModuleInput](https://pytorch.org/docs/2.12/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.PrepareModuleInput)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.tensor.parallel.PrepareModuleOutput

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.PrepareModuleOutput](https://pytorch.org/docs/2.12/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.PrepareModuleOutput)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.distributed.tensor.parallel.loss_parallel

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.loss_parallel](https://pytorch.org/docs/2.12/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.loss_parallel)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，int64
- 针对<term>Ascend 950DT</term>，当logits数值特别大时，精度可能和<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>存在差异

</div>
