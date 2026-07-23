# torch.distributed.tensor.parallel

> [!NOTE]  
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)

## base API

### torch.distributed.tensor.parallel.parallelize_module

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.parallelize_module](https://pytorch.org/docs/2.11/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.parallelize_module)

**是否支持**：否

</div>

### _`class`_ torch.distributed.tensor.parallel.ColwiseParallel

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.ColwiseParallel](https://pytorch.org/docs/2.11/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.ColwiseParallel)

**是否支持**：是

**限制与说明**： 支持bf16，fp16，fp32

</div>

### _`class`_ torch.distributed.tensor.parallel.RowwiseParallel

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.RowwiseParallel](https://pytorch.org/docs/2.11/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.RowwiseParallel)

**是否支持**：否

</div>

### _`class`_ torch.distributed.tensor.parallel.PrepareModuleInput

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.PrepareModuleInput](https://pytorch.org/docs/2.11/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.PrepareModuleInput)

**是否支持**：否

</div>

### _`class`_ torch.distributed.tensor.parallel.PrepareModuleOutput

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.PrepareModuleOutput](https://pytorch.org/docs/2.11/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.PrepareModuleOutput)

**是否支持**：是

</div>

### torch.distributed.tensor.parallel.loss_parallel

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.loss_parallel](https://pytorch.org/docs/2.11/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.loss_parallel)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，int64
- 针对<term>Ascend 950DT</term>，当logits数值特别大时，精度可能和<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>存在差异

</div>
