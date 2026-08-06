# torch.distributed.tensor.parallel

> [!NOTE]
>
> API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>

## 目录

- [base API](#base-api)

## base API

### torch.distributed.tensor.parallel.parallelize_module

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.parallelize_module](https://pytorch.org/docs/2.7/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.parallelize_module)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### _`class`_ torch.distributed.tensor.parallel.ColwiseParallel

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.ColwiseParallel](https://pytorch.org/docs/2.7/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.ColwiseParallel)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

### _`class`_ torch.distributed.tensor.parallel.RowwiseParallel

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.RowwiseParallel](https://pytorch.org/docs/2.7/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.RowwiseParallel)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### _`class`_ torch.distributed.tensor.parallel.PrepareModuleInput

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.PrepareModuleInput](https://pytorch.org/docs/2.7/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.PrepareModuleInput)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### _`class`_ torch.distributed.tensor.parallel.PrepareModuleOutput

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.PrepareModuleOutput](https://pytorch.org/docs/2.7/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.PrepareModuleOutput)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.distributed.tensor.parallel.loss_parallel

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.tensor.parallel.loss_parallel](https://pytorch.org/docs/2.7/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.loss_parallel)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**：

- `input`仅支持bf16，fp16，fp32，int64
- 针对<term>Ascend 950DT</term>，当logits数值特别大时，精度可能和<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>存在差异

</div>
