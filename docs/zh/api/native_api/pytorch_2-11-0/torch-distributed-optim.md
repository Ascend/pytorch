# torch.distributed.optim

> [!NOTE]
>
> API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>

## 目录

- [base API](#base-api)

## base API

### _`class`_ torch.distributed.optim.DistributedOptimizer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.DistributedOptimizer](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.DistributedOptimizer)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.DistributedOptimizer.step](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.DistributedOptimizer.step)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

</div>

### _`class`_ torch.distributed.optim.PostLocalSGDOptimizer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.PostLocalSGDOptimizer](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.PostLocalSGDOptimizer.load_state_dict](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer.load_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.PostLocalSGDOptimizer.state_dict](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer.state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.PostLocalSGDOptimizer.step](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer.step)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

</div>

### _`class`_ torch.distributed.optim.ZeroRedundancyOptimizer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**：

- 支持的输入类型为`torch.nn.Optimizer`对象
- 不支持NPU融合优化器对象

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.add_param_group](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.add_param_group)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">consolidate_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.consolidate_state_dict](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.consolidate_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">join_device()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.join_device](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.join_device)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">join_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.join_hook](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.join_hook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">join_process_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.join_process_group](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.join_process_group)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.load_state_dict](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.load_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.state_dict](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.step](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.step)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

</div>
