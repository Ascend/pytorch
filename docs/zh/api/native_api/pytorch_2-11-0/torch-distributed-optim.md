# torch.distributed.optim

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.11/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.11/distributed.optim.html)。

<div style="display:none;">

## &#8203;torch.distributed.optim

</div>

### <code><i>class</i></code> torch.distributed.optim.DistributedOptimizer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.DistributedOptimizer](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.DistributedOptimizer)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.DistributedOptimizer.step](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.DistributedOptimizer.step)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.distributed.optim.PostLocalSGDOptimizer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.PostLocalSGDOptimizer](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.PostLocalSGDOptimizer.load_state_dict](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer.load_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.PostLocalSGDOptimizer.state_dict](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer.state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.PostLocalSGDOptimizer.step](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer.step)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

### <code><i>class</i></code> torch.distributed.optim.ZeroRedundancyOptimizer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- 支持的输入类型为`torch.nn.Optimizer`对象
- 不支持NPU融合优化器对象

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.add_param_group](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.add_param_group)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">consolidate_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.consolidate_state_dict](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.consolidate_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">join_device()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.join_device](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.join_device)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">join_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.join_hook](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.join_hook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">join_process_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.join_process_group](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.join_process_group)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.load_state_dict](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.load_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.state_dict](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.step](https://pytorch.org/docs/2.11/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.step)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>
