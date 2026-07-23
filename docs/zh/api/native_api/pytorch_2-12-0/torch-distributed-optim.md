# torch.distributed.optim

> [!NOTE]  
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)

## base API

### _`class`_ torch.distributed.optim.DistributedOptimizer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.DistributedOptimizer](https://pytorch.org/docs/2.12/distributed.optim.html#torch.distributed.optim.DistributedOptimizer)

**是否支持**：否

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.DistributedOptimizer.step](https://pytorch.org/docs/2.12/distributed.optim.html#torch.distributed.optim.DistributedOptimizer.step)

**是否支持**：否

</div>

</div>

### _`class`_ torch.distributed.optim.PostLocalSGDOptimizer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.PostLocalSGDOptimizer](https://pytorch.org/docs/2.12/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer)

**是否支持**：是

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.PostLocalSGDOptimizer.load_state_dict](https://pytorch.org/docs/2.12/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer.load_state_dict)

**是否支持**：是

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.PostLocalSGDOptimizer.state_dict](https://pytorch.org/docs/2.12/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer.state_dict)

**是否支持**：是

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.PostLocalSGDOptimizer.step](https://pytorch.org/docs/2.12/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer.step)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributed.optim.ZeroRedundancyOptimizer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer](https://pytorch.org/docs/2.12/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer)

**是否支持**：是

**限制与说明**：

- 支持的输入类型为torch.nn.Optimizer对象
- 不支持NPU融合优化器对象

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.add_param_group](https://pytorch.org/docs/2.12/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.add_param_group)

**是否支持**：是

</div>

> <font size="3">consolidate_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.consolidate_state_dict](https://pytorch.org/docs/2.12/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.consolidate_state_dict)

**是否支持**：是

</div>

> <font size="3">join_device()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.join_device](https://pytorch.org/docs/2.12/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.join_device)

**是否支持**：是

</div>

> <font size="3">join_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.join_hook](https://pytorch.org/docs/2.12/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.join_hook)

**是否支持**：是

</div>

> <font size="3">join_process_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.join_process_group](https://pytorch.org/docs/2.12/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.join_process_group)

**是否支持**：是

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.load_state_dict](https://pytorch.org/docs/2.12/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.load_state_dict)

**是否支持**：是

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.state_dict](https://pytorch.org/docs/2.12/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.state_dict)

**是否支持**：是

</div>

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.optim.ZeroRedundancyOptimizer.step](https://pytorch.org/docs/2.12/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.step)

**是否支持**：是

</div>

</div>
