# torch.distributed.fsdp

> [!NOTE]  
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)

## base API

### _`class`_ torch.distributed.fsdp.FullyShardedDataParallel

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32
- 在昇腾NPU场景中使用FSDP，推荐传入“device_id=torch.device("npu:0")”设备相关参数

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.apply](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.apply)

**是否支持**：是

</div>

> <font size="3">check_is_root()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.check_is_root](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.check_is_root)

**是否支持**：是

</div>

> <font size="3">clip_grad_norm_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_)

**是否支持**：是

</div>

> <font size="3">flatten_sharded_optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.flatten_sharded_optim_state_dict](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.flatten_sharded_optim_state_dict)

**是否支持**：否

</div>

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.forward](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.forward)

**是否支持**：是

</div>

> <font size="3">fsdp_modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.fsdp_modules](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.fsdp_modules)

**是否支持**：是

</div>

> <font size="3">full_optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict)

**是否支持**：是

</div>

> <font size="3">get_state_dict_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.get_state_dict_type](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.get_state_dict_type)

**是否支持**：否

</div>

> <font size="3">module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.module](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.module)

**是否支持**：否

</div>

> <font size="3">named_buffers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.named_buffers](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.named_buffers)

**是否支持**：是

</div>

> <font size="3">named_parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.named_parameters](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.named_parameters)

**是否支持**：是

</div>

> <font size="3">no_sync()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.no_sync](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.no_sync)

**是否支持**：是

</div>

> <font size="3">optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict)

**是否支持**：否

</div>

> <font size="3">optim_state_dict_to_load()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict_to_load](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict_to_load)

**是否支持**：是

</div>

> <font size="3">register_comm_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.register_comm_hook](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.register_comm_hook)

**是否支持**：是

</div>

> <font size="3">rekey_optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.rekey_optim_state_dict](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.rekey_optim_state_dict)

**是否支持**：否

</div>

> <font size="3">scatter_full_optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.scatter_full_optim_state_dict](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.scatter_full_optim_state_dict)

**是否支持**：否

</div>

> <font size="3">set_state_dict_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type)

**是否支持**：否

</div>

> <font size="3">shard_full_optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict)

**是否支持**：否

</div>

> <font size="3">sharded_optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.sharded_optim_state_dict](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.sharded_optim_state_dict)

**是否支持**：否

</div>

> <font size="3">state_dict_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type)

**是否支持**：是

</div>

> <font size="3">summon_full_params()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params)

**是否支持**：否

</div>

</div>

### _`class`_ torch.distributed.fsdp.BackwardPrefetch

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.BackwardPrefetch](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.BackwardPrefetch)

**是否支持**：是

</div>

### _`class`_ torch.distributed.fsdp.ShardingStrategy

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.ShardingStrategy](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.ShardingStrategy)

**是否支持**：是

</div>

### _`class`_ torch.distributed.fsdp.MixedPrecision

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.MixedPrecision](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.MixedPrecision)

**是否支持**：是

</div>

### _`class`_ torch.distributed.fsdp.CPUOffload

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.CPUOffload](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.CPUOffload)

**是否支持**：是

</div>

### _`class`_ torch.distributed.fsdp.StateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.StateDictConfig](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.StateDictConfig)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.distributed.fsdp.FullStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullStateDictConfig](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullStateDictConfig)

**是否支持**：是

</div>

### _`class`_ torch.distributed.fsdp.ShardedStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.ShardedStateDictConfig](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.ShardedStateDictConfig)

**是否支持**：是

</div>

### _`class`_ torch.distributed.fsdp.LocalStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.LocalStateDictConfig](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.LocalStateDictConfig)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.distributed.fsdp.OptimStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.OptimStateDictConfig](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.OptimStateDictConfig)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.distributed.fsdp.FullOptimStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullOptimStateDictConfig](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullOptimStateDictConfig)

**是否支持**：是

</div>

### _`class`_ torch.distributed.fsdp.ShardedOptimStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.ShardedOptimStateDictConfig](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.ShardedOptimStateDictConfig)

**是否支持**：是

</div>

### _`class`_ torch.distributed.fsdp.LocalOptimStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.LocalOptimStateDictConfig](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.LocalOptimStateDictConfig)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.distributed.fsdp.StateDictSettings

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.StateDictSettings](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.StateDictSettings)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>
