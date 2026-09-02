# torch.distributed.fsdp

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.11/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.11/fsdp.html)。

<div style="display:none;">

## &#8203;torch.distributed.fsdp

</div>

### <code><i>class</i></code> torch.distributed.fsdp.FullyShardedDataParallel

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- 在昇腾NPU场景中使用FSDP，推荐传入`device_id=torch.device("npu:0")`设备相关参数

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.apply](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.apply)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">check_is_root()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.check_is_root](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.check_is_root)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">clip_grad_norm_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">flatten_sharded_optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.flatten_sharded_optim_state_dict](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.flatten_sharded_optim_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.forward](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.forward)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">fsdp_modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.fsdp_modules](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.fsdp_modules)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">full_optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">get_state_dict_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.get_state_dict_type](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.get_state_dict_type)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.module](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.module)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">named_buffers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.named_buffers](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.named_buffers)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">named_parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.named_parameters](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.named_parameters)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">no_sync()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.no_sync](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.no_sync)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">optim_state_dict_to_load()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict_to_load](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict_to_load)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">register_comm_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.register_comm_hook](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.register_comm_hook)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">rekey_optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.rekey_optim_state_dict](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.rekey_optim_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">scatter_full_optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.scatter_full_optim_state_dict](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.scatter_full_optim_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">set_state_dict_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">shard_full_optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">sharded_optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.sharded_optim_state_dict](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.sharded_optim_state_dict)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">state_dict_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">summon_full_params()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

### <code><i>class</i></code> torch.distributed.fsdp.BackwardPrefetch

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.BackwardPrefetch](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.BackwardPrefetch)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.distributed.fsdp.ShardingStrategy

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.ShardingStrategy](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.ShardingStrategy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.distributed.fsdp.MixedPrecision

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.MixedPrecision](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.MixedPrecision)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.distributed.fsdp.CPUOffload

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.CPUOffload](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.CPUOffload)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.distributed.fsdp.StateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.StateDictConfig](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.StateDictConfig)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.distributed.fsdp.FullStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullStateDictConfig](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullStateDictConfig)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.distributed.fsdp.ShardedStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.ShardedStateDictConfig](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.ShardedStateDictConfig)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.distributed.fsdp.LocalStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.LocalStateDictConfig](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.LocalStateDictConfig)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.distributed.fsdp.OptimStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.OptimStateDictConfig](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.OptimStateDictConfig)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.distributed.fsdp.FullOptimStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullOptimStateDictConfig](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullOptimStateDictConfig)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.distributed.fsdp.ShardedOptimStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.ShardedOptimStateDictConfig](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.ShardedOptimStateDictConfig)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.distributed.fsdp.LocalOptimStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.LocalOptimStateDictConfig](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.LocalOptimStateDictConfig)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.distributed.fsdp.StateDictSettings

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.StateDictSettings](https://pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.StateDictSettings)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>
