# torch.distributed.fsdp

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.14/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.14/fsdp.html)。

<div style="display:none;">

## &#8203;torch.distributed.fsdp

</div>

### <code><i>class</i></code> torch.distributed.fsdp.FullyShardedDataParallel

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT</term>：支持
<!-- end id3 -->

**限制与说明**：

- `input`仅支持bf16，fp16，fp32
- 在昇腾NPU场景中使用FSDP，推荐传入`device_id=torch.device("npu:0")`设备相关参数

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.apply](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.apply)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT</term>：支持
<!-- end id6 -->

</div>

> <font size="3">check_is_root()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.check_is_root](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.check_is_root)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id9 -->

</div>

> <font size="3">clip_grad_norm_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT</term>：支持
<!-- end id12 -->

</div>

> <font size="3">flatten_sharded_optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.flatten_sharded_optim_state_dict](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.flatten_sharded_optim_state_dict)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id15 -->

</div>

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.forward](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.forward)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT</term>：支持
<!-- end id18 -->

</div>

> <font size="3">fsdp_modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.fsdp_modules](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.fsdp_modules)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id21 -->

</div>

> <font size="3">full_optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id24 -->

</div>

> <font size="3">get_state_dict_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.get_state_dict_type](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.get_state_dict_type)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id27 -->

</div>

> <font size="3">module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.module](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.module)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id30 -->

</div>

> <font size="3">named_buffers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.named_buffers](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.named_buffers)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT</term>：支持
<!-- end id33 -->

</div>

> <font size="3">named_parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.named_parameters](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.named_parameters)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT</term>：支持
<!-- end id36 -->

</div>

> <font size="3">no_sync()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.no_sync](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.no_sync)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT</term>：支持
<!-- end id39 -->

</div>

> <font size="3">optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id42 -->

</div>

> <font size="3">optim_state_dict_to_load()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict_to_load](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict_to_load)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id45 -->

</div>

> <font size="3">register_comm_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.register_comm_hook](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.register_comm_hook)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT</term>：支持
<!-- end id48 -->

</div>

> <font size="3">rekey_optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.rekey_optim_state_dict](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.rekey_optim_state_dict)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id51 -->

</div>

> <font size="3">scatter_full_optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.scatter_full_optim_state_dict](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.scatter_full_optim_state_dict)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id54 -->

</div>

> <font size="3">set_state_dict_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id57 -->

</div>

> <font size="3">shard_full_optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id60 -->

</div>

> <font size="3">sharded_optim_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.sharded_optim_state_dict](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.sharded_optim_state_dict)

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id63 -->

</div>

> <font size="3">state_dict_type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type)

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id66 -->

</div>

> <font size="3">summon_full_params()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params)

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id69 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.fsdp.BackwardPrefetch

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.BackwardPrefetch](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.BackwardPrefetch)

**产品支持情况**：

<!-- npu="910b" id70 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT</term>：支持
<!-- end id72 -->

</div>

### <code><i>class</i></code> torch.distributed.fsdp.ShardingStrategy

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.ShardingStrategy](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.ShardingStrategy)

**产品支持情况**：

<!-- npu="910b" id73 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT</term>：支持
<!-- end id75 -->

</div>

### <code><i>class</i></code> torch.distributed.fsdp.MixedPrecision

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.MixedPrecision](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.MixedPrecision)

**产品支持情况**：

<!-- npu="910b" id76 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT</term>：支持
<!-- end id78 -->

</div>

### <code><i>class</i></code> torch.distributed.fsdp.CPUOffload

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.CPUOffload](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.CPUOffload)

**产品支持情况**：

<!-- npu="910b" id79 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT</term>：支持
<!-- end id81 -->

</div>

### <code><i>class</i></code> torch.distributed.fsdp.StateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.StateDictConfig](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.StateDictConfig)

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id84 -->

</div>

### <code><i>class</i></code> torch.distributed.fsdp.FullStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullStateDictConfig](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullStateDictConfig)

**产品支持情况**：

<!-- npu="910b" id85 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id87 -->

</div>

### <code><i>class</i></code> torch.distributed.fsdp.ShardedStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.ShardedStateDictConfig](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.ShardedStateDictConfig)

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id90 -->

</div>

### <code><i>class</i></code> torch.distributed.fsdp.LocalStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.LocalStateDictConfig](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.LocalStateDictConfig)

**产品支持情况**：

<!-- npu="910b" id91 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id93 -->

</div>

### <code><i>class</i></code> torch.distributed.fsdp.OptimStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.OptimStateDictConfig](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.OptimStateDictConfig)

**产品支持情况**：

<!-- npu="910b" id94 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id96 -->

</div>

### <code><i>class</i></code> torch.distributed.fsdp.FullOptimStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.FullOptimStateDictConfig](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.FullOptimStateDictConfig)

**产品支持情况**：

<!-- npu="910b" id97 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id99 -->

</div>

### <code><i>class</i></code> torch.distributed.fsdp.ShardedOptimStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.ShardedOptimStateDictConfig](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.ShardedOptimStateDictConfig)

**产品支持情况**：

<!-- npu="910b" id100 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id102 -->

</div>

### <code><i>class</i></code> torch.distributed.fsdp.LocalOptimStateDictConfig

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.LocalOptimStateDictConfig](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.LocalOptimStateDictConfig)

**产品支持情况**：

<!-- npu="910b" id103 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id105 -->

</div>

### <code><i>class</i></code> torch.distributed.fsdp.StateDictSettings

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.fsdp.StateDictSettings](https://pytorch.org/docs/2.14/fsdp.html#torch.distributed.fsdp.StateDictSettings)

**产品支持情况**：

<!-- npu="910b" id106 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id108 -->

</div>
