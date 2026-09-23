# torch.utils.data

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.9/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.9/data.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Memory Pinning](#memory-pinning)

</div>

<div style="display:none;">

## &#8203;torch.utils.data

</div>

## Memory Pinning

### <code><i>class</i></code> torch.utils.data.TensorDataset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.TensorDataset](https://pytorch.org/docs/2.9/data.html#torch.utils.data.TensorDataset)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id3 -->

</div>

### <code><i>class</i></code> torch.utils.data.StackDataset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.StackDataset](https://pytorch.org/docs/2.9/data.html#torch.utils.data.StackDataset)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id6 -->

</div>

### <code><i>class</i></code> torch.utils.data.ConcatDataset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.ConcatDataset](https://pytorch.org/docs/2.9/data.html#torch.utils.data.ConcatDataset)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id9 -->

</div>

### <code><i>class</i></code> torch.utils.data.ChainDataset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.ChainDataset](https://pytorch.org/docs/2.9/data.html#torch.utils.data.ChainDataset)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id12 -->

</div>

### torch.utils.data._utils.collate.collate

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data._utils.collate.collate](https://pytorch.org/docs/2.9/data.html#torch.utils.data._utils.collate.collate)

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

**限制与说明**：`batch`仅支持int64

</div>

### torch.utils.data.default_convert

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.default_convert](https://pytorch.org/docs/2.9/data.html#torch.utils.data.default_convert)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id18 -->

</div>

### torch.utils.data.random_split

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.random_split](https://pytorch.org/docs/2.9/data.html#torch.utils.data.random_split)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id21 -->

</div>

### <code><i>class</i></code> torch.utils.data.SequentialSampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.SequentialSampler](https://pytorch.org/docs/2.9/data.html#torch.utils.data.SequentialSampler)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id24 -->

**限制与说明**： `data_source`仅支持fp32

</div>

### <code><i>class</i></code> torch.utils.data.RandomSampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.RandomSampler](https://pytorch.org/docs/2.9/data.html#torch.utils.data.RandomSampler)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id27 -->

**限制与说明**： `data_source`仅支持fp32

</div>

### <code><i>class</i></code> torch.utils.data.SubsetRandomSampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.SubsetRandomSampler](https://pytorch.org/docs/2.9/data.html#torch.utils.data.SubsetRandomSampler)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id30 -->

**限制与说明**： `indices`仅支持fp32

</div>

### <code><i>class</i></code> torch.utils.data.WeightedRandomSampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.WeightedRandomSampler](https://pytorch.org/docs/2.9/data.html#torch.utils.data.WeightedRandomSampler)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id33 -->

**限制与说明**： `weights`仅支持fp32

</div>

### <code><i>class</i></code> torch.utils.data.BatchSampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.BatchSampler](https://pytorch.org/docs/2.9/data.html#torch.utils.data.BatchSampler)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id36 -->

</div>

### <code><i>class</i></code> torch.utils.data.distributed.DistributedSampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.distributed.DistributedSampler](https://pytorch.org/docs/2.9/data.html#torch.utils.data.distributed.DistributedSampler)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id39 -->

**限制与说明**： `dataset`仅支持int32

</div>

### <code><i>class</i></code> torch.utils.data.DataLoader

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.DataLoader](https://pytorch.org/docs/2.9/data.html#torch.utils.data.DataLoader)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id42 -->

**限制与说明**：

- `dataset`仅支持fp32
- 不支持多进程加载NPU数据，`num_workers`仅支持0

</div>

### <code><i>class</i></code> torch.utils.data.Dataset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.Dataset](https://pytorch.org/docs/2.9/data.html#torch.utils.data.Dataset)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id45 -->

**限制与说明**： `input`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### <code><i>class</i></code> torch.utils.data.IterableDataset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.IterableDataset](https://pytorch.org/docs/2.9/data.html#torch.utils.data.IterableDataset)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id48 -->

</div>

### <code><i>class</i></code> torch.utils.data.Subset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.Subset](https://pytorch.org/docs/2.9/data.html#torch.utils.data.Subset)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id51 -->

**限制与说明**：`dataset`仅支持int64

</div>

### torch.utils.data.default_collate

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.default_collate](https://pytorch.org/docs/2.9/data.html#torch.utils.data.default_collate)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id54 -->

**限制与说明**： `batch`仅支持fp64，int64，bool

</div>

### torch.utils.data.get_worker_info

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.get_worker_info](https://pytorch.org/docs/2.9/data.html#torch.utils.data.get_worker_info)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id57 -->

</div>

### <code><i>class</i></code> torch.utils.data.Sampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.Sampler](https://pytorch.org/docs/2.9/data.html#torch.utils.data.Sampler)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id60 -->

**限制与说明**：`data_source`仅支持int64

</div>
