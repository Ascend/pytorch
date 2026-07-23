# torch.utils.data

> [!NOTE]  
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)
- [Dataset Types](#dataset-types)
- [Memory Pinning](#memory-pinning)
- [Loading Batched and Non-Batched Data](#loading-batched-and-non-batched-data)
- [Single- and Multi-process Data Loading](#single--and-multi-process-data-loading)
- [Data Loading Order andSampler](#data-loading-order-andsampler)

## base API

### _`class`_ torch.utils.data.IterDataPipe

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.IterDataPipe](https://pytorch.org/docs/2.7/data.html#torch.utils.data.IterDataPipe)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.utils.data.MapDataPipe

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.MapDataPipe](https://pytorch.org/docs/2.7/data.html#torch.utils.data.MapDataPipe)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.utils.data.TensorDataset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.TensorDataset](https://pytorch.org/docs/2.7/data.html#torch.utils.data.TensorDataset)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.utils.data.StackDataset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.StackDataset](https://pytorch.org/docs/2.7/data.html#torch.utils.data.StackDataset)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.utils.data.ConcatDataset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.ConcatDataset](https://pytorch.org/docs/2.7/data.html#torch.utils.data.ConcatDataset)

**是否支持**：是

</div>

### _`class`_ torch.utils.data.ChainDataset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.ChainDataset](https://pytorch.org/docs/2.7/data.html#torch.utils.data.ChainDataset)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.utils.data._utils.collate.collate

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data._utils.collate.collate](https://pytorch.org/docs/2.7/data.html#torch.utils.data._utils.collate.collate)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持int64

</div>

### torch.utils.data.default_convert

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.default_convert](https://pytorch.org/docs/2.7/data.html#torch.utils.data.default_convert)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.utils.data.graph_settings.apply_sharding

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.graph_settings.apply_sharding](https://pytorch.org/docs/2.7/data.html#torch.utils.data.graph_settings.apply_sharding)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.utils.data.graph_settings.get_all_graph_pipes

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.graph_settings.get_all_graph_pipes](https://pytorch.org/docs/2.7/data.html#torch.utils.data.graph_settings.get_all_graph_pipes)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.utils.data.random_split

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.random_split](https://pytorch.org/docs/2.7/data.html#torch.utils.data.random_split)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.utils.data.SequentialSampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.SequentialSampler](https://pytorch.org/docs/2.7/data.html#torch.utils.data.SequentialSampler)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

### _`class`_ torch.utils.data.RandomSampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.RandomSampler](https://pytorch.org/docs/2.7/data.html#torch.utils.data.RandomSampler)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

### _`class`_ torch.utils.data.SubsetRandomSampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.SubsetRandomSampler](https://pytorch.org/docs/2.7/data.html#torch.utils.data.SubsetRandomSampler)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### _`class`_ torch.utils.data.WeightedRandomSampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.WeightedRandomSampler](https://pytorch.org/docs/2.7/data.html#torch.utils.data.WeightedRandomSampler)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### _`class`_ torch.utils.data.BatchSampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.BatchSampler](https://pytorch.org/docs/2.7/data.html#torch.utils.data.BatchSampler)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.utils.data.distributed.DistributedSampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.distributed.DistributedSampler](https://pytorch.org/docs/2.7/data.html#torch.utils.data.distributed.DistributedSampler)

**是否支持**：是

**限制与说明**： 支持int32

</div>

## Dataset Types

### _`class`_ torch.utils.data.DataLoader

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.DataLoader](https://pytorch.org/docs/2.7/data.html#torch.utils.data.DataLoader)

**是否支持**：是

**限制与说明**：

- 支持fp32
- 不支持多进程加载NPU数据，num_workers仅支持0

</div>

### _`class`_ torch.utils.data.Dataset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.Dataset](https://pytorch.org/docs/2.7/data.html#torch.utils.data.Dataset)

**是否支持**：是

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### _`class`_ torch.utils.data.IterableDataset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.IterableDataset](https://pytorch.org/docs/2.7/data.html#torch.utils.data.IterableDataset)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## Memory Pinning

### _`class`_ torch.utils.data.Subset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.Subset](https://pytorch.org/docs/2.7/data.html#torch.utils.data.Subset)

**是否支持**：是

**限制与说明**： 支持int64

</div>

## Loading Batched and Non-Batched Data

### torch.utils.data.default_collate

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.default_collate](https://pytorch.org/docs/2.7/data.html#torch.utils.data.default_collate)

**是否支持**：是

**限制与说明**： 支持fp64，int64，bool

</div>

## Single- and Multi-process Data Loading

### torch.utils.data.get_worker_info

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.get_worker_info](https://pytorch.org/docs/2.7/data.html#torch.utils.data.get_worker_info)

**是否支持**：是

</div>

## Data Loading Order andSampler

### _`class`_ torch.utils.data.Sampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.Sampler](https://pytorch.org/docs/2.7/data.html#torch.utils.data.Sampler)

**是否支持**：是

**限制与说明**： 支持int64

</div>
