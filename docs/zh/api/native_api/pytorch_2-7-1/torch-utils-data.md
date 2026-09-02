# torch.utils.data

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.7/data.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Memory Pinning](#memory-pinning)

</div>

<div style="display:none;">

## &#8203;torch.utils.data

</div>

### <code><i>class</i></code> torch.utils.data.IterDataPipe

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.IterDataPipe](https://pytorch.org/docs/2.7/data.html#torch.utils.data.IterDataPipe)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.utils.data.MapDataPipe

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.MapDataPipe](https://pytorch.org/docs/2.7/data.html#torch.utils.data.MapDataPipe)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.utils.data.graph_settings.apply_sharding

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.graph_settings.apply_sharding](https://pytorch.org/docs/2.7/data.html#torch.utils.data.graph_settings.apply_sharding)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.utils.data.graph_settings.get_all_graph_pipes

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.graph_settings.get_all_graph_pipes](https://pytorch.org/docs/2.7/data.html#torch.utils.data.graph_settings.get_all_graph_pipes)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

## Memory Pinning

### <code><i>class</i></code> torch.utils.data.TensorDataset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.TensorDataset](https://pytorch.org/docs/2.7/data.html#torch.utils.data.TensorDataset)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.utils.data.StackDataset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.StackDataset](https://pytorch.org/docs/2.7/data.html#torch.utils.data.StackDataset)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.utils.data.ConcatDataset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.ConcatDataset](https://pytorch.org/docs/2.7/data.html#torch.utils.data.ConcatDataset)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.utils.data.ChainDataset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.ChainDataset](https://pytorch.org/docs/2.7/data.html#torch.utils.data.ChainDataset)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.utils.data._utils.collate.collate

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data._utils.collate.collate](https://pytorch.org/docs/2.7/data.html#torch.utils.data._utils.collate.collate)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`batch`仅支持int64

</div>

### torch.utils.data.default_convert

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.default_convert](https://pytorch.org/docs/2.7/data.html#torch.utils.data.default_convert)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.utils.data.random_split

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.random_split](https://pytorch.org/docs/2.7/data.html#torch.utils.data.random_split)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.utils.data.SequentialSampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.SequentialSampler](https://pytorch.org/docs/2.7/data.html#torch.utils.data.SequentialSampler)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `data_source`仅支持fp32

</div>

### <code><i>class</i></code> torch.utils.data.RandomSampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.RandomSampler](https://pytorch.org/docs/2.7/data.html#torch.utils.data.RandomSampler)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `data_source`仅支持fp32

</div>

### <code><i>class</i></code> torch.utils.data.SubsetRandomSampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.SubsetRandomSampler](https://pytorch.org/docs/2.7/data.html#torch.utils.data.SubsetRandomSampler)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `indices`仅支持fp32

</div>

### <code><i>class</i></code> torch.utils.data.WeightedRandomSampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.WeightedRandomSampler](https://pytorch.org/docs/2.7/data.html#torch.utils.data.WeightedRandomSampler)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `weights`仅支持fp32

</div>

### <code><i>class</i></code> torch.utils.data.BatchSampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.BatchSampler](https://pytorch.org/docs/2.7/data.html#torch.utils.data.BatchSampler)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.utils.data.distributed.DistributedSampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.distributed.DistributedSampler](https://pytorch.org/docs/2.7/data.html#torch.utils.data.distributed.DistributedSampler)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `dataset`仅支持int32

</div>

### <code><i>class</i></code> torch.utils.data.DataLoader

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.DataLoader](https://pytorch.org/docs/2.7/data.html#torch.utils.data.DataLoader)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `dataset`仅支持fp32
- 不支持多进程加载NPU数据，`num_workers`仅支持0

</div>

### <code><i>class</i></code> torch.utils.data.Dataset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.Dataset](https://pytorch.org/docs/2.7/data.html#torch.utils.data.Dataset)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `input`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64

</div>

### <code><i>class</i></code> torch.utils.data.IterableDataset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.IterableDataset](https://pytorch.org/docs/2.7/data.html#torch.utils.data.IterableDataset)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.utils.data.Subset

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.Subset](https://pytorch.org/docs/2.7/data.html#torch.utils.data.Subset)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`dataset`仅支持int64

</div>

### torch.utils.data.default_collate

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.default_collate](https://pytorch.org/docs/2.7/data.html#torch.utils.data.default_collate)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： `batch`仅支持fp64，int64，bool

</div>

### torch.utils.data.get_worker_info

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.get_worker_info](https://pytorch.org/docs/2.7/data.html#torch.utils.data.get_worker_info)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.utils.data.Sampler

<div style="margin-left: 2em">

**原生文档**：[torch.utils.data.Sampler](https://pytorch.org/docs/2.7/data.html#torch.utils.data.Sampler)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`data_source`仅支持int64

</div>
