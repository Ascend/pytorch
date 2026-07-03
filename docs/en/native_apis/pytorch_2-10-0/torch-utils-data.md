# torch.utils.data

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-14T08:00:39.008Z pushedAt=2026-06-14T09:16:34.795Z -->

> [!NOTE]
>
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.utils.data.DataLoader](https://pytorch.org/docs/2.10/data.html#torch.utils.data.DataLoader)|Yes|fp32 supported<br>Multi-process NPU data loading is not supported, num_workers only supports 0|
|[torch.utils.data.Dataset](https://pytorch.org/docs/2.10/data.html#torch.utils.data.Dataset)|Yes|fp16, fp32, fp64, uint8, int8, int16, int32, int64 supported|
|[torch.utils.data.IterableDataset](https://pytorch.org/docs/2.10/data.html#torch.utils.data.IterableDataset)|Yes|-|
|[torch.utils.data.TensorDataset](https://pytorch.org/docs/2.10/data.html#torch.utils.data.TensorDataset)|Yes|-|
|[torch.utils.data.StackDataset](https://pytorch.org/docs/2.10/data.html#torch.utils.data.StackDataset)|Yes|-|
|[torch.utils.data.ConcatDataset](https://pytorch.org/docs/2.10/data.html#torch.utils.data.ConcatDataset)|Yes|-|
|[torch.utils.data.ChainDataset](https://pytorch.org/docs/2.10/data.html#torch.utils.data.ChainDataset)|Yes|-|
|[torch.utils.data.Subset](https://pytorch.org/docs/2.10/data.html#torch.utils.data.Subset)|Yes|int64 supported|
|[torch.utils.data._utils.collate.collate](https://pytorch.org/docs/2.10/data.html#torch.utils.data._utils.collate.collate)|Yes|int64 supported|
|[torch.utils.data.default_collate](https://pytorch.org/docs/2.10/data.html#torch.utils.data.default_collate)|Yes|fp64, int64, bool supported|
|[torch.utils.data.default_convert](https://pytorch.org/docs/2.10/data.html#torch.utils.data.default_convert)|Yes|-|
|[torch.utils.data.get_worker_info](https://pytorch.org/docs/2.10/data.html#torch.utils.data.get_worker_info)|Yes|-|
|[torch.utils.data.random_split](https://pytorch.org/docs/2.10/data.html#torch.utils.data.random_split)|Yes|-|
|[torch.utils.data.Sampler](https://pytorch.org/docs/2.10/data.html#torch.utils.data.Sampler)|Yes|int64 supported|
|[torch.utils.data.SequentialSampler](https://pytorch.org/docs/2.10/data.html#torch.utils.data.SequentialSampler)|Yes|fp32 supported|
|[torch.utils.data.RandomSampler](https://pytorch.org/docs/2.10/data.html#torch.utils.data.RandomSampler)|Yes|fp32 supported|
|[torch.utils.data.SubsetRandomSampler](https://pytorch.org/docs/2.10/data.html#torch.utils.data.SubsetRandomSampler)|Yes|fp32 supported|
|[torch.utils.data.WeightedRandomSampler](https://pytorch.org/docs/2.10/data.html#torch.utils.data.WeightedRandomSampler)|Yes|fp32 supported|
|[torch.utils.data.BatchSampler](https://pytorch.org/docs/2.10/data.html#torch.utils.data.BatchSampler)|Yes|fp32 supported|
|[torch.utils.data.distributed.DistributedSampler](https://pytorch.org/docs/2.10/data.html#torch.utils.data.distributed.DistributedSampler)|Yes|int32 supported|
