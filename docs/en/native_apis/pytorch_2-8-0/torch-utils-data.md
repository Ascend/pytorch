# torch.utils.data

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:34:22.254Z pushedAt=2026-07-09T08:44:08.321Z -->

> [!NOTE]
> If the "Supported" column is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.utils.data.DataLoader|Yes|Supports fp32<br>Multi-process loading of NPU data is not supported; num_workers only supports 0|
|torch.utils.data.Dataset|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64|
|torch.utils.data.IterableDataset|Yes|-|
|torch.utils.data.TensorDataset|Yes|-|
|torch.utils.data.StackDataset|Yes|-|
|torch.utils.data.ConcatDataset|Yes|-|
|torch.utils.data.ChainDataset|Yes|-|
|torch.utils.data.Subset|Yes|Supports int64|
|torch.utils.data._utils.collate.collate|Yes|Supports int64|
|torch.utils.data.default_collate|Yes|Supports fp64, int64, bool|
|torch.utils.data.default_convert|Yes|-|
|torch.utils.data.get_worker_info|Yes|-|
|torch.utils.data.random_split|Yes|-|
|torch.utils.data.Sampler|Yes|Supports int64|
|torch.utils.data.SequentialSampler|Yes|Supports fp32|
|torch.utils.data.RandomSampler|Yes|Supports fp32|
|torch.utils.data.SubsetRandomSampler|Yes|Supports fp32|
|torch.utils.data.WeightedRandomSampler|Yes|Supports fp32|
|torch.utils.data.BatchSampler|Yes|Supports fp32|
|torch.utils.data.distributed.DistributedSampler|Yes|Supports int32|
