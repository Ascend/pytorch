# torch.utils.data

> [!NOTE]  
> 若API“是否支持“为“是“，“限制与说明“为“-“，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.utils.data.DataLoader|是|支持fp32<br>不支持多进程加载NPU数据，num_workers仅支持0|
|torch.utils.data.Dataset|是|支持fp16，fp32，fp64，uint8，int8，int16，int32，int64|
|torch.utils.data.IterableDataset|是|-|
|torch.utils.data.TensorDataset|是|-|
|torch.utils.data.StackDataset|是|-|
|torch.utils.data.ConcatDataset|是|-|
|torch.utils.data.ChainDataset|是|-|
|torch.utils.data.Subset|是|支持int64|
|torch.utils.data._utils.collate.collate|是|支持int64|
|torch.utils.data.default_collate|是|支持fp64，int64，bool|
|torch.utils.data.default_convert|是|-|
|torch.utils.data.get_worker_info|是|-|
|torch.utils.data.random_split|是|-|
|torch.utils.data.Sampler|是|支持int64|
|torch.utils.data.SequentialSampler|是|支持fp32|
|torch.utils.data.RandomSampler|是|支持fp32|
|torch.utils.data.SubsetRandomSampler|是|支持fp32|
|torch.utils.data.WeightedRandomSampler|是|支持fp32|
|torch.utils.data.BatchSampler|是|支持fp32|
|torch.utils.data.distributed.DistributedSampler|是|支持int32|


