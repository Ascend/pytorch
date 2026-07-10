# torch.distributed.tensor.parallel

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:30:51.977Z pushedAt=2026-07-09T08:44:08.276Z -->

> [!NOTE]
> If an API's "Supported" column is "Yes" and its "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.distributed.tensor.parallel.parallelize_module|No|-|
|torch.distributed.tensor.parallel.ColwiseParallel|Yes|Supports bf16, fp16, fp32|
|torch.distributed.tensor.parallel.RowwiseParallel|No|-|
|torch.distributed.tensor.parallel.PrepareModuleInput|No|-|
|torch.distributed.tensor.parallel.PrepareModuleOutput|Yes|-|
|torch.distributed.tensor.parallel.loss_parallel|Yes|Supports bf16, fp16, fp32, int64|
