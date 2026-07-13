# torch.distributed.tensor.parallel

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T02:15:36.491Z pushedAt=2026-06-15T03:25:49.164Z -->

> [!NOTE]  
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.distributed.tensor.parallel.parallelize_module|No|-|
|torch.distributed.tensor.parallel.ColwiseParallel|Yes|Supports bf16, fp16, fp32|
|torch.distributed.tensor.parallel.RowwiseParallel|No|-|
|torch.distributed.tensor.parallel.PrepareModuleInput|No|-|
|torch.distributed.tensor.parallel.PrepareModuleOutput|Yes|-|
|torch.distributed.tensor.parallel.loss_parallel|Yes|Supports bf16, fp16, fp32, int64|
