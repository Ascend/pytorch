# torch.distributed.tensor.parallel

> [!NOTE]  
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.distributed.tensor.parallel.parallelize_module|否|-|
|torch.distributed.tensor.parallel.ColwiseParallel|是|支持bf16，fp16，fp32|
|torch.distributed.tensor.parallel.RowwiseParallel|否|-|
|torch.distributed.tensor.parallel.PrepareModuleInput|否|-|
|torch.distributed.tensor.parallel.PrepareModuleOutput|是|-|
|torch.distributed.tensor.parallel.loss_parallel|是|支持bf16，fp16，fp32，int64|
