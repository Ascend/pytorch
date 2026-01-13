# torch.distributed.optim

> [!NOTE]  
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.distributed.optim.DistributedOptimizer|否|-|
|torch.distributed.optim.DistributedOptimizer.step|否|-|
|torch.distributed.optim.PostLocalSGDOptimizer|是|-|
|torch.distributed.optim.PostLocalSGDOptimizer.load_state_dict|是|-|
|torch.distributed.optim.PostLocalSGDOptimizer.state_dict|是|-|
|torch.distributed.optim.PostLocalSGDOptimizer.step|是|-|
|torch.distributed.optim.ZeroRedundancyOptimizer|是|支持的输入类型为torch.nn.Optimizer对象<br>不支持NPU融合优化器对象|
|torch.distributed.optim.ZeroRedundancyOptimizer.add_param_group|是|-|
|torch.distributed.optim.ZeroRedundancyOptimizer.consolidate_state_dict|是|-|
|torch.distributed.optim.ZeroRedundancyOptimizer.join_device|是|-|
|torch.distributed.optim.ZeroRedundancyOptimizer.join_hook|是|-|
|torch.distributed.optim.ZeroRedundancyOptimizer.join_process_group|是|-|
|torch.distributed.optim.ZeroRedundancyOptimizer.load_state_dict|是|-|
|torch.distributed.optim.ZeroRedundancyOptimizer.state_dict|是|-|
|torch.distributed.optim.ZeroRedundancyOptimizer.step|是|-|


