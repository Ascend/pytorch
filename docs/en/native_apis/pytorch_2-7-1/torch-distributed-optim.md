# torch.distributed.optim

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T01:05:44.480Z pushedAt=2026-06-15T02:04:36.500Z -->

> [!NOTE]  
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.distributed.optim.DistributedOptimizer|No|-|
|torch.distributed.optim.DistributedOptimizer.step|No|-|
|torch.distributed.optim.PostLocalSGDOptimizer|Yes|-|
|torch.distributed.optim.PostLocalSGDOptimizer.load_state_dict|Yes|-|
|torch.distributed.optim.PostLocalSGDOptimizer.state_dict|Yes|-|
|torch.distributed.optim.PostLocalSGDOptimizer.step|Yes|-|
|torch.distributed.optim.ZeroRedundancyOptimizer|Yes|Supported input type is torch.nn.Optimizer object<br>NPU fused optimizer object is not supported|
|torch.distributed.optim.ZeroRedundancyOptimizer.add_param_group|Yes|-|
|torch.distributed.optim.ZeroRedundancyOptimizer.consolidate_state_dict|Yes|-|
|torch.distributed.optim.ZeroRedundancyOptimizer.join_device|Yes|-|
|torch.distributed.optim.ZeroRedundancyOptimizer.join_hook|Yes|-|
|torch.distributed.optim.ZeroRedundancyOptimizer.join_process_group|Yes|-|
|torch.distributed.optim.ZeroRedundancyOptimizer.load_state_dict|Yes|-|
|torch.distributed.optim.ZeroRedundancyOptimizer.state_dict|Yes|-|
|torch.distributed.optim.ZeroRedundancyOptimizer.step|Yes|-|
