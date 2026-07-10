# torch.distributed.optim

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:30:42.098Z pushedAt=2026-07-09T08:44:08.274Z -->

> [!NOTE]
> If the API's "Supported" column is "Yes" and "Restrictions and Notes" is "-", it means the API support is consistent with the native API.

|API Name|Support|Restrictions and Notes|
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
