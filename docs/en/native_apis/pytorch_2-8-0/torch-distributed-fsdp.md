# torch.distributed.fsdp

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:30:40.077Z pushedAt=2026-07-09T08:44:08.273Z -->

> [!NOTE]
> If the API's "Supported" column is "Yes" and "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.distributed.fsdp.FullyShardedDataParallel|Yes|Supports bf16, fp16, fp32<br>When using FSDP on Ascend NPU, it is recommended to pass the device-related parameter "device_id=torch.device("npu:0")"|
|torch.distributed.fsdp.FullyShardedDataParallel.apply|Yes|-|
|torch.distributed.fsdp.FullyShardedDataParallel.check_is_root|Yes|-|
|torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_|Yes|-|
|torch.distributed.fsdp.FullyShardedDataParallel.flatten_sharded_optim_state_dict|No|-|
|torch.distributed.fsdp.FullyShardedDataParallel.forward|Yes|-|
|torch.distributed.fsdp.FullyShardedDataParallel.fsdp_modules|Yes|-|
|torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict|Yes|-|
|torch.distributed.fsdp.FullyShardedDataParallel.get_state_dict_type|No|-|
|torch.distributed.fsdp.FullyShardedDataParallel.module|No|-|
|torch.distributed.fsdp.FullyShardedDataParallel.named_buffers|Yes|-|
|torch.distributed.fsdp.FullyShardedDataParallel.named_parameters|Yes|-|
|torch.distributed.fsdp.FullyShardedDataParallel.no_sync|Yes|-|
|torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict|No|-|
|torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict_to_load|Yes|-|
|torch.distributed.fsdp.FullyShardedDataParallel.register_comm_hook|Yes|-|
|torch.distributed.fsdp.FullyShardedDataParallel.rekey_optim_state_dict|No|-|
|torch.distributed.fsdp.FullyShardedDataParallel.scatter_full_optim_state_dict|No|-|
|torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type|No|-|
|torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict|No|-|
|torch.distributed.fsdp.FullyShardedDataParallel.sharded_optim_state_dict|No|-|
|torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type|Yes|-|
|torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params|No|-|
|torch.distributed.fsdp.BackwardPrefetch|Yes|-|
|torch.distributed.fsdp.ShardingStrategy|Yes|-|
|torch.distributed.fsdp.MixedPrecision|Yes|-|
|torch.distributed.fsdp.CPUOffload|Yes|-|
|torch.distributed.fsdp.StateDictConfig|Yes|-|
|torch.distributed.fsdp.FullStateDictConfig|Yes|-|
|torch.distributed.fsdp.ShardedStateDictConfig|Yes|-|
|torch.distributed.fsdp.LocalStateDictConfig|Yes|-|
|torch.distributed.fsdp.OptimStateDictConfig|Yes|-|
|torch.distributed.fsdp.FullOptimStateDictConfig|Yes|-|
|torch.distributed.fsdp.ShardedOptimStateDictConfig|Yes|-|
|torch.distributed.fsdp.LocalOptimStateDictConfig|Yes|-|
|torch.distributed.fsdp.StateDictSettings|Yes|-|
