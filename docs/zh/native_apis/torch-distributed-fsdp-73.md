# torch.distributed.fsdp

> [!NOTE]  
> 若API“是否支持“为“是“，“限制与说明“为“-“，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.distributed.fsdp.FullyShardedDataParallel|是|在昇腾NPU场景中使用FSDP，推荐传入“device_id=torch.device("npu:0")”设备相关参数|
|torch.distributed.fsdp.FullyShardedDataParallel.apply|是|-|
|torch.distributed.fsdp.FullyShardedDataParallel.check_is_root|是|-|
|torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_|是|-|
|torch.distributed.fsdp.FullyShardedDataParallel.flatten_sharded_optim_state_dict|否|-|
|torch.distributed.fsdp.FullyShardedDataParallel.forward|是|-|
|torch.distributed.fsdp.FullyShardedDataParallel.fsdp_modules|是|-|
|torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict|是|-|
|torch.distributed.fsdp.FullyShardedDataParallel.get_state_dict_type|否|-|
|torch.distributed.fsdp.FullyShardedDataParallel.module|否|-|
|torch.distributed.fsdp.FullyShardedDataParallel.named_buffers|是|-|
|torch.distributed.fsdp.FullyShardedDataParallel.named_parameters|是|-|
|torch.distributed.fsdp.FullyShardedDataParallel.no_sync|是|-|
|torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict|否|-|
|torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict_to_load|是|-|
|torch.distributed.fsdp.FullyShardedDataParallel.register_comm_hook|是|-|
|torch.distributed.fsdp.FullyShardedDataParallel.rekey_optim_state_dict|否|-|
|torch.distributed.fsdp.FullyShardedDataParallel.scatter_full_optim_state_dict|否|-|
|torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type|否|-|
|torch.distributed.fsdp.FullyShardedDataParallel.shard_full_optim_state_dict|否|-|
|torch.distributed.fsdp.FullyShardedDataParallel.sharded_optim_state_dict|否|-|
|torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type|是|-|
|torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params|否|-|
|torch.distributed.fsdp.BackwardPrefetch|是|-|
|torch.distributed.fsdp.ShardingStrategy|是|-|
|torch.distributed.fsdp.MixedPrecision|是|-|
|torch.distributed.fsdp.CPUOffload|是|-|
|torch.distributed.fsdp.StateDictConfig|是|-|
|torch.distributed.fsdp.FullStateDictConfig|是|-|
|torch.distributed.fsdp.ShardedStateDictConfig|是|-|
|torch.distributed.fsdp.LocalStateDictConfig|是|-|
|torch.distributed.fsdp.OptimStateDictConfig|是|-|
|torch.distributed.fsdp.FullOptimStateDictConfig|是|-|
|torch.distributed.fsdp.ShardedOptimStateDictConfig|是|-|
|torch.distributed.fsdp.LocalOptimStateDictConfig|是|-|
|torch.distributed.fsdp.StateDictSettings|是|-|


