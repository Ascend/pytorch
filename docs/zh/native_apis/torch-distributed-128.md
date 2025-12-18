# torch.distributed

> [!NOTE]  
> 若API“是否支持“为“是“，“限制与说明“为“-“，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.distributed.is_available|是|-|
|torch.distributed.init_process_group|是|当pg_options函数传入类型为torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()时，配置该变量属性hccl_config可控制HCCL通信域缓存区大小。具体示例可参考《PyTorch 训练模型迁移调优指南》的“hccl_buffer_size”章节，配置变量属性hccl_config的group_name字段可以设置HCCL通信域的通信组自定义名称，取值为长度不超过32的字符串。|
|torch.distributed.is_initialized|是|-|
|torch.distributed.is_mpi_available|是|-|
|torch.distributed.is_nccl_available|是|-|
|torch.distributed.is_gloo_available|是|-|
|torch.distributed.is_torchelastic_launched|是|-|
|torch.distributed.Backend|是|-|
|torch.distributed.Backend.register_backend|是|-|
|torch.distributed.get_backend|是|-|
|torch.distributed.get_rank|是|-|
|torch.distributed.get_world_size|是|-|
|torch.distributed.Store|是|-|
|torch.distributed.TCPStore|是|-|
|torch.distributed.HashStore|是|-|
|torch.distributed.FileStore|是|-|
|torch.distributed.PrefixStore|是|-|
|torch.distributed.Store.set|是|-|
|torch.distributed.Store.get|是|-|
|torch.distributed.Store.add|是|-|
|torch.distributed.Store.compare_set|是|-|
|torch.distributed.Store.wait|是|-|
|torch.distributed.Store.num_keys|是|-|
|torch.distributed.Store.delete_key|是|-|
|torch.distributed.Store.set_timeout|是|-|
|torch.distributed.new_group|是|当pg_options函数传入类型为torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()时，配置该变量属性hccl_config可控制HCCL通信域缓存区大小。具体示例可参考《PyTorch 训练模型迁移调优指南》的“hccl_buffer_size”章节，配置变量属性hccl_config的group_name字段可以设置HCCL通信域的通信组自定义名称，取值为长度不超过32的字符串。|
|torch.distributed.get_group_rank|是|-|
|torch.distributed.get_global_rank|是|-|
|torch.distributed.get_process_group_ranks|是|-|
|torch.distributed.device_mesh.DeviceMesh|是|-|
|torch.distributed.send|是|支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool|
|torch.distributed.recv|是|支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool|
|torch.distributed.isend|是|支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool|
|torch.distributed.irecv|是|支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool|
|torch.distributed.batch_isend_irecv|是|支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool|
|torch.distributed.P2POp|是|支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool|
|torch.distributed.broadcast|是|支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool|
|torch.distributed.broadcast_object_list|是|-|
|torch.distributed.all_reduce|是|支持bf16，fp16， fp32， int32， int64， bool|
|torch.distributed.reduce|是|支持bf16，fp16，fp32，uint8，int8，int32，int64，bool|
|torch.distributed.all_gather|是|支持bf16，fp16，fp32，int8，int32，bool|
|torch.distributed.all_gather_into_tensor|是|支持bf16，fp16，fp32，int8，int32，boolworld size<br>不支持3，5，6，7|
|torch.distributed.all_gather_object|是|-|
|torch.distributed.gather|是|支持bf16，fp16，fp32，int8，int32，bool|
|torch.distributed.gather_object|是|支持的输入类型为Python Object对象|
|torch.distributed.scatter|是|支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool|
|torch.distributed.scatter_object_list|是|不涉及dtype参数|
|torch.distributed.reduce_scatter|是|支持bf16，fp16，fp32，int8，int32|
|torch.distributed.reduce_scatter_tensor|是|支持bf16，fp16，fp32，int8，int32world size<br>不支持3，5，6，7|
|torch.distributed.all_to_all_single|是|支持fp32|
|torch.distributed.all_to_all|是|支持fp32|
|torch.distributed.barrier|是|-|
|torch.distributed.monitored_barrier|是|-|
|torch.distributed.ReduceOp|是|支持bf16，fp16，fp32，uint8，int8，int32，int64，bool|
|torch.distributed.reduce_op|是|支持bf16，fp16，fp32，uint8，int8，int32，int64|
|torch.distributed.DistBackendError|是|-|
|torch.distributed.device_mesh.DeviceMesh.from_group|是|-|
|torch.distributed.device_mesh.DeviceMesh.get_all_groups|是|-|
|torch.distributed.device_mesh.DeviceMesh.get_coordinate|是|-|
|torch.distributed.device_mesh.DeviceMesh.get_group|是|-|
|torch.distributed.device_mesh.DeviceMesh.get_local_rank|是|-|
|torch.distributed.device_mesh.DeviceMesh.get_rank|是|-|


