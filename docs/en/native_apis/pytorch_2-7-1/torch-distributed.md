# torch.distributed

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T01:06:35.557Z pushedAt=2026-06-15T02:04:36.518Z -->

> [!NOTE]
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.distributed.is_available](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.is_available)|Yes|-|
|[torch.distributed.init_process_group](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.init_process_group)|Yes|When the `pg_options` function passes in an object of type `torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()`, configuring the `hccl_config` attribute of this variable can control the HCCL communication domain buffer size. For a specific example, refer to the "[hccl_buffer_size](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/hccl_buffer_size.md)" section of the *PyTorch Model Migration and Tuning Guide*. Configuring the `group_name` field of the `hccl_config` variable attribute can set a custom name for the communication group of the HCCL communication domain, with a value being a string of no more than 32 characters.|
|[torch.distributed.is_initialized](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.is_initialized)|Yes|-|
|[torch.distributed.is_mpi_available](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.is_mpi_available)|Yes|-|
|[torch.distributed.is_nccl_available](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.is_nccl_available)|Yes|-|
|[torch.distributed.is_gloo_available](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.is_gloo_available)|Yes|-|
|[torch.distributed.is_torchelastic_launched](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.is_torchelastic_launched)|Yes|-|
|[torch.distributed.Backend](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Backend)|Yes|-|
|[torch.distributed.Backend.register_backend](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Backend.register_backend)|Yes|-|
|[torch.distributed.get_backend](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.get_backend)|Yes|-|
|[torch.distributed.get_rank](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.get_rank)|Yes|-|
|[torch.distributed.get_world_size](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.get_world_size)|Yes|-|
|[torch.distributed.Store](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store)|Yes|-|
|[torch.distributed.TCPStore](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.TCPStore)|Yes|-|
|[torch.distributed.HashStore](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.HashStore)|Yes|-|
|[torch.distributed.FileStore](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.FileStore)|Yes|-|
|[torch.distributed.PrefixStore](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.PrefixStore)|Yes|-|
|[torch.distributed.Store.\_\_init\_\_](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.__init__)|Yes|-|
|[torch.distributed.Store.set](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.set)|Yes|-|
|[torch.distributed.Store.get](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.get)|Yes|-|
|[torch.distributed.Store.add](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.add)|Yes|-|
|[torch.distributed.Store.compare_set](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.compare_set)|Yes|-|
|[torch.distributed.Store.wait](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.wait)|Yes|-|
|[torch.distributed.Store.num_keys](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.num_keys)|Yes|-|
|[torch.distributed.Store.delete_key](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.delete_key)|Yes|-|
|[torch.distributed.Store.set_timeout](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.set_timeout)|Yes|-|
|[torch.distributed.Store.append](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.append)|Yes|-|
|[torch.distributed.Store.check](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.check)|Yes|-|
|[torch.distributed.Store.has_extended_api](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.has_extended_api)|Yes|-|
|[torch.distributed.Store.multi_set](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.multi_set)|Yes|-|
|[torch.distributed.Store.multi_get](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.multi_get)|Yes|-|
|[torch.distributed.Store.timeout](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.Store.timeout)|Yes|-|
|[torch.distributed.TCPStore.\_\_init\_\_](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.TCPStore.__init__)|Yes|-|
|[torch.distributed.TCPStore.host](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.TCPStore.host)|Yes|-|
|[torch.distributed.TCPStore.libuvBackend](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.TCPStore.libuvBackend)|Yes|-|
|[torch.distributed.TCPStore.port](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.TCPStore.port)|Yes|-|
|[torch.distributed.HashStore.\_\_init\_\_](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.HashStore.__init__)|Yes|-|
|[torch.distributed.FileStore.\_\_init\_\_](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.FileStore.__init__)|Yes|-|
|[torch.distributed.FileStore.path](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.FileStore.path)|Yes|-|
|[torch.distributed.PrefixStore.\_\_init\_\_](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.PrefixStore.__init__)|Yes|-|
|[torch.distributed.PrefixStore.underlying_store](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.PrefixStore.underlying_store)|Yes|-|
|[torch.distributed.new_group](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.new_group)|Yes|When the `pg_options` function passes in an object of type `torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()`, configuring the `hccl_config` attribute of this variable can control the HCCL communication domain buffer size. For a specific example, refer to the "[hccl_buffer_size](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/hccl_buffer_size.md)" section of the *PyTorch Model Migration and Fine-tuning Guide*. Configuring the `group_name` field of the `hccl_config` variable attribute can set a custom name for the communication group of the HCCL communication domain, with a value being a string of no more than 32 characters.|
|[torch.distributed.get_group_rank](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.get_group_rank)|Yes|-|
|[torch.distributed.get_global_rank](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.get_global_rank)|Yes|-|
|[torch.distributed.get_process_group_ranks](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.get_process_group_ranks)|Yes|-|
|[torch.distributed.device_mesh.DeviceMesh](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.device_mesh.DeviceMesh)|Yes|-|
|[torch.distributed.send](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.send)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.distributed.recv](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.recv)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.distributed.isend](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.isend)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.distributed.irecv](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.irecv)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.distributed.batch_isend_irecv](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.batch_isend_irecv)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.distributed.P2POp](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.P2POp)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.distributed.broadcast](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.broadcast)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|[torch.distributed.broadcast_object_list](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.broadcast_object_list)|Yes|-|
|[torch.distributed.all_reduce](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.all_reduce)|Yes|Supports bf16, fp16, fp32, int32, int64, bool|
|[torch.distributed.reduce](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.reduce)|Yes|Supports bf16, fp16, fp32, uint8, int8, int32, int64, bool|
|[torch.distributed.all_gather](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.all_gather)|Yes|Supports bf16, fp16, fp32, int8, int32, bool|
|[torch.distributed.all_gather_into_tensor](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.all_gather_into_tensor)|Yes|Supports bf16, fp16, fp32, int8, int32, bool<br>world size does not support 3, 5, 6, 7|
|[torch.distributed.all_gather_object](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.all_gather_object)|Yes|-|
|[torch.distributed.gather](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.gather)|Yes|Supports bf16, fp16, fp32, int8, int32, bool<br>By setting `torch_npu.npu.use_compatible_impl(True)`, `torch.distributed.gather` switches to be consistent with the native implementation|
|[torch.distributed.gather_object](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.gather_object)|Yes|Supported input type is Python Object|
|[torch.distributed.scatter](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.scatter)|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>By setting `torch_npu.npu.use_compatible_impl(True)`, `torch.distributed.scatter` switches to be consistent with the native implementation|
|[torch.distributed.scatter_object_list](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.scatter_object_list)|Yes|Does not involve the `dtype` parameter|
|[torch.distributed.reduce_scatter](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.reduce_scatter)|Yes|Supports bf16, fp16, fp32, int8, int32|
|[torch.distributed.reduce_scatter_tensor](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.reduce_scatter_tensor)|Yes|Supports bf16, fp16, fp32, int8, int32<br>world size does not support 3, 5, 6, 7<br>For Atlas A2 Training Series, the "prod" operation does not support int16 and bf16 data types in the current version|
|[torch.distributed.all_to_all_single](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.all_to_all_single)|Yes|Supports fp32|
|[torch.distributed.all_to_all](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.all_to_all)|Yes|Supports fp32<br>By setting `torch_npu.npu.use_compatible_impl(True)`, `torch.distributed.all_to_all` switches to be consistent with the native implementation|
|[torch.distributed.barrier](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.barrier)|Yes|-|
|[torch.distributed.monitored_barrier](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.monitored_barrier)|Yes|-|
|[torch.distributed.ReduceOp](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.ReduceOp)|Yes|Supports bf16, fp16, fp32, uint8, int8, int32, int64, bool|
|[torch.distributed.reduce_op](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.reduce_op)|Yes|Supports bf16, fp16, fp32, uint8, int8, int32, int64|
|[torch.distributed.DistBackendError](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.DistBackendError)|Yes|-|
|[torch.distributed.device_mesh.DeviceMesh.from_group](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.device_mesh.DeviceMesh.from_group)|Yes|-|
|[torch.distributed.device_mesh.DeviceMesh.get_all_groups](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_all_groups)|Yes|-|
|[torch.distributed.device_mesh.DeviceMesh.get_coordinate](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_coordinate)|Yes|-|
|[torch.distributed.device_mesh.DeviceMesh.get_group](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_group)|Yes|-|
|[torch.distributed.device_mesh.DeviceMesh.get_local_rank](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_local_rank)|Yes|-|
|[torch.distributed.device_mesh.DeviceMesh.get_rank](https://pytorch.org/docs/2.7/distributed.html#torch.distributed.device_mesh.DeviceMesh.get_rank)|Yes|-|
