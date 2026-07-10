# torch.distributed

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:35:23.381Z pushedAt=2026-07-09T08:44:08.335Z -->

> [!NOTE]
> If the "Supported" column is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

| API Name | Supported | Restrictions and Notes |
| -- | -- | -- |
| torch.distributed.is_available | Yes | - |
| torch.distributed.init_process_group | Yes | When the `pg_options` parameter is passed as `torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()`, configuring the `hccl_config` attribute of this variable can control the HCCL communication domain buffer size. For a specific example, see the "[hccl_buffer_size](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/hccl_buffer_size.md)" section of *PyTorch Training Model Migration and Tuning Guide*. Configuring the `group_name` field of the `hccl_config` attribute allows you to set a custom name for the communication group of the HCCL communication domain. The value must be a string no longer than 32 characters. |
| torch.distributed.is_initialized | Yes | - |
| torch.distributed.is_mpi_available | Yes | - |
| torch.distributed.is_nccl_available | Yes | - |
| torch.distributed.is_gloo_available | Yes | - |
| torch.distributed.is_torchelastic_launched | Yes | - |
| torch.distributed.Backend | Yes | - |
| torch.distributed.Backend.register_backend | Yes | - |
| torch.distributed.get_backend | Yes | - |
| torch.distributed.get_rank | Yes | - |
| torch.distributed.get_world_size | Yes | - |
| torch.distributed.Store | Yes | - |
| torch.distributed.TCPStore | Yes | - |
| torch.distributed.HashStore | Yes | - |
| torch.distributed.FileStore | Yes | - |
| torch.distributed.PrefixStore | Yes | - |
| torch.distributed.Store.set | Yes | - |
| torch.distributed.Store.get | Yes | - |
| torch.distributed.Store.add | Yes | - |
| torch.distributed.Store.compare_set | Yes | - |
| torch.distributed.Store.wait | Yes | - |
| torch.distributed.Store.num_keys | Yes | - |
| torch.distributed.Store.delete_key | Yes | - |
| torch.distributed.Store.set_timeout | Yes | - |
| torch.distributed.new_group | Yes | When the `pg_options` parameter is passed as `torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()`, configuring the `hccl_config` attribute of this variable can control the HCCL communication domain buffer size. For a specific example, see the "[hccl_buffer_size](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/hccl_buffer_size.md)" section of *PyTorch Training Model Migration and Tuning Guide*. Configuring the `group_name` field of the `hccl_config` attribute allows you to set a custom name for the communication group of the HCCL communication domain. The value must be a string no longer than 32 characters. |
| torch.distributed.get_group_rank | Yes | - |
| torch.distributed.get_global_rank | Yes | - |
| torch.distributed.get_process_group_ranks | Yes | - |
| torch.distributed.device_mesh.DeviceMesh | Yes | - |
| torch.distributed.send | Yes | Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool |
| torch.distributed.recv | Yes | Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool |
| torch.distributed.isend | Yes | Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool |
| torch.distributed.irecv | Yes | Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool |
| torch.distributed.batch_isend_irecv | Yes | Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool |
| torch.distributed.P2POp | Yes | Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool |
| torch.distributed.broadcast | Yes | Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool |
| torch.distributed.broadcast_object_list | Yes | - |
| torch.distributed.all_reduce | Yes | Supports bf16, fp16, fp32, int32, int64, bool |
| torch.distributed.reduce | Yes | Supports bf16, fp16, fp32, uint8, int8, int32, int64, bool |
| torch.distributed.all_gather | Yes | Supports bf16, fp16, fp32, int8, int32, bool |
| torch.distributed.all_gather_into_tensor | Yes | Supports bf16, fp16, fp32, int8, int32, bool<br>world size does not support 3, 5, 6, 7 |
| torch.distributed.all_gather_object | Yes | - |
| torch.distributed.gather | Yes | Supports bf16, fp16, fp32, int8, int32, bool<br>By setting `torch_npu.npu.use_compatible_impl(True)`, `torch.distributed.gather` switches to be consistent with the native implementation |
| torch.distributed.gather_object | Yes | Supported input type is Python Object |
| torch.distributed.scatter | Yes | Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool<br>By setting `torch_npu.npu.use_compatible_impl(True)`, `torch.distributed.scatter` switches to be consistent with the native implementation |
| torch.distributed.scatter_object_list | Yes | Does not involve the dtype parameter |
| torch.distributed.reduce_scatter | Yes | Supports bf16, fp16, fp32, int8, int32 |
| torch.distributed.reduce_scatter_tensor | Yes | Supports bf16, fp16, fp32, int8, int32<br>world size does not support 3, 5, 6, 7<br>For <term>Atlas A2 Training Series</term>, the "prod" operation does not support int16 and bf16 data types in the current version |
| torch.distributed.all_to_all_single | Yes | Supports fp32 |
| torch.distributed.all_to_all | Yes | Supports fp32<br>By setting `torch_npu.npu.use_compatible_impl(True)`, `torch.distributed.all_to_all` switches to be consistent with the native implementation |
| torch.distributed.barrier | Yes | - |
| torch.distributed.monitored_barrier | Yes | - |
| torch.distributed.ReduceOp | Yes | Supports bf16, fp16, fp32, uint8, int8, int32, int64, bool |
| torch.distributed.reduce_op | Yes | Supports bf16, fp16, fp32, uint8, int8, int32, int64 |
| torch.distributed.DistBackendError | Yes | - |
| torch.distributed.device_mesh.DeviceMesh.from_group | Yes | - |
| torch.distributed.device_mesh.DeviceMesh.get_all_groups | Yes | - |
| torch.distributed.device_mesh.DeviceMesh.get_coordinate | Yes | - |
| torch.distributed.device_mesh.DeviceMesh.get_group | Yes | - |
| torch.distributed.device_mesh.DeviceMesh.get_local_rank | Yes | - |
| torch.distributed.device_mesh.DeviceMesh.get_rank | Yes | - |
