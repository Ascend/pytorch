# torch.distributed.checkpoint

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T02:15:00.946Z pushedAt=2026-06-15T03:25:49.158Z -->

> [!NOTE]  
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.distributed.checkpoint.state_dict_saver.save](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.save)|Yes|-|
|[torch.distributed.checkpoint.state_dict_saver.save_state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.save_state_dict)|Yes|-|
|[torch.distributed.checkpoint.state_dict_loader.load_state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_loader.load_state_dict)|Yes|-|
|[torch.distributed.checkpoint.stateful.Stateful](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.stateful.Stateful)|Yes|-|
|[torch.distributed.checkpoint.stateful.Stateful.load_state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.stateful.Stateful.load_state_dict)|Yes|-|
|[torch.distributed.checkpoint.stateful.Stateful.state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.stateful.Stateful.state_dict)|Yes|-|
|[torch.distributed.checkpoint.StorageReader](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader)|Yes|-|
|[torch.distributed.checkpoint.StorageReader.prepare_global_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.prepare_global_plan)|Yes|-|
|[torch.distributed.checkpoint.StorageReader.prepare_local_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.prepare_local_plan)|Yes|-|
|[torch.distributed.checkpoint.StorageReader.read_data](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.read_data)|Yes|-|
|[torch.distributed.checkpoint.StorageReader.read_metadata](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.read_metadata)|Yes|-|
|[torch.distributed.checkpoint.StorageReader.reset](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.reset)|Yes|-|
|[torch.distributed.checkpoint.StorageReader.set_up_storage_reader](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.set_up_storage_reader)|Yes|-|
|[torch.distributed.checkpoint.StorageReader.classmethod](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader.classmethod)|Yes|-|
|[torch.distributed.checkpoint.StorageWriter](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter)|Yes|-|
|[torch.distributed.checkpoint.StorageWriter.finish](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.finish)|Yes|-|
|[torch.distributed.checkpoint.StorageWriter.prepare_global_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.prepare_global_plan)|Yes|-|
|[torch.distributed.checkpoint.StorageWriter.prepare_local_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.prepare_local_plan)|Yes|-|
|[torch.distributed.checkpoint.StorageWriter.reset](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.reset)|Yes|-|
|[torch.distributed.checkpoint.StorageWriter.set_up_storage_writer](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.set_up_storage_writer)|Yes|-|
|[torch.distributed.checkpoint.StorageWriter.storage_meta](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.storage_meta)|Yes|-|
|[torch.distributed.checkpoint.StorageWriter.classmethod](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.classmethod)|Yes|-|
|[torch.distributed.checkpoint.StorageWriter.write_data](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter.write_data)|Yes|-|
|[torch.distributed.checkpoint.LoadPlanner](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner)|Yes|-|
|[torch.distributed.checkpoint.LoadPlanner.commit_tensor](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.commit_tensor)|Yes|-|
|[torch.distributed.checkpoint.LoadPlanner.create_global_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.create_global_plan)|Yes|-|
|[torch.distributed.checkpoint.LoadPlanner.create_local_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.create_local_plan)|Yes|-|
|[torch.distributed.checkpoint.LoadPlanner.finish_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.finish_plan)|Yes|-|
|[torch.distributed.checkpoint.LoadPlanner.load_bytes](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.load_bytes)|Yes|-|
|[torch.distributed.checkpoint.LoadPlanner.resolve_tensor](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.resolve_tensor)|Yes|-|
|[torch.distributed.checkpoint.LoadPlanner.set_up_planner](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner.set_up_planner)|Yes|-|
|[torch.distributed.checkpoint.LoadPlan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlan)|Yes|-|
|[torch.distributed.checkpoint.ReadItem](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.ReadItem)|Yes|-|
|[torch.distributed.checkpoint.SavePlanner](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner)|Yes|-|
|[torch.distributed.checkpoint.SavePlanner.create_global_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner.create_global_plan)|Yes|-|
|[torch.distributed.checkpoint.SavePlanner.create_local_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner.create_local_plan)|Yes|-|
|[torch.distributed.checkpoint.SavePlanner.finish_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner.finish_plan)|Yes|-|
|[torch.distributed.checkpoint.SavePlanner.resolve_data](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner.resolve_data)|Yes|-|
|[torch.distributed.checkpoint.SavePlanner.set_up_planner](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner.set_up_planner)|Yes|-|
|[torch.distributed.checkpoint.SavePlan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlan)|Yes|-|
|[torch.distributed.checkpoint.planner.WriteItem](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.planner.WriteItem)|Yes|-|
|[torch.distributed.checkpoint.planner.WriteItem.tensor_storage_size](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.planner.WriteItem.tensor_storage_size)|Yes|-|
|[torch.distributed.checkpoint.FileSystemReader](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.FileSystemReader)|Yes|-|
|[torch.distributed.checkpoint.FileSystemWriter](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.FileSystemWriter)|Yes|-|
|[torch.distributed.checkpoint.staging.AsyncStager](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.AsyncStager)|Yes|-|
|[torch.distributed.checkpoint.staging.AsyncStager.should_synchronize_after_execute](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.AsyncStager.should_synchronize_after_execute)|Yes|-|
|[torch.distributed.checkpoint.staging.AsyncStager.stage](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.AsyncStager.stage)|Yes|-|
|[torch.distributed.checkpoint.staging.AsyncStager.synchronize_staging](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.AsyncStager.synchronize_staging)|Yes|-|
|[torch.distributed.checkpoint.staging.BlockingAsyncStager](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.BlockingAsyncStager)|Yes|-|
|[torch.distributed.checkpoint.staging.BlockingAsyncStager.stage](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.BlockingAsyncStager.stage)|Yes|-|
|[torch.distributed.checkpoint.staging.BlockingAsyncStager.synchronize_staging](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.staging.BlockingAsyncStager.synchronize_staging)|Yes|-|
|[torch.distributed.checkpoint.DefaultSavePlanner](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultSavePlanner)|Yes|-|
|[torch.distributed.checkpoint.DefaultSavePlanner.lookup_object](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultSavePlanner.lookup_object)|Yes|-|
|[torch.distributed.checkpoint.DefaultSavePlanner.transform_object](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultSavePlanner.transform_object)|Yes|-|
|[torch.distributed.checkpoint.DefaultLoadPlanner](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultLoadPlanner)|Yes|-|
|[torch.distributed.checkpoint.DefaultLoadPlanner.lookup_tensor](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultLoadPlanner.lookup_tensor)|Yes|-|
|[torch.distributed.checkpoint.DefaultLoadPlanner.transform_tensor](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultLoadPlanner.transform_tensor)|Yes|-|
|[torch.distributed.checkpoint.state_dict.get_state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict)|Yes|-|
|[torch.distributed.checkpoint.state_dict.get_model_state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_model_state_dict)|Yes|-|
|[torch.distributed.checkpoint.state_dict.get_optimizer_state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_optimizer_state_dict)|Yes|-|
|[torch.distributed.checkpoint.state_dict.set_state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_state_dict)|Yes|-|
|[torch.distributed.checkpoint.state_dict.set_model_state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_model_state_dict)|Yes|-|
|[torch.distributed.checkpoint.state_dict.set_optimizer_state_dict](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_optimizer_state_dict)|Yes|-|
|[torch.distributed.checkpoint.state_dict.StateDictOptions](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.StateDictOptions)|Yes|-|
|[torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader)|Yes|-|
|[torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.read_metadata](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.read_metadata)|Yes|-|
|[torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.prepare_local_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.prepare_local_plan)|Yes|-|
|[torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.prepare_global_plan](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.prepare_global_plan)|Yes|-|
|[torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.read_data](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.read_data)|Yes|-|
|[torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.reset](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.reset)|Yes|-|
|[torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.set_up_storage_reader](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.set_up_storage_reader)|Yes|-|
|[torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.validate_checkpoint_id](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader.validate_checkpoint_id)|Yes|-|
|[torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner](https://pytorch.org/docs/2.9/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner)|Yes|-|
