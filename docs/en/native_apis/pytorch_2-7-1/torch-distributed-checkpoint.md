# torch.distributed.checkpoint

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:37:56.746Z pushedAt=2026-07-09T08:44:08.357Z -->

> [!NOTE]  
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.distributed.checkpoint.state_dict_saver.save_state_dict|Yes|-|
|torch.distributed.checkpoint.state_dict_loader.load_state_dict|Yes|-|
|torch.distributed.checkpoint.stateful.Stateful|Yes|-|
|torch.distributed.checkpoint.stateful.Stateful.load_state_dict|Yes|-|
|torch.distributed.checkpoint.stateful.Stateful.state_dict|Yes|-|
|torch.distributed.checkpoint.StorageReader|Yes|-|
|torch.distributed.checkpoint.StorageReader.prepare_global_plan|Yes|-|
|torch.distributed.checkpoint.StorageReader.prepare_local_plan|Yes|-|
|torch.distributed.checkpoint.StorageReader.read_data|Yes|-|
|torch.distributed.checkpoint.StorageReader.read_metadata|Yes|-|
|torch.distributed.checkpoint.StorageReader.reset|Yes|-|
|torch.distributed.checkpoint.StorageReader.set_up_storage_reader|Yes|-|
|torch.distributed.checkpoint.StorageReader.classmethod|Yes|-|
|torch.distributed.checkpoint.StorageWriter|Yes|-|
|torch.distributed.checkpoint.StorageWriter.finish|Yes|-|
|torch.distributed.checkpoint.StorageWriter.prepare_global_plan|Yes|-|
|torch.distributed.checkpoint.StorageWriter.prepare_local_plan|Yes|-|
|torch.distributed.checkpoint.StorageWriter.reset|Yes|-|
|torch.distributed.checkpoint.StorageWriter.set_up_storage_writer|Yes|-|
|torch.distributed.checkpoint.StorageWriter.storage_meta|Yes|-|
|torch.distributed.checkpoint.StorageWriter.classmethod|Yes|-|
|torch.distributed.checkpoint.StorageWriter.write_data|Yes|-|
|torch.distributed.checkpoint.LoadPlanner|Yes|-|
|torch.distributed.checkpoint.LoadPlanner.commit_tensor|Yes|-|
|torch.distributed.checkpoint.LoadPlanner.create_global_plan|Yes|-|
|torch.distributed.checkpoint.LoadPlanner.create_local_plan|Yes|-|
|torch.distributed.checkpoint.LoadPlanner.finish_plan|Yes|-|
|torch.distributed.checkpoint.LoadPlanner.load_bytes|Yes|-|
|torch.distributed.checkpoint.LoadPlanner.resolve_tensor|Yes|-|
|torch.distributed.checkpoint.LoadPlanner.set_up_planner|Yes|-|
|torch.distributed.checkpoint.LoadPlan|Yes|-|
|torch.distributed.checkpoint.ReadItem|Yes|-|
|torch.distributed.checkpoint.SavePlanner|Yes|-|
|torch.distributed.checkpoint.SavePlanner.create_global_plan|Yes|-|
|torch.distributed.checkpoint.SavePlanner.create_local_plan|Yes|-|
|torch.distributed.checkpoint.SavePlanner.finish_plan|Yes|-|
|torch.distributed.checkpoint.SavePlanner.resolve_data|Yes|-|
|torch.distributed.checkpoint.SavePlanner.set_up_planner|Yes|-|
|torch.distributed.checkpoint.SavePlan|Yes|-|
|torch.distributed.checkpoint.planner.WriteItem|Yes|-|
|torch.distributed.checkpoint.FileSystemReader|Yes|-|
|torch.distributed.checkpoint.FileSystemWriter|Yes|-|
|torch.distributed.checkpoint.DefaultSavePlanner|Yes|-|
|torch.distributed.checkpoint.DefaultSavePlanner.lookup_object|Yes|-|
|torch.distributed.checkpoint.DefaultSavePlanner.transform_object|Yes|-|
|torch.distributed.checkpoint.DefaultLoadPlanner|Yes|-|
|torch.distributed.checkpoint.DefaultLoadPlanner.lookup_tensor|Yes|-|
|torch.distributed.checkpoint.DefaultLoadPlanner.transform_tensor|Yes|-|
|torch.distributed.checkpoint.state_dict.get_state_dict|Yes|-|
|torch.distributed.checkpoint.state_dict.get_model_state_dict|Yes|-|
|torch.distributed.checkpoint.state_dict.get_optimizer_state_dict|Yes|-|
|torch.distributed.checkpoint.state_dict.set_state_dict|Yes|-|
|torch.distributed.checkpoint.state_dict.set_model_state_dict|Yes|-|
|torch.distributed.checkpoint.state_dict.set_optimizer_state_dict|Yes|-|
|torch.distributed.checkpoint.state_dict.StateDictOptions|Yes|-|
