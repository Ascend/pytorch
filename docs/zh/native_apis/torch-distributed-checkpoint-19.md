# torch.distributed.checkpoint

> [!NOTE]  
> 若API“是否支持“为“是“，“限制与说明“为“-“，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.distributed.checkpoint.state_dict_saver.save_state_dict|是|-|
|torch.distributed.checkpoint.state_dict_loader.load_state_dict|是|-|
|torch.distributed.checkpoint.stateful.Stateful|是|-|
|torch.distributed.checkpoint.stateful.Stateful.load_state_dict|是|-|
|torch.distributed.checkpoint.stateful.Stateful.state_dict|是|-|
|torch.distributed.checkpoint.StorageReader|是|-|
|torch.distributed.checkpoint.StorageReader.prepare_global_plan|是|-|
|torch.distributed.checkpoint.StorageReader.prepare_local_plan|是|-|
|torch.distributed.checkpoint.StorageReader.read_data|是|-|
|torch.distributed.checkpoint.StorageReader.read_metadata|是|-|
|torch.distributed.checkpoint.StorageReader.reset|是|-|
|torch.distributed.checkpoint.StorageReader.set_up_storage_reader|是|-|
|torch.distributed.checkpoint.StorageReader.classmethod|是|-|
|torch.distributed.checkpoint.StorageWriter|是|-|
|torch.distributed.checkpoint.StorageWriter.finish|是|-|
|torch.distributed.checkpoint.StorageWriter.prepare_global_plan|是|-|
|torch.distributed.checkpoint.StorageWriter.prepare_local_plan|是|-|
|torch.distributed.checkpoint.StorageWriter.reset|是|-|
|torch.distributed.checkpoint.StorageWriter.set_up_storage_writer|是|-|
|torch.distributed.checkpoint.StorageWriter.storage_meta|是|-|
|torch.distributed.checkpoint.StorageWriter.classmethod|是|-|
|torch.distributed.checkpoint.StorageWriter.write_data|是|-|
|torch.distributed.checkpoint.LoadPlanner|是|-|
|torch.distributed.checkpoint.LoadPlanner.commit_tensor|是|-|
|torch.distributed.checkpoint.LoadPlanner.create_global_plan|是|-|
|torch.distributed.checkpoint.LoadPlanner.create_local_plan|是|-|
|torch.distributed.checkpoint.LoadPlanner.finish_plan|是|-|
|torch.distributed.checkpoint.LoadPlanner.load_bytes|是|-|
|torch.distributed.checkpoint.LoadPlanner.resolve_tensor|是|-|
|torch.distributed.checkpoint.LoadPlanner.set_up_planner|是|-|
|torch.distributed.checkpoint.LoadPlan|是|-|
|torch.distributed.checkpoint.ReadItem|是|-|
|torch.distributed.checkpoint.SavePlanner|是|-|
|torch.distributed.checkpoint.SavePlanner.create_global_plan|是|-|
|torch.distributed.checkpoint.SavePlanner.create_local_plan|是|-|
|torch.distributed.checkpoint.SavePlanner.finish_plan|是|-|
|torch.distributed.checkpoint.SavePlanner.resolve_data|是|-|
|torch.distributed.checkpoint.SavePlanner.set_up_planner|是|-|
|torch.distributed.checkpoint.SavePlan|是|-|
|torch.distributed.checkpoint.planner.WriteItem|是|-|
|torch.distributed.checkpoint.FileSystemReader|是|-|
|torch.distributed.checkpoint.FileSystemWriter|是|-|
|torch.distributed.checkpoint.DefaultSavePlanner|是|-|
|torch.distributed.checkpoint.DefaultSavePlanner.lookup_object|是|-|
|torch.distributed.checkpoint.DefaultSavePlanner.transform_object|是|-|
|torch.distributed.checkpoint.DefaultLoadPlanner|是|-|
|torch.distributed.checkpoint.DefaultLoadPlanner.lookup_tensor|是|-|
|torch.distributed.checkpoint.DefaultLoadPlanner.transform_tensor|是|-|
|torch.distributed.checkpoint.state_dict.get_state_dict|是|-|
|torch.distributed.checkpoint.state_dict.get_model_state_dict|是|-|
|torch.distributed.checkpoint.state_dict.get_optimizer_state_dict|是|-|
|torch.distributed.checkpoint.state_dict.set_state_dict|是|-|
|torch.distributed.checkpoint.state_dict.set_model_state_dict|是|-|
|torch.distributed.checkpoint.state_dict.set_optimizer_state_dict|是|-|
|torch.distributed.checkpoint.state_dict.StateDictOptions|是|-|


