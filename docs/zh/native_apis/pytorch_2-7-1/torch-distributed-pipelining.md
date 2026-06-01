# torch.distributed.pipelining

> [!NOTE]  
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|[torch.distributed.pipelining.SplitPoint](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.SplitPoint)|是|-|
|[torch.distributed.pipelining.pipeline](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.pipeline)|是|-|
|[torch.distributed.pipelining.Pipe](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.Pipe)|是|-|
|[torch.distributed.pipelining.pipe_split](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.pipe_split)|是|-|
|[torch.distributed.pipelining.microbatch.TensorChunkSpec](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.microbatch.TensorChunkSpec)|是|-|
|[torch.distributed.pipelining.microbatch.split_args_kwargs_into_chunks](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.microbatch.split_args_kwargs_into_chunks)|是|-|
|[torch.distributed.pipelining.microbatch.merge_chunks](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.microbatch.merge_chunks)|是|-|
|[torch.distributed.pipelining.stage.PipelineStage](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.stage.PipelineStage)|是|-|
|[torch.distributed.pipelining.stage.build_stage](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.stage.build_stage)|是|-|
|[torch.distributed.pipelining.schedules.ScheduleGPipe](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleGPipe)|是|-|
|[torch.distributed.pipelining.schedules.Schedule1F1B](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.Schedule1F1B)|是|-|
|[torch.distributed.pipelining.schedules.ScheduleInterleaved1F1B](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleInterleaved1F1B)|是|-|
|[torch.distributed.pipelining.schedules.ScheduleLoopedBFS](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleLoopedBFS)|是|-|
|[torch.distributed.pipelining.schedules.PipelineScheduleSingle](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleSingle)|是|-|
|[torch.distributed.pipelining.schedules.PipelineScheduleSingle.step](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleSingle.step)|是|-|
|[torch.distributed.pipelining.schedules.PipelineScheduleMulti](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleMulti)|是|-|
|[torch.distributed.pipelining.schedules.PipelineScheduleMulti.step](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleMulti.step)|是|-|
|[torch.distributed.pipelining.schedules.ScheduleInterleavedZeroBubble](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleInterleavedZeroBubble)|是|-|
