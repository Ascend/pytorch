# torch.distributed.pipelining

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T01:05:44.419Z pushedAt=2026-06-15T02:04:36.498Z -->

> [!NOTE]  
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.distributed.pipelining.SplitPoint](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.SplitPoint)|Yes|-|
|[torch.distributed.pipelining.pipeline](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.pipeline)|Yes|-|
|[torch.distributed.pipelining.Pipe](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.Pipe)|Yes|-|
|[torch.distributed.pipelining.pipe_split](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.pipe_split)|Yes|-|
|[torch.distributed.pipelining.microbatch.TensorChunkSpec](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.microbatch.TensorChunkSpec)|Yes|-|
|[torch.distributed.pipelining.microbatch.split_args_kwargs_into_chunks](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.microbatch.split_args_kwargs_into_chunks)|Yes|-|
|[torch.distributed.pipelining.microbatch.merge_chunks](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.microbatch.merge_chunks)|Yes|-|
|[torch.distributed.pipelining.stage.PipelineStage](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.stage.PipelineStage)|Yes|-|
|[torch.distributed.pipelining.stage.build_stage](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.stage.build_stage)|Yes|-|
|[torch.distributed.pipelining.schedules.ScheduleGPipe](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleGPipe)|Yes|-|
|[torch.distributed.pipelining.schedules.Schedule1F1B](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.Schedule1F1B)|Yes|-|
|[torch.distributed.pipelining.schedules.ScheduleInterleaved1F1B](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleInterleaved1F1B)|Yes|-|
|[torch.distributed.pipelining.schedules.ScheduleLoopedBFS](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleLoopedBFS)|Yes|-|
|[torch.distributed.pipelining.schedules.PipelineScheduleSingle](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleSingle)|Yes|-|
|[torch.distributed.pipelining.schedules.PipelineScheduleSingle.step](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleSingle.step)|Yes|-|
|[torch.distributed.pipelining.schedules.PipelineScheduleMulti](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleMulti)|Yes|-|
|[torch.distributed.pipelining.schedules.PipelineScheduleMulti.step](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleMulti.step)|Yes|-|
|[torch.distributed.pipelining.schedules.ScheduleInterleavedZeroBubble](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleInterleavedZeroBubble)|Yes|-|
