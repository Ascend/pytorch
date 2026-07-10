# torch.distributed.pipelining

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:30:51.179Z pushedAt=2026-07-09T08:44:08.275Z -->

> [!NOTE]
> If the API's "Supported" column is "Yes" and "Restrictions and Notes" is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.distributed.pipelining.SplitPoint|Yes|-|
|torch.distributed.pipelining.pipeline|Yes|-|
|torch.distributed.pipelining.Pipe|Yes|-|
|torch.distributed.pipelining.pipe_split|Yes|-|
|torch.distributed.pipelining.microbatch.TensorChunkSpec|Yes|-|
|torch.distributed.pipelining.microbatch.split_args_kwargs_into_chunks|Yes|-|
|torch.distributed.pipelining.microbatch.merge_chunks|Yes|-|
|torch.distributed.pipelining.stage.PipelineStage|Yes|-|
|torch.distributed.pipelining.stage.build_stage|Yes|-|
|torch.distributed.pipelining.schedules.ScheduleGPipe|Yes|-|
|torch.distributed.pipelining.schedules.Schedule1F1B|Yes|-|
|torch.distributed.pipelining.schedules.ScheduleInterleaved1F1B|Yes|-|
|torch.distributed.pipelining.schedules.ScheduleLoopedBFS|Yes|-|
|torch.distributed.pipelining.schedules.PipelineScheduleSingle|Yes|-|
|torch.distributed.pipelining.schedules.PipelineScheduleSingle.step|Yes|-|
|torch.distributed.pipelining.schedules.PipelineScheduleMulti|Yes|-|
|torch.distributed.pipelining.schedules.PipelineScheduleMulti.step|Yes|-|
|torch.distributed.pipelining.schedules.ScheduleInterleavedZeroBubble|Yes|-|
