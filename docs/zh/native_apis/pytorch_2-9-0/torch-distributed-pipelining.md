# torch.distributed.pipelining

> [!NOTE]  
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)
- [API Reference](#api-reference)

## base API

### torch.distributed.pipelining.pipeline

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.pipeline](https://pytorch.org/docs/2.9/distributed.pipelining.html#torch.distributed.pipelining.pipeline)

**是否支持**：是

</div>

### _`class`_ torch.distributed.pipelining.Pipe

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.Pipe](https://pytorch.org/docs/2.9/distributed.pipelining.html#torch.distributed.pipelining.Pipe)

**是否支持**：是

</div>

### torch.distributed.pipelining.pipe_split

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.pipe_split](https://pytorch.org/docs/2.9/distributed.pipelining.html#torch.distributed.pipelining.pipe_split)

**是否支持**：是

</div>

### torch.distributed.pipelining.microbatch.split_args_kwargs_into_chunks

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.microbatch.split_args_kwargs_into_chunks](https://pytorch.org/docs/2.9/distributed.pipelining.html#torch.distributed.pipelining.microbatch.split_args_kwargs_into_chunks)

**是否支持**：是

</div>

### torch.distributed.pipelining.microbatch.merge_chunks

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.microbatch.merge_chunks](https://pytorch.org/docs/2.9/distributed.pipelining.html#torch.distributed.pipelining.microbatch.merge_chunks)

**是否支持**：是

</div>

### _`class`_ torch.distributed.pipelining.stage.PipelineStage

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.stage.PipelineStage](https://pytorch.org/docs/2.9/distributed.pipelining.html#torch.distributed.pipelining.stage.PipelineStage)

**是否支持**：是

</div>

### torch.distributed.pipelining.stage.build_stage

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.stage.build_stage](https://pytorch.org/docs/2.9/distributed.pipelining.html#torch.distributed.pipelining.stage.build_stage)

**是否支持**：是

</div>

### _`class`_ torch.distributed.pipelining.schedules.ScheduleGPipe

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.schedules.ScheduleGPipe](https://pytorch.org/docs/2.9/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleGPipe)

**是否支持**：是

</div>

### _`class`_ torch.distributed.pipelining.schedules.Schedule1F1B

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.schedules.Schedule1F1B](https://pytorch.org/docs/2.9/distributed.pipelining.html#torch.distributed.pipelining.schedules.Schedule1F1B)

**是否支持**：是

</div>

### _`class`_ torch.distributed.pipelining.schedules.ScheduleInterleaved1F1B

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.schedules.ScheduleInterleaved1F1B](https://pytorch.org/docs/2.9/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleInterleaved1F1B)

**是否支持**：是

</div>

### _`class`_ torch.distributed.pipelining.schedules.ScheduleLoopedBFS

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.schedules.ScheduleLoopedBFS](https://pytorch.org/docs/2.9/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleLoopedBFS)

**是否支持**：是

</div>

### _`class`_ torch.distributed.pipelining.schedules.PipelineScheduleSingle

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.schedules.PipelineScheduleSingle](https://pytorch.org/docs/2.9/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleSingle)

**是否支持**：是

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.schedules.PipelineScheduleSingle.step](https://pytorch.org/docs/2.9/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleSingle.step)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributed.pipelining.schedules.PipelineScheduleMulti

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.schedules.PipelineScheduleMulti](https://pytorch.org/docs/2.9/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleMulti)

**是否支持**：是

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.schedules.PipelineScheduleMulti.step](https://pytorch.org/docs/2.9/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleMulti.step)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributed.pipelining.schedules.ScheduleInterleavedZeroBubble

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.schedules.ScheduleInterleavedZeroBubble](https://pytorch.org/docs/2.9/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleInterleavedZeroBubble)

**是否支持**：是

</div>

## API Reference

### _`class`_ torch.distributed.pipelining.SplitPoint

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.SplitPoint](https://pytorch.org/docs/2.9/distributed.pipelining.html#torch.distributed.pipelining.SplitPoint)

**是否支持**：是

</div>

### _`class`_ torch.distributed.pipelining.microbatch.TensorChunkSpec

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.microbatch.TensorChunkSpec](https://pytorch.org/docs/2.9/distributed.pipelining.html#torch.distributed.pipelining.microbatch.TensorChunkSpec)

**是否支持**：是

</div>
