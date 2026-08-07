# torch.distributed.pipelining

> [!NOTE]
>
> - API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。

## 目录

- [base API](#base-api)
- [API Reference](#api-reference)

## base API

### torch.distributed.pipelining.pipeline

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.pipeline](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.pipeline)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### <code><i>class</i></code> torch.distributed.pipelining.Pipe

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.Pipe](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.Pipe)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### torch.distributed.pipelining.pipe_split

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.pipe_split](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.pipe_split)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### torch.distributed.pipelining.microbatch.split_args_kwargs_into_chunks

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.microbatch.split_args_kwargs_into_chunks](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.microbatch.split_args_kwargs_into_chunks)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### torch.distributed.pipelining.microbatch.merge_chunks

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.microbatch.merge_chunks](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.microbatch.merge_chunks)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### <code><i>class</i></code> torch.distributed.pipelining.stage.PipelineStage

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.stage.PipelineStage](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.stage.PipelineStage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### torch.distributed.pipelining.stage.build_stage

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.stage.build_stage](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.stage.build_stage)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### <code><i>class</i></code> torch.distributed.pipelining.schedules.ScheduleGPipe

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.schedules.ScheduleGPipe](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleGPipe)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### <code><i>class</i></code> torch.distributed.pipelining.schedules.Schedule1F1B

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.schedules.Schedule1F1B](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.Schedule1F1B)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### <code><i>class</i></code> torch.distributed.pipelining.schedules.ScheduleInterleaved1F1B

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.schedules.ScheduleInterleaved1F1B](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleInterleaved1F1B)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### <code><i>class</i></code> torch.distributed.pipelining.schedules.ScheduleLoopedBFS

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.schedules.ScheduleLoopedBFS](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleLoopedBFS)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### <code><i>class</i></code> torch.distributed.pipelining.schedules.PipelineScheduleSingle

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.schedules.PipelineScheduleSingle](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleSingle)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.schedules.PipelineScheduleSingle.step](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleSingle.step)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

</div>

### <code><i>class</i></code> torch.distributed.pipelining.schedules.PipelineScheduleMulti

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.schedules.PipelineScheduleMulti](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleMulti)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

> <font size="3">step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.schedules.PipelineScheduleMulti.step](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.PipelineScheduleMulti.step)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

</div>

### <code><i>class</i></code> torch.distributed.pipelining.schedules.ScheduleInterleavedZeroBubble

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.schedules.ScheduleInterleavedZeroBubble](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleInterleavedZeroBubble)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

## API Reference

### <code><i>class</i></code> torch.distributed.pipelining.SplitPoint

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.SplitPoint](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.SplitPoint)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### <code><i>class</i></code> torch.distributed.pipelining.microbatch.TensorChunkSpec

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.pipelining.microbatch.TensorChunkSpec](https://pytorch.org/docs/2.7/distributed.pipelining.html#torch.distributed.pipelining.microbatch.TensorChunkSpec)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>
