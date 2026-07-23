# torch.utils.tensorboard

> [!NOTE]  
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)

## base API

### _`class`_ torch.utils.tensorboard.SummaryWriter

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.SummaryWriter](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.utils.tensorboard.writer.SummaryWriter

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">__init__()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.__init__](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.__init__)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">add_scalar()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_scalar](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalar)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">add_scalars()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_scalars](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalars)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">add_histogram()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_histogram](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_histogram)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">add_image()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_image](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_image)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_images()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_images](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_images)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_figure()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_figure](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_figure)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">add_video()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_video](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_video)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_audio()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_audio](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_audio)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_text()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_text](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_text)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">add_graph()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_graph](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_graph)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32

</div>

> <font size="3">add_embedding()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_embedding](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_embedding)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_pr_curve()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_pr_curve](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_pr_curve)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_custom_scalars()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_custom_scalars](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_custom_scalars)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">add_mesh()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_mesh](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_mesh)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_hparams()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_hparams](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_hparams)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">flush()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.flush](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.flush)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">close()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.close](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.close)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>
