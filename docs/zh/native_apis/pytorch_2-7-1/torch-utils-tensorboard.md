# torch.utils.tensorboard

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.7/tensorboard.html)。

<div style="display:none;">

## &#8203;torch.utils.tensorboard

</div>

### <code><i>class</i></code> torch.utils.tensorboard.writer.SummaryWriter

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">add_scalar()</font>

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">add_scalars()</font>

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">add_histogram()</font>

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">add_image()</font>

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`img_tensor`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_images()</font>

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`img_tensor`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_figure()</font>

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">add_video()</font>

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`vid_tensor`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_audio()</font>

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`snd_tensor`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_text()</font>

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">add_graph()</font>

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持bf16，fp16，fp32

</div>

> <font size="3">add_embedding()</font>

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`mat`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_pr_curve()</font>

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`labels`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_custom_scalars()</font>

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">add_mesh()</font>

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`vertices`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_hparams()</font>

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">flush()</font>

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">close()</font>

<div style="margin-left: 2em">

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>
