# torch.utils.tensorboard

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.11/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.11/tensorboard.html)。

<div style="display:none;">

## &#8203;torch.utils.tensorboard

</div>

### <code><i>class</i></code> torch.utils.tensorboard.writer.SummaryWriter

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id3 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.\_\_init\_\_](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.__init__)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id6 -->

</div>

> <font size="3">add_scalar()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_scalar](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalar)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id9 -->

</div>

> <font size="3">add_scalars()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_scalars](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalars)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id12 -->

</div>

> <font size="3">add_histogram()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_histogram](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_histogram)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id15 -->

</div>

> <font size="3">add_image()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_image](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_image)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id18 -->

**限制与说明**：`img_tensor`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_images()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_images](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_images)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id21 -->

**限制与说明**：`img_tensor`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_figure()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_figure](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_figure)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id24 -->

</div>

> <font size="3">add_video()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_video](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_video)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id27 -->

**限制与说明**：`vid_tensor`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_audio()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_audio](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_audio)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id30 -->

**限制与说明**：`snd_tensor`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_text()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_text](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_text)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id33 -->

</div>

> <font size="3">add_graph()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_graph](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_graph)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id36 -->

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

> <font size="3">add_embedding()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_embedding](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_embedding)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id39 -->

**限制与说明**：`mat`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_pr_curve()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_pr_curve](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_pr_curve)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id42 -->

**限制与说明**：`labels`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_custom_scalars()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_custom_scalars](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_custom_scalars)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id45 -->

</div>

> <font size="3">add_mesh()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_mesh](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_mesh)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id48 -->

**限制与说明**： `vertices`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_hparams()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_hparams](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_hparams)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id51 -->

</div>

> <font size="3">flush()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.flush](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.flush)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id54 -->

</div>

> <font size="3">close()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.close](https://pytorch.org/docs/2.11/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.close)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id57 -->

</div>

</div>
