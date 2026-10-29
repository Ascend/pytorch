# torch.utils.tensorboard

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.12/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.12/tensorboard.html).

<div style="display:none;">

## &#8203;torch.utils.tensorboard

</div>

### <code><i>class</i></code> torch.utils.tensorboard.writer.SummaryWriter

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id3 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter.\_\_init\_\_](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.__init__)

**Support Status**:

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id6 -->

</div>

> <font size="3">add_scalar()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter.add_scalar](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalar)

**Support Status**:

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id9 -->

</div>

> <font size="3">add_scalars()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter.add_scalars](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalars)

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id12 -->

</div>

> <font size="3">add_histogram()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter.add_histogram](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_histogram)

**Support Status**:

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id15 -->

</div>

> <font size="3">add_image()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter.add_image](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_image)

**Support Status**:

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id18 -->

**Constraints & Notes**: `img_tensor` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">add_images()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter.add_images](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_images)

**Support Status**:

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id21 -->

**Constraints & Notes**: `img_tensor` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">add_figure()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter.add_figure](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_figure)

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id24 -->

</div>

> <font size="3">add_video()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter.add_video](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_video)

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id27 -->

**Constraints & Notes**: `vid_tensor` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">add_audio()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter.add_audio](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_audio)

**Support Status**:

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id30 -->

**Constraints & Notes**: `snd_tensor` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">add_text()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter.add_text](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_text)

**Support Status**:

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id33 -->

</div>

> <font size="3">add_graph()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter.add_graph](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_graph)

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id36 -->

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

> <font size="3">add_embedding()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter.add_embedding](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_embedding)

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id39 -->

**Constraints & Notes**: `mat` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">add_pr_curve()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter.add_pr_curve](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_pr_curve)

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id42 -->

**Constraints & Notes**: `labels` only supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">add_custom_scalars()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter.add_custom_scalars](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_custom_scalars)

**Support Status**:

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id45 -->

</div>

> <font size="3">add_mesh()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter.add_mesh](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_mesh)

**Support Status**:

<!-- npu="910b" id46 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id48 -->

**Constraints & Notes**: `vertices` only supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">add_hparams()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter.add_hparams](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_hparams)

**Support Status**:

<!-- npu="910b" id49 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id51 -->

</div>

> <font size="3">flush()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter.flush](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.flush)

**Support Status**:

<!-- npu="910b" id52 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id54 -->

</div>

> <font size="3">close()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.tensorboard.writer.SummaryWriter.close](https://pytorch.org/docs/2.12/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.close)

**Support Status**:

<!-- npu="910b" id55 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id57 -->

</div>

</div>
