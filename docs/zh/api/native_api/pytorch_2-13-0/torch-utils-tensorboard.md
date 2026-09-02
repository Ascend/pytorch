# torch.utils.tensorboard

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.13/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.13/tensorboard.html)。

<div style="display:none;">

## &#8203;torch.utils.tensorboard

</div>

### <code><i>class</i></code> torch.utils.tensorboard.writer.SummaryWriter

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.\_\_init\_\_](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.__init__)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">add_scalar()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_scalar](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalar)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">add_scalars()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_scalars](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalars)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">add_histogram()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_histogram](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_histogram)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">add_image()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_image](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_image)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`img_tensor`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_images()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_images](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_images)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`img_tensor`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_figure()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_figure](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_figure)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">add_video()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_video](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_video)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`vid_tensor`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_audio()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_audio](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_audio)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`snd_tensor`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_text()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_text](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_text)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">add_graph()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_graph](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_graph)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `input`仅支持bf16，fp16，fp32

</div>

> <font size="3">add_embedding()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_embedding](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_embedding)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`mat`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_pr_curve()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_pr_curve](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_pr_curve)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`labels`仅支持fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_custom_scalars()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_custom_scalars](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_custom_scalars)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">add_mesh()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_mesh](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_mesh)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `vertices`仅支持bf16，fp16，fp32，fp64，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">add_hparams()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.add_hparams](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_hparams)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">flush()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.flush](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.flush)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">close()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.utils.tensorboard.writer.SummaryWriter.close](https://pytorch.org/docs/2.13/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.close)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>
