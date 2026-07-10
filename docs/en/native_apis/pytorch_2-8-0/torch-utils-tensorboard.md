# torch.utils.tensorboard

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:34:37.165Z pushedAt=2026-07-09T08:44:08.326Z -->

> [!NOTE]
> If the "Supported" column shows "Yes" and the "Restrictions and Notes" column shows "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.utils.tensorboard.writer.SummaryWriter|Yes|-|
|torch.utils.tensorboard.writer.SummaryWriter.\_\_init_\_|Yes|-|
|torch.utils.tensorboard.writer.SummaryWriter.add_scalar|Yes|-|
|torch.utils.tensorboard.writer.SummaryWriter.add_scalars|Yes|-|
|torch.utils.tensorboard.writer.SummaryWriter.add_histogram|Yes|-|
|torch.utils.tensorboard.writer.SummaryWriter.add_image|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.utils.tensorboard.writer.SummaryWriter.add_images|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.utils.tensorboard.writer.SummaryWriter.add_figure|Yes|-|
|torch.utils.tensorboard.writer.SummaryWriter.add_video|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.utils.tensorboard.writer.SummaryWriter.add_audio|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.utils.tensorboard.writer.SummaryWriter.add_text|Yes|-|
|torch.utils.tensorboard.writer.SummaryWriter.add_graph|Yes|Supports bf16, fp16, fp32|
|torch.utils.tensorboard.writer.SummaryWriter.add_embedding|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.utils.tensorboard.writer.SummaryWriter.add_pr_curve|Yes|Supports fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.utils.tensorboard.writer.SummaryWriter.add_custom_scalars|Yes|-|
|torch.utils.tensorboard.writer.SummaryWriter.add_mesh|Yes|Supports bf16, fp16, fp32, fp64, uint8, int8, int16, int32, int64, bool|
|torch.utils.tensorboard.writer.SummaryWriter.add_hparams|Yes|-|
|torch.utils.tensorboard.writer.SummaryWriter.flush|Yes|-|
|torch.utils.tensorboard.writer.SummaryWriter.close|Yes|-|
