# torch.amp

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.7/amp.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Autocasting](#autocasting)
- [Gradient Scaling](#gradient-scaling)

</div>

<div style="display:none;">

## &#8203;torch.amp

</div>

## Autocasting

### torch.cuda.amp.autocast

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.amp.autocast](https://pytorch.org/docs/2.7/amp.html#torch.cuda.amp.autocast)

**NPU 形式名称**：torch_npu.npu.amp.autocast

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.amp.custom_bwd

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.amp.custom_bwd](https://pytorch.org/docs/2.7/amp.html#torch.cuda.amp.custom_bwd)

**NPU 形式名称**：torch_npu.npu.amp.custom_bwd

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cpu.amp.autocast

<div style="margin-left: 2em">

**原生文档**：[torch.cpu.amp.autocast](https://pytorch.org/docs/2.7/amp.html#torch.cpu.amp.autocast)

**NPU 形式名称**：torch.cpu.amp.autocast

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.autocast

<div style="margin-left: 2em">

**原生文档**：[torch.autocast](https://pytorch.org/docs/2.7/amp.html#torch.autocast)

**NPU 形式名称**：torch.autocast

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.cuda.amp.custom_fwd

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.amp.custom_fwd](https://pytorch.org/docs/2.7/amp.html#torch.cuda.amp.custom_fwd)

**NPU 形式名称**：torch_npu.npu.amp.custom_fwd

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

## Gradient Scaling

### <code><i>class</i></code> torch.cuda.amp.GradScaler

<div style="margin-left: 2em">

**原生文档**：[torch.cuda.amp.GradScaler](https://pytorch.org/docs/2.7/amp.html#torch.cuda.amp.GradScaler)

**NPU 形式名称**：torch_npu.npu.amp.GradScaler

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>
