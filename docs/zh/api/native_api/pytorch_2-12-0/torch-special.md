# torch.special

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.12/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.12/special.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Functions](#functions)

</div>

<div style="display:none;">

## &#8203;torch.special

</div>

## Functions

### torch.special.erf

<div style="margin-left: 2em">

**原生文档**：[torch.special.erf](https://pytorch.org/docs/2.12/special.html#torch.special.erf)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**：`input`仅支持fp16，fp32，int64，bool

</div>

### torch.special.erfc

<div style="margin-left: 2em">

**原生文档**：[torch.special.erfc](https://pytorch.org/docs/2.12/special.html#torch.special.erfc)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**：`input`仅支持fp16，fp32，int64，bool

</div>

### torch.special.erfinv

<div style="margin-left: 2em">

**原生文档**：[torch.special.erfinv](https://pytorch.org/docs/2.12/special.html#torch.special.erfinv)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.special.exp2

<div style="margin-left: 2em">

**原生文档**：[torch.special.exp2](https://pytorch.org/docs/2.12/special.html#torch.special.exp2)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.special.expit

<div style="margin-left: 2em">

**原生文档**：[torch.special.expit](https://pytorch.org/docs/2.12/special.html#torch.special.expit)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.special.ndtr

<div style="margin-left: 2em">

**原生文档**：[torch.special.ndtr](https://pytorch.org/docs/2.12/special.html#torch.special.ndtr)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.special.xlogy

<div style="margin-left: 2em">

**原生文档**：[torch.special.xlogy](https://pytorch.org/docs/2.12/special.html#torch.special.xlogy)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.special.log1p

<div style="margin-left: 2em">

**原生文档**：[torch.special.log1p](https://pytorch.org/docs/2.12/special.html#torch.special.log1p)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.special.logsumexp

<div style="margin-left: 2em">

**原生文档**：[torch.special.logsumexp](https://pytorch.org/docs/2.12/special.html#torch.special.logsumexp)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.special.multigammaln

<div style="margin-left: 2em">

**原生文档**：[torch.special.multigammaln](https://pytorch.org/docs/2.12/special.html#torch.special.multigammaln)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：

- `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool
- 参数`p`需为正整数，且`input`元素需满足`input > (p - 1) / 2`，否则结果为`nan`或`inf`

</div>
