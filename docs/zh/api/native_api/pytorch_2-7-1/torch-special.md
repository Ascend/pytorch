# torch.special

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.7/special.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Functions](#functions)

</div>

<div style="display:none;">

## &#8203;torch.special

</div>

### torch.special.zeta

<div style="margin-left: 2em">

**原生文档**：[torch.special.zeta](https://docs.pytorch.org/docs/2.7/special.html#torch.special.zeta)

**产品支持情况**：

<!-- npu="910b" id91 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id93 -->

</div>

## Functions

### torch.special.airy_ai

<div style="margin-left: 2em">

**原生文档**：[torch.special.airy_ai](https://docs.pytorch.org/docs/2.7/special.html#torch.special.airy_ai)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id33 -->

</div>

### torch.special.bessel_j0

<div style="margin-left: 2em">

**原生文档**：[torch.special.bessel_j0](https://docs.pytorch.org/docs/2.7/special.html#torch.special.bessel_j0)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id36 -->

</div>

### torch.special.bessel_j1

<div style="margin-left: 2em">

**原生文档**：[torch.special.bessel_j1](https://docs.pytorch.org/docs/2.7/special.html#torch.special.bessel_j1)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id39 -->

</div>

### torch.special.digamma

<div style="margin-left: 2em">

**原生文档**：[torch.special.digamma](https://docs.pytorch.org/docs/2.7/special.html#torch.special.digamma)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id42 -->

</div>

### torch.special.entr

<div style="margin-left: 2em">

**原生文档**：[torch.special.entr](https://docs.pytorch.org/docs/2.7/special.html#torch.special.entr)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id45 -->

</div>

### torch.special.erf

<div style="margin-left: 2em">

**原生文档**：[torch.special.erf](https://pytorch.org/docs/2.7/special.html#torch.special.erf)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id3 -->

**限制与说明**：`input`仅支持fp16，fp32，int64，bool

</div>

### torch.special.erfc

<div style="margin-left: 2em">

**原生文档**：[torch.special.erfc](https://pytorch.org/docs/2.7/special.html#torch.special.erfc)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id6 -->

**限制与说明**：`input`仅支持fp16，fp32，int64，bool

</div>

### torch.special.erfcx

<div style="margin-left: 2em">

**原生文档**：[torch.special.erfcx](https://docs.pytorch.org/docs/2.7/special.html#torch.special.erfcx)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id48 -->

</div>

### torch.special.erfinv

<div style="margin-left: 2em">

**原生文档**：[torch.special.erfinv](https://pytorch.org/docs/2.7/special.html#torch.special.erfinv)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id9 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.special.exp2

<div style="margin-left: 2em">

**原生文档**：[torch.special.exp2](https://pytorch.org/docs/2.7/special.html#torch.special.exp2)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id12 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.special.expit

<div style="margin-left: 2em">

**原生文档**：[torch.special.expit](https://pytorch.org/docs/2.7/special.html#torch.special.expit)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id15 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool，complex64，complex128

</div>

### torch.special.gammainc

<div style="margin-left: 2em">

**原生文档**：[torch.special.gammainc](https://docs.pytorch.org/docs/2.7/special.html#torch.special.gammainc)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id51 -->

</div>

### torch.special.gammaincc

<div style="margin-left: 2em">

**原生文档**：[torch.special.gammaincc](https://docs.pytorch.org/docs/2.7/special.html#torch.special.gammaincc)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id54 -->

</div>

### torch.special.gammaln

<div style="margin-left: 2em">

**原生文档**：[torch.special.gammaln](https://docs.pytorch.org/docs/2.7/special.html#torch.special.gammaln)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id57 -->

</div>

### torch.special.i0

<div style="margin-left: 2em">

**原生文档**：[torch.special.i0](https://docs.pytorch.org/docs/2.7/special.html#torch.special.i0)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id60 -->

</div>

### torch.special.i0e

<div style="margin-left: 2em">

**原生文档**：[torch.special.i0e](https://docs.pytorch.org/docs/2.7/special.html#torch.special.i0e)

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id63 -->

</div>

### torch.special.i1

<div style="margin-left: 2em">

**原生文档**：[torch.special.i1](https://docs.pytorch.org/docs/2.7/special.html#torch.special.i1)

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id66 -->

</div>

### torch.special.i1e

<div style="margin-left: 2em">

**原生文档**：[torch.special.i1e](https://docs.pytorch.org/docs/2.7/special.html#torch.special.i1e)

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id69 -->

</div>

### torch.special.log_ndtr

<div style="margin-left: 2em">

**原生文档**：[torch.special.log_ndtr](https://docs.pytorch.org/docs/2.7/special.html#torch.special.log_ndtr)

**产品支持情况**：

<!-- npu="910b" id70 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id72 -->

</div>

### torch.special.ndtr

<div style="margin-left: 2em">

**原生文档**：[torch.special.ndtr](https://pytorch.org/docs/2.7/special.html#torch.special.ndtr)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id18 -->

</div>

### torch.special.ndtri

<div style="margin-left: 2em">

**原生文档**：[torch.special.ndtri](https://docs.pytorch.org/docs/2.7/special.html#torch.special.ndtri)

**产品支持情况**：

<!-- npu="910b" id73 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id75 -->

</div>

### torch.special.polygamma

<div style="margin-left: 2em">

**原生文档**：[torch.special.polygamma](https://docs.pytorch.org/docs/2.7/special.html#torch.special.polygamma)

**产品支持情况**：

<!-- npu="910b" id76 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id78 -->

</div>

### torch.special.scaled_modified_bessel_k0

<div style="margin-left: 2em">

**原生文档**：[torch.special.scaled_modified_bessel_k0](https://docs.pytorch.org/docs/2.7/special.html#torch.special.scaled_modified_bessel_k0)

**产品支持情况**：

<!-- npu="910b" id79 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id81 -->

</div>

### torch.special.scaled_modified_bessel_k1

<div style="margin-left: 2em">

**原生文档**：[torch.special.scaled_modified_bessel_k1](https://docs.pytorch.org/docs/2.7/special.html#torch.special.scaled_modified_bessel_k1)

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id84 -->

</div>

### torch.special.spherical_bessel_j0

<div style="margin-left: 2em">

**原生文档**：[torch.special.spherical_bessel_j0](https://docs.pytorch.org/docs/2.7/special.html#torch.special.spherical_bessel_j0)

**产品支持情况**：

<!-- npu="910b" id85 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id87 -->

</div>

### torch.special.xlog1py

<div style="margin-left: 2em">

**原生文档**：[torch.special.xlog1py](https://docs.pytorch.org/docs/2.7/special.html#torch.special.xlog1py)

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id90 -->

</div>

### torch.special.xlogy

<div style="margin-left: 2em">

**原生文档**：[torch.special.xlogy](https://pytorch.org/docs/2.7/special.html#torch.special.xlogy)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id21 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.special.log1p

<div style="margin-left: 2em">

**原生文档**：[torch.special.log1p](https://pytorch.org/docs/2.7/special.html#torch.special.log1p)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id24 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.special.logsumexp

<div style="margin-left: 2em">

**原生文档**：[torch.special.logsumexp](https://pytorch.org/docs/2.7/special.html#torch.special.logsumexp)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id27 -->

**限制与说明**： `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

### torch.special.multigammaln

<div style="margin-left: 2em">

**原生文档**：[torch.special.multigammaln](https://pytorch.org/docs/2.7/special.html#torch.special.multigammaln)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id30 -->

**限制与说明**：

- `input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool
- 参数`p`需为正整数，且`input`元素需满足`input > (p - 1) / 2`，否则结果为`nan`或`inf`

</div>
