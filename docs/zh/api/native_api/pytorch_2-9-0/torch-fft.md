# torch.fft

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.9/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
>
> - 使用以下接口，需要配置以下信息：
>
>   `source {CANN包安装路径}/nnal/atb/set_env.sh`
>
>   `source {CANN包安装路径}/nnal/asdsip/set_env.sh`
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.9/fft.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Fast Fourier Transforms](#fast-fourier-transforms)
- [Helper Functions](#helper-functions)

</div>

<div style="display:none;">

## &#8203;torch.fft

</div>

## Fast Fourier Transforms

### torch.fft.fft

<div style="margin-left: 2em">

**原生文档**：[torch.fft.fft](https://docs.pytorch.org/docs/2.9/generated/torch.fft.fft.html#torch.fft.fft)

**产品支持情况**：

<!-- npu="910b" id35 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="A3" id36 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id36 -->
<!-- npu="950" id37 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id37 -->

</div>

### torch.fft.ifft

<div style="margin-left: 2em">

**原生文档**：[torch.fft.ifft](https://docs.pytorch.org/docs/2.9/generated/torch.fft.ifft.html#torch.fft.ifft)

**产品支持情况**：

<!-- npu="910b" id44 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="A3" id45 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id45 -->
<!-- npu="950" id46 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id46 -->

</div>

### torch.fft.fft2

<div style="margin-left: 2em">

**原生文档**：[torch.fft.fft2](https://docs.pytorch.org/docs/2.9/generated/torch.fft.fft2.html#torch.fft.fft2)

**产品支持情况**：

<!-- npu="910b" id38 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="A3" id39 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id39 -->
<!-- npu="950" id40 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id40 -->

</div>

### torch.fft.ifft2

<div style="margin-left: 2em">

**原生文档**：[torch.fft.ifft2](https://docs.pytorch.org/docs/2.9/generated/torch.fft.ifft2.html#torch.fft.ifft2)

**产品支持情况**：

<!-- npu="910b" id47 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="A3" id48 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id48 -->
<!-- npu="950" id49 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id49 -->

</div>

### torch.fft.fftn

<div style="margin-left: 2em">

**原生文档**：[torch.fft.fftn](https://docs.pytorch.org/docs/2.9/generated/torch.fft.fftn.html#torch.fft.fftn)

**产品支持情况**：

<!-- npu="910b" id41 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="A3" id42 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id42 -->
<!-- npu="950" id43 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id43 -->

</div>

### torch.fft.ifftn

<div style="margin-left: 2em">

**原生文档**：[torch.fft.ifftn](https://docs.pytorch.org/docs/2.9/generated/torch.fft.ifftn.html#torch.fft.ifftn)

**产品支持情况**：

<!-- npu="910b" id50 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="A3" id51 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id51 -->
<!-- npu="950" id52 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id52 -->

</div>

### torch.fft.rfft

<div style="margin-left: 2em">

**原生文档**：[torch.fft.rfft](https://docs.pytorch.org/docs/2.9/generated/torch.fft.rfft.html#torch.fft.rfft)

**产品支持情况**：

<!-- npu="910b" id56 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="A3" id57 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id57 -->
<!-- npu="950" id58 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id58 -->

</div>

### torch.fft.irfft

<div style="margin-left: 2em">

**原生文档**：[torch.fft.irfft](https://docs.pytorch.org/docs/2.9/generated/torch.fft.irfft.html#torch.fft.irfft)

**产品支持情况**：

<!-- npu="910b" id53 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="A3" id54 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id54 -->
<!-- npu="950" id55 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id55 -->

</div>

### torch.fft.rfftn

<div style="margin-left: 2em">

**原生文档**：[torch.fft.rfftn](https://pytorch.org/docs/2.9/generated/torch.fft.rfftn.html)

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

**限制与说明**：

- `input`仅支持fp32
- 数值范围：每个元素必须在[-100, 100]内
- 支持1-8维。2维维度为(batch, n)，1维维度为(n)
    1. batch维度：[1, 8, 16, 24, 32, 64]
    2. n维度限制（满足任一即可）：
        - $2^n$，n为23以内
        - 2, 3, 5, 7任意相乘，例如：2\*2\*2\*3\*3\*5\*7\*7，但结果需在1000000以内
        - 200以内质数任意相乘，结果需在100000以内

</div>

### torch.fft.hfft

<div style="margin-left: 2em">

**原生文档**：[torch.fft.hfft](https://pytorch.org/docs/2.9/generated/torch.fft.hfft.html)

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

</div>

### torch.fft.ihfft

<div style="margin-left: 2em">

**原生文档**：[torch.fft.ihfft](https://pytorch.org/docs/2.9/generated/torch.fft.ihfft.html)

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

</div>

### torch.fft.hfft2

<div style="margin-left: 2em">

**原生文档**：[torch.fft.hfft2](https://pytorch.org/docs/2.9/generated/torch.fft.hfft2.html)

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

</div>

### torch.fft.ihfft2

<div style="margin-left: 2em">

**原生文档**：[torch.fft.ihfft2](https://pytorch.org/docs/2.9/generated/torch.fft.ihfft2.html)

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

</div>

### torch.fft.hfftn

<div style="margin-left: 2em">

**原生文档**：[torch.fft.hfftn](https://pytorch.org/docs/2.9/generated/torch.fft.hfftn.html)

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

### torch.fft.ihfftn

<div style="margin-left: 2em">

**原生文档**：[torch.fft.ihfftn](https://pytorch.org/docs/2.9/generated/torch.fft.ihfftn.html)

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

</div>

## Helper Functions

### torch.fft.fftfreq

<div style="margin-left: 2em">

**原生文档**：[torch.fft.fftfreq](https://pytorch.org/docs/2.9/generated/torch.fft.fftfreq.html)

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

</div>

### torch.fft.rfftfreq

<div style="margin-left: 2em">

**原生文档**：[torch.fft.rfftfreq](https://pytorch.org/docs/2.9/generated/torch.fft.rfftfreq.html)

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

</div>

### torch.fft.fftshift

<div style="margin-left: 2em">

**原生文档**：[torch.fft.fftshift](https://pytorch.org/docs/2.9/generated/torch.fft.fftshift.html)

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

</div>

### torch.fft.ifftshift

<div style="margin-left: 2em">

**原生文档**：[torch.fft.ifftshift](https://pytorch.org/docs/2.9/generated/torch.fft.ifftshift.html)

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

<!-- npu="950" id34 -->
**限制与说明**： <term>Ascend 950DT系列产品</term>：不支持fp64，complex64，complex128
<!-- end id34 -->

</div>
