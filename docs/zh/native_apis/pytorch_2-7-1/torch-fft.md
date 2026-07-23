# torch.fft

> [!NOTE]  
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>
> 
> 使用以下接口，需要配置以下信息：
>
> `source {CANN包安装路径}/nnal/atb/set_env.sh`
> 
> `source {CANN包安装路径}/nnal/asdsip/set_env.sh`

## 目录

- [Fast Fourier Transforms](#fast-fourier-transforms)
- [Helper Functions](#helper-functions)

## Fast Fourier Transforms

### torch.fft.rfftn

<div style="margin-left: 2em">

**原生文档**：[torch.fft.rfftn](https://pytorch.org/docs/2.7/generated/torch.fft.rfftn.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**：

- 支持fp32
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

**原生文档**：[torch.fft.hfft](https://pytorch.org/docs/2.7/generated/torch.fft.hfft.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fft.ihfft

<div style="margin-left: 2em">

**原生文档**：[torch.fft.ihfft](https://pytorch.org/docs/2.7/generated/torch.fft.ihfft.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fft.hfft2

<div style="margin-left: 2em">

**原生文档**：[torch.fft.hfft2](https://pytorch.org/docs/2.7/generated/torch.fft.hfft2.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fft.ihfft2

<div style="margin-left: 2em">

**原生文档**：[torch.fft.ihfft2](https://pytorch.org/docs/2.7/generated/torch.fft.ihfft2.html)

**是否支持**：否

</div>

### torch.fft.hfftn

<div style="margin-left: 2em">

**原生文档**：[torch.fft.hfftn](https://pytorch.org/docs/2.7/generated/torch.fft.hfftn.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fft.ihfftn

<div style="margin-left: 2em">

**原生文档**：[torch.fft.ihfftn](https://pytorch.org/docs/2.7/generated/torch.fft.ihfftn.html)

**是否支持**：否

</div>

## Helper Functions

### torch.fft.fftfreq

<div style="margin-left: 2em">

**原生文档**：[torch.fft.fftfreq](https://pytorch.org/docs/2.7/generated/torch.fft.fftfreq.html)

**是否支持**：否

</div>

### torch.fft.rfftfreq

<div style="margin-left: 2em">

**原生文档**：[torch.fft.rfftfreq](https://pytorch.org/docs/2.7/generated/torch.fft.rfftfreq.html)

**是否支持**：否

</div>

### torch.fft.fftshift

<div style="margin-left: 2em">

**原生文档**：[torch.fft.fftshift](https://pytorch.org/docs/2.7/generated/torch.fft.fftshift.html)

**是否支持**：否

</div>

### torch.fft.ifftshift

<div style="margin-left: 2em">

**原生文档**：[torch.fft.ifftshift](https://pytorch.org/docs/2.7/generated/torch.fft.ifftshift.html)

**是否支持**：否

**限制与说明**： <term>Ascend 950DT</term>：不支持fp64，complex64，complex128

</div>
