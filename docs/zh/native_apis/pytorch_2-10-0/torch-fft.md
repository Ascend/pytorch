# torch.fft

> [!NOTE]  
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。
> 
> 使用以下接口，需要配置以下信息：
>
> `source {CANN包安装路径}/nnal/atb/set_env.sh`
> 
> `source {CANN包安装路径}/nnal/asdsip/set_env.sh`

|API名称|是否支持|限制与说明|
|--|--|--|
|[torch.fft.rfftn](https://pytorch.org/docs/2.10/generated/torch.fft.rfftn.html)|是|支持fp32<br>数值范围：每个元素必须在[-100, 100]内<br>支持1-8维。2维维度为(batch, n)，1维维度为(n)<br>1. batch维度：[1, 8, 16, 24, 32, 64]<br>2. n维度限制（满足任一即可）：<br>- $2^n$，n为23以内<br>- 2, 3, 5, 7任意相乘，例如：2\*2\*2\*3\*3\*5\*7\*7，但结果需在1000000以内<br>- 200以内质数任意相乘，结果需在100000以内|
|[torch.fft.hfft](https://pytorch.org/docs/2.10/generated/torch.fft.hfft.html)|是|-|
|[torch.fft.ihfft](https://pytorch.org/docs/2.10/generated/torch.fft.ihfft.html)|是|-|
|[torch.fft.hfft2](https://pytorch.org/docs/2.10/generated/torch.fft.hfft2.html)|是|-|
|[torch.fft.ihfft2](https://pytorch.org/docs/2.10/generated/torch.fft.ihfft2.html)|否|-|
|[torch.fft.hfftn](https://pytorch.org/docs/2.10/generated/torch.fft.hfftn.html)|是|-|
|[torch.fft.ihfftn](https://pytorch.org/docs/2.10/generated/torch.fft.ihfftn.html)|否|-|
|[torch.fft.fftfreq](https://pytorch.org/docs/2.10/generated/torch.fft.fftfreq.html)|否|-|
|[torch.fft.rfftfreq](https://pytorch.org/docs/2.10/generated/torch.fft.rfftfreq.html)|否|-|
|[torch.fft.fftshift](https://pytorch.org/docs/2.10/generated/torch.fft.fftshift.html)|否|-|
|[torch.fft.ifftshift](https://pytorch.org/docs/2.10/generated/torch.fft.ifftshift.html)|否|-|
