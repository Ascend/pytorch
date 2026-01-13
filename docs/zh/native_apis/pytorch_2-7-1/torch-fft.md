# torch.fft

> [!NOTE]  
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.fft.rfftn|是|支持fp32<br>支持1-2维，2维维度为(batch, n)， 1维维度为(n)<br>1. batch维度：[1, 8, 16, 24, 32, 64]<br>2. n维度：<br>- $2^n$，n为23以内<br>- 2, 3, 5, 7任意相乘，例如： 2*2*2*5*7*7*9，但结果需在1000000以内<br>- 200以内质数任意相乘，结果需在100000以内<br>取值范围[-100, 100]
|torch.fft.hfft|是|-|
|torch.fft.ihfft|是|-|
|torch.fft.hfft2|是|-|
|torch.fft.ihfft2|否|-|
|torch.fft.hfftn|是|-|
|torch.fft.ihfftn|否|-|
|torch.fft.fftfreq|否|-|
|torch.fft.rfftfreq|否|-|
|torch.fft.fftshift|否|-|
|torch.fft.ifftshift|否|-|


