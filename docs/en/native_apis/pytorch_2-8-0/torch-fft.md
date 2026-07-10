# torch.fft

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:31:21.099Z pushedAt=2026-07-09T08:44:08.284Z -->

> [!NOTE]  
> If the "Supported" column shows "Yes" and "Restrictions and Notes" shows "-", it means the API support is consistent with the native API.
>
> To use the following APIs, you need to configure the following:
>
> `source {CANN installation path}/nnal/atb/set_env.sh`
>
> `source {CANN installation path}/nnal/asdsip/set_env.sh`

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.fft.rfftn|Yes|Supports fp32<br>Value range: each element must be within [-100, 100]<br>Supports 1-8 dimensions. 2D dimensions are (batch, n), 1D dimensions are (n)<br>1. Batch dimension: [1, 8, 16, 24, 32, 64]<br>2. n dimension restrictions (meeting any one is sufficient):<br>- $2^n$, where n is within 23<br>- Arbitrary product of 2, 3, 5, 7, e.g.: 2\*2\*2\*3\*3\*5\*7\*7, but the result must be within 1000000<br>- Arbitrary product of primes within 200, result must be within 100000|
|torch.fft.hfft|Yes|-|
|torch.fft.ihfft|Yes|-|
|torch.fft.hfft2|Yes|-|
|torch.fft.ihfft2|No|-|
|torch.fft.hfftn|Yes|-|
|torch.fft.ihfftn|No|-|
|torch.fft.fftfreq|No|-|
|torch.fft.rfftfreq|No|-|
|torch.fft.fftshift|No|-|
|torch.fft.ifftshift|No|-|
