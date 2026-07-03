# torch.fft

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T02:16:21.476Z pushedAt=2026-06-15T03:25:49.172Z -->

> [!NOTE]  
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.
> To use the following APIs, configure the following information:
>
>- `source {CANN installation path}/nnal/atb/set_env.sh`
>- `source {CANN installation path}/nnal/asdsip/set_env.sh`

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.fft.rfftn](https://pytorch.org/docs/2.9/generated/torch.fft.rfftn.html)|Yes|Supports fp32<br>Value range: Each element must be within [-100, 100]<br>Supports 1-8 dimensions. 2D dimensions are (batch, n), 1D dimension is (n)<br>1. batch dimension: [1, 8, 16, 24, 32, 64]<br>2. n dimension restrictions (meet any one):<br>- $2^n$, n within 23<br>- Arbitrary multiplication of 2, 3, 5, 7, for example: 2\*2\*2\*5\*7\*7\*9, but the result must be within 1000000<br>- Arbitrary multiplication of prime numbers within 200, result must be within 100000|
|[torch.fft.hfft](https://pytorch.org/docs/2.9/generated/torch.fft.hfft.html)|Yes|-|
|[torch.fft.ihfft](https://pytorch.org/docs/2.9/generated/torch.fft.ihfft.html)|Yes|-|
|[torch.fft.hfft2](https://pytorch.org/docs/2.9/generated/torch.fft.hfft2.html)|Yes|-|
|[torch.fft.ihfft2](https://pytorch.org/docs/2.9/generated/torch.fft.ihfft2.html)|No|-|
|[torch.fft.hfftn](https://pytorch.org/docs/2.9/generated/torch.fft.hfftn.html)|Yes|-|
|[torch.fft.ihfftn](https://pytorch.org/docs/2.9/generated/torch.fft.ihfftn.html)|No|-|
|[torch.fft.fftfreq](https://pytorch.org/docs/2.9/generated/torch.fft.fftfreq.html)|No|-|
|[torch.fft.rfftfreq](https://pytorch.org/docs/2.9/generated/torch.fft.rfftfreq.html)|No|-|
|[torch.fft.fftshift](https://pytorch.org/docs/2.9/generated/torch.fft.fftshift.html)|No|-|
|[torch.fft.ifftshift](https://pytorch.org/docs/2.9/generated/torch.fft.ifftshift.html)|No|-|
