# torch.fft

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.14/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - To use the following interfaces, configure the following information:
>
>   `source {CANN installation path}/nnal/atb/set_env.sh`
>
>   `source {CANN installation path}/nnal/asdsip/set_env.sh`
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.14/fft.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Fast Fourier Transforms](#fast-fourier-transforms)
- [Helper Functions](#helper-functions)

</div>

<div style="display:none;">

## &#8203;torch.fft

</div>

## Fast Fourier Transforms

### torch.fft.fft

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.fft](https://docs.pytorch.org/docs/2.14/generated/torch.fft.fft.html#torch.fft.fft)

**Support Status**:

<!-- npu="910b" id35 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id35 -->
<!-- npu="A3" id36 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id36 -->
<!-- npu="950" id37 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id37 -->

</div>

### torch.fft.ifft

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.ifft](https://docs.pytorch.org/docs/2.14/generated/torch.fft.ifft.html#torch.fft.ifft)

**Support Status**:

<!-- npu="910b" id44 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id44 -->
<!-- npu="A3" id45 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id45 -->
<!-- npu="950" id46 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id46 -->

</div>

### torch.fft.fft2

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.fft2](https://docs.pytorch.org/docs/2.14/generated/torch.fft.fft2.html#torch.fft.fft2)

**Support Status**:

<!-- npu="910b" id38 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id38 -->
<!-- npu="A3" id39 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id39 -->
<!-- npu="950" id40 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id40 -->

</div>

### torch.fft.ifft2

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.ifft2](https://docs.pytorch.org/docs/2.14/generated/torch.fft.ifft2.html#torch.fft.ifft2)

**Support Status**:

<!-- npu="910b" id47 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id47 -->
<!-- npu="A3" id48 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id48 -->
<!-- npu="950" id49 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id49 -->

</div>

### torch.fft.fftn

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.fftn](https://docs.pytorch.org/docs/2.14/generated/torch.fft.fftn.html#torch.fft.fftn)

**Support Status**:

<!-- npu="910b" id41 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id41 -->
<!-- npu="A3" id42 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id42 -->
<!-- npu="950" id43 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id43 -->

</div>

### torch.fft.ifftn

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.ifftn](https://docs.pytorch.org/docs/2.14/generated/torch.fft.ifftn.html#torch.fft.ifftn)

**Support Status**:

<!-- npu="910b" id50 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id50 -->
<!-- npu="A3" id51 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id51 -->
<!-- npu="950" id52 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id52 -->

</div>

### torch.fft.rfft

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.rfft](https://docs.pytorch.org/docs/2.14/generated/torch.fft.rfft.html#torch.fft.rfft)

**Support Status**:

<!-- npu="910b" id56 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id56 -->
<!-- npu="A3" id57 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id57 -->
<!-- npu="950" id58 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id58 -->

</div>

### torch.fft.irfft

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.irfft](https://docs.pytorch.org/docs/2.14/generated/torch.fft.irfft.html#torch.fft.irfft)

**Support Status**:

<!-- npu="910b" id53 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id53 -->
<!-- npu="A3" id54 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id54 -->
<!-- npu="950" id55 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id55 -->

</div>

### torch.fft.rfft2

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.rfft2](https://docs.pytorch.org/docs/2.14/generated/torch.fft.rfft2.html#torch.fft.rfft2)

**Support Status**:

<!-- npu="910b" id59 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id59 -->
<!-- npu="A3" id60 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id60 -->
<!-- npu="950" id61 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id61 -->

</div>

### torch.fft.irfft2

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.irfft2](https://docs.pytorch.org/docs/2.14/generated/torch.fft.irfft2.html#torch.fft.irfft2)

**Support Status**:

<!-- npu="910b" id62 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id62 -->
<!-- npu="A3" id63 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id63 -->
<!-- npu="950" id64 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id64 -->

</div>

### torch.fft.rfftn

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.rfftn](https://pytorch.org/docs/2.14/generated/torch.fft.rfftn.html)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id3 -->

**Constraints & Notes**:

- `input` only supports fp32
- The value range: each element must be within [-100, 100]
- Supports 1-8 dimensions. For 2 dimensions, the shape is (batch, n); for 1 dimension, the shape is (n)
    1. batch dimension: [1, 8, 16, 24, 32, 64]
    2. n dimension restrictions (any one of the following must be satisfied):
        - $2^n$, where n is no more than 23
        - Any product of 2, 3, 5, and 7, for example: 2\*2\*2\*3\*3\*5\*7\*7, but the result must be within 1000000
        - Any product of prime numbers within 200, and the result must be within 100000

</div>

### torch.fft.irfftn

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.irfftn](https://docs.pytorch.org/docs/2.14/generated/torch.fft.irfftn.html#torch.fft.irfftn)

**Support Status**:

<!-- npu="910b" id65 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id65 -->
<!-- npu="A3" id66 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id66 -->
<!-- npu="950" id67 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id67 -->

</div>

### torch.fft.hfft

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.hfft](https://pytorch.org/docs/2.14/generated/torch.fft.hfft.html)

**Support Status**:

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id6 -->

</div>

### torch.fft.ihfft

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.ihfft](https://pytorch.org/docs/2.14/generated/torch.fft.ihfft.html)

**Support Status**:

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id9 -->

</div>

### torch.fft.hfft2

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.hfft2](https://pytorch.org/docs/2.14/generated/torch.fft.hfft2.html)

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id12 -->

</div>

### torch.fft.ihfft2

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.ihfft2](https://pytorch.org/docs/2.14/generated/torch.fft.ihfft2.html)

**Support Status**:

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id15 -->

</div>

### torch.fft.hfftn

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.hfftn](https://pytorch.org/docs/2.14/generated/torch.fft.hfftn.html)

**Support Status**:

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id18 -->

</div>

### torch.fft.ihfftn

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.ihfftn](https://pytorch.org/docs/2.14/generated/torch.fft.ihfftn.html)

**Support Status**:

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id21 -->

</div>

## Helper Functions

### torch.fft.fftfreq

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.fftfreq](https://pytorch.org/docs/2.14/generated/torch.fft.fftfreq.html)

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id24 -->

</div>

### torch.fft.rfftfreq

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.rfftfreq](https://pytorch.org/docs/2.14/generated/torch.fft.rfftfreq.html)

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id27 -->

</div>

### torch.fft.fftshift

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.fftshift](https://pytorch.org/docs/2.14/generated/torch.fft.fftshift.html)

**Support Status**:

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id30 -->

</div>

### torch.fft.ifftshift

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.ifftshift](https://pytorch.org/docs/2.14/generated/torch.fft.ifftshift.html)

**Support Status**:

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id33 -->

<!-- npu="950" id34 -->
**Constraints & Notes**: <term>Ascend 950DT products</term>: do not support fp64, complex64, complex128
<!-- end id34 -->

</div>