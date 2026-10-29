# torch.fft

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.7/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> 
> - To use the following interfaces, configure the following information:
>
>   `source {CANN package installation path}/nnal/atb/set_env.sh`
> 
>   `source {CANN package installation path}/nnal/asdsip/set_env.sh`
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.7/fft.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Fast Fourier Transforms](#fast-fourier-transforms)
- [Helper Functions](#helper-functions)

</div>

<div style="display:none;">

## &#8203;torch.fft

</div>

## Fast Fourier Transforms

### torch.fft.rfftn

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.rfftn](https://pytorch.org/docs/2.7/generated/torch.fft.rfftn.html)

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

### torch.fft.hfft

<div style="margin-left: 2em">

**Native Documentation**: [torch.fft.hfft](https://pytorch.org/docs/2.7/generated/torch.fft.hfft.html)

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

**Native Documentation**: [torch.fft.ihfft](https://pytorch.org/docs/2.7/generated/torch.fft.ihfft.html)

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

**Native Documentation**: [torch.fft.hfft2](https://pytorch.org/docs/2.7/generated/torch.fft.hfft2.html)

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

**Native Documentation**: [torch.fft.ihfft2](https://pytorch.org/docs/2.7/generated/torch.fft.ihfft2.html)

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

**Native Documentation**: [torch.fft.hfftn](https://pytorch.org/docs/2.7/generated/torch.fft.hfftn.html)

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

**Native Documentation**: [torch.fft.ihfftn](https://pytorch.org/docs/2.7/generated/torch.fft.ihfftn.html)

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

**Native Documentation**: [torch.fft.fftfreq](https://pytorch.org/docs/2.7/generated/torch.fft.fftfreq.html)

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

**Native Documentation**: [torch.fft.rfftfreq](https://pytorch.org/docs/2.7/generated/torch.fft.rfftfreq.html)

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

**Native Documentation**: [torch.fft.fftshift](https://pytorch.org/docs/2.7/generated/torch.fft.fftshift.html)

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

**Native Documentation**: [torch.fft.ifftshift](https://pytorch.org/docs/2.7/generated/torch.fft.ifftshift.html)

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
**Constraints & Notes**: <term>Ascend 950DT products</term>: do not support fp64, complex64, or complex128
<!-- end id34 -->

</div>
