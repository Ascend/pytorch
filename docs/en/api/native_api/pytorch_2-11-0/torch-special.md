# torch.special

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.11/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.11/special.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Functions](#functions)

</div>

<div style="display:none;">

## &#8203;torch.special

</div>

## Functions

### torch.special.airy_ai

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.airy_ai](https://docs.pytorch.org/docs/2.11/special.html#torch.special.airy_ai)

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

</div>

### torch.special.bessel_j0

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.bessel_j0](https://docs.pytorch.org/docs/2.11/special.html#torch.special.bessel_j0)

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id36 -->

</div>

### torch.special.bessel_j1

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.bessel_j1](https://docs.pytorch.org/docs/2.11/special.html#torch.special.bessel_j1)

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id39 -->

</div>

### torch.special.digamma

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.digamma](https://docs.pytorch.org/docs/2.11/special.html#torch.special.digamma)

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id42 -->

</div>

### torch.special.entr

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.entr](https://docs.pytorch.org/docs/2.11/special.html#torch.special.entr)

**Support Status**:

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id45 -->

</div>

### torch.special.erf

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.erf](https://pytorch.org/docs/2.11/special.html#torch.special.erf)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id3 -->

**Constraints & Notes**: `input` only supports fp16, fp32, int64, bool.

</div>

### torch.special.erfc

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.erfc](https://pytorch.org/docs/2.11/special.html#torch.special.erfc)

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

**Constraints & Notes**: `input` only supports fp16, fp32, int64, bool.

</div>

### torch.special.erfcx

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.erfcx](https://docs.pytorch.org/docs/2.11/special.html#torch.special.erfcx)

**Support Status**:

<!-- npu="910b" id46 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id48 -->

</div>

### torch.special.erfinv

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.erfinv](https://pytorch.org/docs/2.11/special.html#torch.special.erfinv)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool.

</div>

### torch.special.exp2

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.exp2](https://pytorch.org/docs/2.11/special.html#torch.special.exp2)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool.

</div>

### torch.special.expit

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.expit](https://pytorch.org/docs/2.11/special.html#torch.special.expit)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128.

</div>

### torch.special.gammainc

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.gammainc](https://docs.pytorch.org/docs/2.11/special.html#torch.special.gammainc)

**Support Status**:

<!-- npu="910b" id49 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id51 -->

</div>

### torch.special.gammaincc

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.gammaincc](https://docs.pytorch.org/docs/2.11/special.html#torch.special.gammaincc)

**Support Status**:

<!-- npu="910b" id52 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id54 -->

</div>

### torch.special.gammaln

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.gammaln](https://docs.pytorch.org/docs/2.11/special.html#torch.special.gammaln)

**Support Status**:

<!-- npu="910b" id55 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id57 -->

</div>

### torch.special.i0

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.i0](https://docs.pytorch.org/docs/2.11/special.html#torch.special.i0)

**Support Status**:

<!-- npu="910b" id58 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id60 -->

</div>

### torch.special.i0e

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.i0e](https://docs.pytorch.org/docs/2.11/special.html#torch.special.i0e)

**Support Status**:

<!-- npu="910b" id61 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id63 -->

</div>

### torch.special.i1

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.i1](https://docs.pytorch.org/docs/2.11/special.html#torch.special.i1)

**Support Status**:

<!-- npu="910b" id64 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id66 -->

</div>

### torch.special.i1e

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.i1e](https://docs.pytorch.org/docs/2.11/special.html#torch.special.i1e)

**Support Status**:

<!-- npu="910b" id67 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id69 -->

</div>

### torch.special.log1p

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.log1p](https://pytorch.org/docs/2.11/special.html#torch.special.log1p)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool.

</div>

### torch.special.log_ndtr

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.log_ndtr](https://docs.pytorch.org/docs/2.11/special.html#torch.special.log_ndtr)

**Support Status**:

<!-- npu="910b" id70 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id72 -->

</div>

### torch.special.log\_softmax

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.log\_softmax](https://docs.pytorch.org/docs/2.11/special.html#torch.special.log_softmax)

**Support Status**:

<!-- npu="910b" id94 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id96 -->

</div>

### torch.special.logsumexp

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.logsumexp](https://pytorch.org/docs/2.11/special.html#torch.special.logsumexp)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool.

</div>

### torch.special.multigammaln

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.multigammaln](https://pytorch.org/docs/2.11/special.html#torch.special.multigammaln)

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

**Constraints & Notes**:

- `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool
- The parameter `p` must be a positive integer, and each element of `input` must satisfy `input > (p - 1) / 2`; otherwise, the result is `nan` or `inf`

</div>

### torch.special.ndtr

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.ndtr](https://pytorch.org/docs/2.11/special.html#torch.special.ndtr)

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

### torch.special.ndtri

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.ndtri](https://docs.pytorch.org/docs/2.11/special.html#torch.special.ndtri)

**Support Status**:

<!-- npu="910b" id73 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id75 -->

</div>

### torch.special.polygamma

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.polygamma](https://docs.pytorch.org/docs/2.11/special.html#torch.special.polygamma)

**Support Status**:

<!-- npu="910b" id76 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id78 -->

</div>

### torch.special.scaled_modified_bessel_k0

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.scaled_modified_bessel_k0](https://docs.pytorch.org/docs/2.11/special.html#torch.special.scaled_modified_bessel_k0)

**Support Status**:

<!-- npu="910b" id79 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id81 -->

</div>

### torch.special.scaled_modified_bessel_k1

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.scaled_modified_bessel_k1](https://docs.pytorch.org/docs/2.11/special.html#torch.special.scaled_modified_bessel_k1)

**Support Status**:

<!-- npu="910b" id82 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id84 -->

</div>

### torch.special.sinc

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.sinc](https://docs.pytorch.org/docs/2.11/special.html#torch.special.sinc)

**Support Status**:

<!-- npu="910b" id97 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id99 -->

</div>

### torch.special.softmax

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.softmax](https://docs.pytorch.org/docs/2.11/special.html#torch.special.softmax)

**Support Status**:

<!-- npu="910b" id100 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id102 -->

</div>

### torch.special.spherical_bessel_j0

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.spherical_bessel_j0](https://docs.pytorch.org/docs/2.11/special.html#torch.special.spherical_bessel_j0)

**Support Status**:

<!-- npu="910b" id85 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id87 -->

</div>

### torch.special.xlog1py

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.xlog1py](https://docs.pytorch.org/docs/2.11/special.html#torch.special.xlog1py)

**Support Status**:

<!-- npu="910b" id88 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id90 -->

</div>

### torch.special.xlogy

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.xlogy](https://pytorch.org/docs/2.11/special.html#torch.special.xlogy)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool.

</div>

### torch.special.zeta

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.zeta](https://docs.pytorch.org/docs/2.11/special.html#torch.special.zeta)

**Support Status**:

<!-- npu="910b" id91 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id93 -->

</div>
