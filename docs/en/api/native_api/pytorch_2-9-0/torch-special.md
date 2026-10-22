# torch.special

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.9/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.9/special.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Functions](#functions)

</div>

<div style="display:none;">

## &#8203;torch.special

</div>

## Functions

### torch.special.erf

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.erf](https://pytorch.org/docs/2.9/special.html#torch.special.erf)

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

**Constraints & Notes**: `input` only supports fp16, fp32, int64, bool

</div>

### torch.special.erfc

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.erfc](https://pytorch.org/docs/2.9/special.html#torch.special.erfc)

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

**Constraints & Notes**: `input` only supports fp16, fp32, int64, bool

</div>

### torch.special.erfinv

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.erfinv](https://pytorch.org/docs/2.9/special.html#torch.special.erfinv)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.special.exp2

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.exp2](https://pytorch.org/docs/2.9/special.html#torch.special.exp2)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.special.expit

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.expit](https://pytorch.org/docs/2.9/special.html#torch.special.expit)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool, complex64, complex128

</div>

### torch.special.ndtr

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.ndtr](https://pytorch.org/docs/2.9/special.html#torch.special.ndtr)

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

### torch.special.xlogy

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.xlogy](https://pytorch.org/docs/2.9/special.html#torch.special.xlogy)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.special.log1p

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.log1p](https://pytorch.org/docs/2.9/special.html#torch.special.log1p)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.special.logsumexp

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.logsumexp](https://pytorch.org/docs/2.9/special.html#torch.special.logsumexp)

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

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

### torch.special.multigammaln

<div style="margin-left: 2em">

**Native Documentation**: [torch.special.multigammaln](https://pytorch.org/docs/2.9/special.html#torch.special.multigammaln)

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
- The parameter `p` must be a positive integer, and the elements of `input` must satisfy `input > (p - 1) / 2`; otherwise, the result is `nan` or `inf`

</div>
