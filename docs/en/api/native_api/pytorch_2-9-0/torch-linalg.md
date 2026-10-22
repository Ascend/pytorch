# torch.linalg

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.9/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.9/linalg.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Matrix Properties](#matrix-properties)
- [Decompositions](#decompositions)
- [Solvers](#solvers)
- [Matrix Products](#matrix-products)
- [Experimental Functions](#experimental-functions)

</div>

<div style="display:none;">

## &#8203;torch.linalg

</div>

## Matrix Properties

### torch.linalg.norm

<div style="margin-left: 2em">

**Native Documentation**: [torch.linalg.norm](https://pytorch.org/docs/2.9/generated/torch.linalg.norm.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### torch.linalg.vector_norm

<div style="margin-left: 2em">

**Native Documentation**: [torch.linalg.vector_norm](https://pytorch.org/docs/2.9/generated/torch.linalg.vector_norm.html)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

## Decompositions

### torch.linalg.cholesky

<div style="margin-left: 2em">

**Native Documentation**: [torch.linalg.cholesky](https://pytorch.org/docs/2.9/generated/torch.linalg.cholesky.html)

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

**Constraints & Notes**: `A` only supports fp32

</div>

### torch.linalg.qr

<div style="margin-left: 2em">

**Native Documentation**: [torch.linalg.qr](https://pytorch.org/docs/2.9/generated/torch.linalg.qr.html)

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

**Constraints & Notes**: `A` only supports fp32, fp64, complex64, complex128

</div>

## Solvers

### torch.linalg.lstsq

<div style="margin-left: 2em">

**Native Documentation**: [torch.linalg.lstsq](https://pytorch.org/docs/2.9/generated/torch.linalg.lstsq.html)

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

**Constraints & Notes**: May fall back to CPU execution

</div>

### torch.linalg.solve_triangular

<div style="margin-left: 2em">

**Native Documentation**: [torch.linalg.solve_triangular](https://pytorch.org/docs/2.9/generated/torch.linalg.solve_triangular.html)

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

**Constraints & Notes**: `A` and `B` support only fp32, fp64, complex64, and complex128

</div>

## Matrix Products

### torch.linalg.matmul

<div style="margin-left: 2em">

**Native Documentation**: [torch.linalg.matmul](https://pytorch.org/docs/2.9/generated/torch.linalg.matmul.html)

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

**Constraints & Notes**:

- `input` only supports fp16, fp32
- The input supports a maximum of 6 dimensions.

</div>

## Experimental Functions

### torch.linalg.ldl_factor

<div style="margin-left: 2em">

**Native Documentation**: [torch.linalg.ldl_factor](https://pytorch.org/docs/2.9/generated/torch.linalg.ldl_factor.html)

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
