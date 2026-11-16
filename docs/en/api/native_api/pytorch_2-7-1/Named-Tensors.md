# Named Tensors

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.7/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.7/named_tensor.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Named tensor API reference](#named-tensor-api-reference)

</div>

<div style="display:none;">

## &#8203;Named Tensors

</div>

## Named tensor API reference

<!-- npu="910b,A3,950" id25 -->
### <code><i>class</i></code> torch.Tensor
<!-- end id25 -->

<div style="margin-left: 2em">

<!-- npu="910b,A3" id26 -->
**Native Documentation**: [torch.Tensor](https://docs.pytorch.org/docs/2.7/named_tensor.html#named-tensors)

**Support Status**:
<!-- end id26 -->

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->

<!-- npu="910b,A3" id27 -->
**Constraints & Notes**: `self` only supports fp32
<!-- end id27 -->

<!-- npu="910b,A3" id28 -->
> <font size="3">names</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Tensor.names](https://pytorch.org/docs/2.7/named_tensor.html#torch.Tensor.names)

**Support Status**:
<!-- end id28 -->

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->

<!-- npu="910b,A3" id29 -->
**Constraints & Notes**: `self` only supports fp32

</div>
<!-- end id29 -->

<!-- npu="910b,A3" id30 -->
> <font size="3">rename()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Tensor.rename](https://pytorch.org/docs/2.7/named_tensor.html#torch.Tensor.rename)

**Support Status**:
<!-- end id30 -->

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->

<!-- npu="910b,A3" id31 -->
**Constraints & Notes**: `self` only supports fp32

</div>
<!-- end id31 -->

<!-- npu="910b,A3" id32 -->
> <font size="3">rename_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Tensor.rename_](https://pytorch.org/docs/2.7/named_tensor.html#torch.Tensor.rename_)

**Support Status**:
<!-- end id32 -->

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->

<!-- npu="910b,A3" id33 -->
**Constraints & Notes**: `self` only supports fp32

</div>
<!-- end id33 -->

<!-- npu="910b,A3" id34 -->
> <font size="3">align_to</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Tensor.align_to](https://pytorch.org/docs/2.7/named_tensor.html#torch.Tensor.align_to)

**Support Status**:
<!-- end id34 -->

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->

<!-- npu="910b,A3" id35 -->
**Constraints & Notes**: `self` only supports fp32

</div>
<!-- end id35 -->

<!-- npu="910b,A3" id36 -->
> <font size="3">refine_names()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Tensor.refine_names](https://pytorch.org/docs/2.7/named_tensor.html#torch.Tensor.refine_names)

**Support Status**:
<!-- end id36 -->

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->

<!-- npu="910b,A3" id37 -->
**Constraints & Notes**: `self` only supports fp32

</div>
<!-- end id37 -->

<!-- npu="910b,A3" id38 -->
> <font size="3">align_as()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Tensor.align_as](https://pytorch.org/docs/2.7/named_tensor.html#torch.Tensor.align_as)

**Support Status**:
<!-- end id38 -->

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->

<!-- npu="910b,A3" id39 -->
**Constraints & Notes**: `self` only supports fp32

</div>
<!-- end id39 -->

> <font size="3">flatten()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.Tensor.flatten](https://docs.pytorch.org/docs/2.7/named_tensor.html#manipulating-dimensions)

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

**Constraints & Notes**: `self` only supports fp32

</div>

</div>