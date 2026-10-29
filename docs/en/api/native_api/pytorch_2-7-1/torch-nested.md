# torch.nested

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.7/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.7/nested.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Detailed Docs for Construction and Conversion Functions](#detailed-docs-for-construction-and-conversion-functions)

</div>

<div style="display:none;">

## &#8203;torch.nested

</div>

## Detailed Docs for Construction and Conversion Functions

### torch.nested.nested_tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.nested.nested_tensor](https://pytorch.org/docs/2.7/nested.html#torch.nested.nested_tensor)

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

**Constraints & Notes**: Currently, nested tensors only support creation and do not support other operations.

</div>

### torch.nested.as_nested_tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.nested.as_nested_tensor](https://pytorch.org/docs/2.7/nested.html#torch.nested.as_nested_tensor)

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

**Constraints & Notes**: Currently, nested tensors only support creation and do not support other operations.

</div>

### torch.nested.to_padded_tensor

<div style="margin-left: 2em">

**Native Documentation**: [torch.nested.to_padded_tensor](https://pytorch.org/docs/2.7/nested.html#torch.nested.to_padded_tensor)

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

### torch.nested.nested_tensor_from_jagged

<div style="margin-left: 2em">

**Native Documentation**: [torch.nested.nested_tensor_from_jagged](https://pytorch.org/docs/2.7/nested.html#torch.nested.nested_tensor_from_jagged)

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

### torch.nested.masked_select

<div style="margin-left: 2em">

**Native Documentation**: [torch.nested.masked_select](https://pytorch.org/docs/2.7/nested.html#torch.nested.masked_select)

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

### torch.nested.narrow

<div style="margin-left: 2em">

**Native Documentation**: [torch.nested.narrow](https://pytorch.org/docs/2.7/nested.html#torch.nested.narrow)

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

<!-- npu="950" id19 -->
**Constraints & Notes**: <term>Ascend 950DT products</term>: do not support complex64 or complex128
<!-- end id19 -->

</div>
