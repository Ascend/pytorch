# torch.overrides

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.11/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.11/torch.overrides.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Functions](#functions)

</div>

<div style="display:none;">

## &#8203;torch.overrides

</div>

## Functions

### torch.overrides.get_ignored_functions

<div style="margin-left: 2em">

**Native Documentation**: [torch.overrides.get_ignored_functions](https://pytorch.org/docs/2.11/torch.overrides.html#torch.overrides.get_ignored_functions)

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

</div>

### torch.overrides.get_overridable_functions

<div style="margin-left: 2em">

**Native Documentation**: [torch.overrides.get_overridable_functions](https://pytorch.org/docs/2.11/torch.overrides.html#torch.overrides.get_overridable_functions)

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

### torch.overrides.resolve_name

<div style="margin-left: 2em">

**Native Documentation**: [torch.overrides.resolve_name](https://pytorch.org/docs/2.11/torch.overrides.html#torch.overrides.resolve_name)

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

### torch.overrides.get_testing_overrides

<div style="margin-left: 2em">

**Native Documentation**: [torch.overrides.get_testing_overrides](https://pytorch.org/docs/2.11/torch.overrides.html#torch.overrides.get_testing_overrides)

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

### torch.overrides.has_torch_function

<div style="margin-left: 2em">

**Native Documentation**: [torch.overrides.has_torch_function](https://pytorch.org/docs/2.11/torch.overrides.html#torch.overrides.has_torch_function)

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

### torch.overrides.is_tensor_method_or_property

<div style="margin-left: 2em">

**Native Documentation**: [torch.overrides.is_tensor_method_or_property](https://pytorch.org/docs/2.11/torch.overrides.html#torch.overrides.is_tensor_method_or_property)

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

### torch.overrides.wrap_torch_function

<div style="margin-left: 2em">

**Native Documentation**: [torch.overrides.wrap_torch_function](https://pytorch.org/docs/2.11/torch.overrides.html#torch.overrides.wrap_torch_function)

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

### torch.overrides.handle_torch_function

<div style="margin-left: 2em">

**Native Documentation**: [torch.overrides.handle_torch_function](https://pytorch.org/docs/2.11/torch.overrides.html#torch.overrides.handle_torch_function)

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

### torch.overrides.is_tensor_like

<div style="margin-left: 2em">

**Native Documentation**: [torch.overrides.is_tensor_like](https://pytorch.org/docs/2.11/torch.overrides.html#torch.overrides.is_tensor_like)

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
