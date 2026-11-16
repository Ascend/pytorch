# torch.compiler

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.14/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.14/torch.compiler_api.html).

<div style="display:none;">

## &#8203;torch.compiler

</div>

### torch.compiler.compile

<div style="margin-left: 2em">

**Native Documentation**: [torch.compiler.compile](https://pytorch.org/docs/2.14/generated/torch.compiler.compile.html)

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

### torch.compiler.reset

<div style="margin-left: 2em">

**Native Documentation**: [torch.compiler.reset](https://pytorch.org/docs/2.14/generated/torch.compiler.reset.html)

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

### torch.compiler.allow_in_graph

<div style="margin-left: 2em">

**Native Documentation**: [torch.compiler.allow_in_graph](https://pytorch.org/docs/2.14/generated/torch.compiler.allow_in_graph.html)

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

### torch.compiler.assume_constant_result

<div style="margin-left: 2em">

**Native Documentation**: [torch.compiler.assume_constant_result](https://pytorch.org/docs/2.14/generated/torch.compiler.assume_constant_result.html)

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

### torch.compiler.list_backends

<div style="margin-left: 2em">

**Native Documentation**: [torch.compiler.list_backends](https://pytorch.org/docs/2.14/generated/torch.compiler.list_backends.html)

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

### torch.compiler.disable

<div style="margin-left: 2em">

**Native Documentation**: [torch.compiler.disable](https://pytorch.org/docs/2.14/generated/torch.compiler.disable.html)

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

### torch.compiler.set_stance

<div style="margin-left: 2em">

**Native Documentation**: [torch.compiler.set_stance](https://pytorch.org/docs/2.14/generated/torch.compiler.set_stance.html)

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

### torch.compiler.cudagraph_mark_step_begin

<div style="margin-left: 2em">

**Native Documentation**: [torch.compiler.cudagraph_mark_step_begin](https://docs.pytorch.org/docs/2.14/generated/torch.compiler.cudagraph_mark_step_begin.html#torch.compiler.cudagraph_mark_step_begin)

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

### torch.compiler.is_compiling

<div style="margin-left: 2em">

**Native Documentation**: [torch.compiler.is_compiling](https://docs.pytorch.org/docs/2.14/generated/torch.compiler.is_compiling.html#torch.compiler.is_compiling)

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

### torch.compiler.is_dynamo_compiling

<div style="margin-left: 2em">

**Native Documentation**: [torch.compiler.is_dynamo_compiling](https://docs.pytorch.org/docs/2.14/generated/torch.compiler.is_dynamo_compiling.html#torch.compiler.is_dynamo_compiling)

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