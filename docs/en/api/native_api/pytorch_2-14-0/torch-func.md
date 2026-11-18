# torch.func

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.14/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.14/func.api.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Function Transforms](#function-transforms)
- [Utilities for working with torch.nn.Modules](#utilities-for-working-with-torchnnmodules)

- [Debug utilities](#debug-utilities)

</div>

<div style="display:none;">

## &#8203;torch.func

</div>

## Function Transforms

<!-- npu="910b,A3" id38 -->
### torch.func.grad

<div style="margin-left: 2em">

**Native Documentation**: [torch.func.grad](https://pytorch.org/docs/2.14/generated/torch.func.grad.html)

**Support Status**:
<!-- end id38 -->

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->

<!-- npu="910b,A3" id39 -->
</div>
<!-- end id39 -->

### torch.func.grad_and_value

<div style="margin-left: 2em">

**Native Documentation**: [torch.func.grad_and_value](https://pytorch.org/docs/2.14/generated/torch.func.grad_and_value.html)

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

<!-- npu="910b,A3" id36 -->
### torch.func.linearize

<div style="margin-left: 2em">

**Native Documentation**: [torch.func.linearize](https://pytorch.org/docs/2.14/generated/torch.func.linearize.html)

**Support Status**:
<!-- end id36 -->

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->

<!-- npu="910b,A3" id37 -->
</div>
<!-- end id37 -->

## Utilities for working with torch.nn.Modules

### torch.func.functional_call

<div style="margin-left: 2em">

**Native Documentation**: [torch.func.functional_call](https://pytorch.org/docs/2.14/generated/torch.func.functional_call.html)

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

<!-- npu="910b,A3" id40 -->
### torch.func.stack_module_state

<div style="margin-left: 2em">

**Native Documentation**: [torch.func.stack_module_state](https://pytorch.org/docs/2.14/generated/torch.func.stack_module_state.html)

**Support Status**:
<!-- end id40 -->

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->

<!-- npu="910b,A3" id41 -->
</div>
<!-- end id41 -->

## Debug utilities

### torch.func.debug\_unwrap

<div style="margin-left: 2em">

**Native Documentation**: [torch.func.debug\_unwrap](https://docs.pytorch.org/docs/2.14/generated/torch.func.debug_unwrap.html#torch.func.debug_unwrap)

**Support Status**:

<!-- npu="910b" id42 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id42 -->
<!-- npu="A3" id43 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id43 -->
<!-- npu="950" id44 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id44 -->

</div>
