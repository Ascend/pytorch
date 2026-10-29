# Serialization semantics

> [!NOTE]
>
> - In the API **support status**, ✔ indicates the API runs on the corresponding hardware environment, while ✘ indicates it is not supported yet.<br>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.7/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.7/notes/serialization.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Utility functions](#utility-functions)

</div>

<div style="display:none;">

## &#8203;Serialization semantics

</div>

## Utility functions

### torch.serialization.clear_safe_globals

<div style="margin-left: 2em">

**Native Documentation**: [torch.serialization.clear_safe_globals](https://pytorch.org/docs/2.7/notes/serialization.html#torch.serialization.clear_safe_globals)

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
