# torch.utils.deterministic

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.11/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.11/deterministic.html).

<div style="display:none;">

## &#8203;torch.utils.deterministic

</div>

### torch.utils.deterministic.fill_uninitialized_memory

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.deterministic.fill_uninitialized_memory](https://pytorch.org/docs/2.11/deterministic.html#torch.utils.deterministic.fill_uninitialized_memory)

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

**Constraints & Notes**: `torch.utils.deterministic.fill_uninitialized_memory` defaults to True. After enabling determinism with `torch.use_deterministic_algorithms`, the PyTorch behavior is to fill uninitialized memory, while TorchNPU does not fill it by default. To make the filling take effect, you need to manually set `torch.utils.deterministic.fill_uninitialized_memory` to True

</div>
