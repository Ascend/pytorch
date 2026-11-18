# torch.utils.checkpoint

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.11/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.11/checkpoint.html).

<div style="display:none;">

## &#8203;torch.utils.checkpoint

</div>

### torch.utils.checkpoint.checkpoint

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.checkpoint.checkpoint](https://pytorch.org/docs/2.11/checkpoint.html#torch.utils.checkpoint.checkpoint)

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

### torch.utils.checkpoint.checkpoint_sequential

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.checkpoint.checkpoint_sequential](https://pytorch.org/docs/2.11/checkpoint.html#torch.utils.checkpoint.checkpoint_sequential)

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

**Constraints & Notes**: `input` only supports fp32.

</div>

<!-- npu="910b,A3" id23 -->
### torch.utils.checkpoint.set_checkpoint_debug_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.checkpoint.set_checkpoint_debug_enabled](https://pytorch.org/docs/2.11/checkpoint.html#torch.utils.checkpoint.set_checkpoint_debug_enabled)

**Support Status**:
<!-- end id23 -->

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->

<!-- npu="910b,A3" id24 -->
</div>
<!-- end id24 -->

### <code><i>class</i></code> torch.utils.checkpoint.CheckpointPolicy

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.checkpoint.CheckpointPolicy](https://pytorch.org/docs/2.11/checkpoint.html#torch.utils.checkpoint.CheckpointPolicy)

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

<!-- npu="910b,A3" id19 -->
### <code><i>class</i></code> torch.utils.checkpoint.SelectiveCheckpointContext

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.checkpoint.SelectiveCheckpointContext](https://pytorch.org/docs/2.11/checkpoint.html#torch.utils.checkpoint.SelectiveCheckpointContext)

**Support Status**:
<!-- end id19 -->

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->

<!-- npu="910b,A3" id20 -->
</div>
<!-- end id20 -->

<!-- npu="910b,A3" id21 -->
### torch.utils.checkpoint.create_selective_checkpoint_contexts

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.checkpoint.create_selective_checkpoint_contexts](https://pytorch.org/docs/2.11/checkpoint.html#torch.utils.checkpoint.create_selective_checkpoint_contexts)

**Support Status**:
<!-- end id21 -->

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->

<!-- npu="910b,A3" id22 -->
</div>
<!-- end id22 -->
