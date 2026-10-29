# torch.utils.checkpoint

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.11/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
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

### torch.utils.checkpoint.get_device_states

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.checkpoint.get_device_states](https://pytorch.org/docs/2.11/checkpoint.html#torch.utils.checkpoint.get_device_states)

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

### <code><i>class</i></code> torch.utils.checkpoint.CheckpointPolicy

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.checkpoint.CheckpointPolicy](https://pytorch.org/docs/2.11/checkpoint.html#torch.utils.checkpoint.CheckpointPolicy)

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

### <code><i>class</i></code> torch.utils.checkpoint.SelectiveCheckpointContext

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.checkpoint.SelectiveCheckpointContext](https://pytorch.org/docs/2.11/checkpoint.html#torch.utils.checkpoint.SelectiveCheckpointContext)

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id12 -->

</div>

### torch.utils.checkpoint.create_selective_checkpoint_contexts

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.checkpoint.create_selective_checkpoint_contexts](https://pytorch.org/docs/2.11/checkpoint.html#torch.utils.checkpoint.create_selective_checkpoint_contexts)

**Support Status**:

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id15 -->

</div>

### torch.utils.checkpoint.set_checkpoint_debug_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.checkpoint.set_checkpoint_debug_enabled](https://pytorch.org/docs/2.11/checkpoint.html#torch.utils.checkpoint.set_checkpoint_debug_enabled)

**Support Status**:

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id18 -->

</div>

### torch.utils.checkpoint.checkpoint_sequential

<div style="margin-left: 2em">

**Native Documentation**: [torch.utils.checkpoint.checkpoint_sequential](https://pytorch.org/docs/2.11/checkpoint.html#torch.utils.checkpoint.checkpoint_sequential)

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

**Constraints & Notes**: `input` only supports fp32

</div>
