# torch.distributed.algorithms.join

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.10/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.10/distributed.algorithms.join.html).

<div style="display:none;">

## &#8203;torch.distributed.algorithms.join

</div>

### <code><i>class</i></code> torch.distributed.algorithms.Join

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.Join](https://pytorch.org/docs/2.10/distributed.algorithms.join.html#torch.distributed.algorithms.Join)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id3 -->

> <font size="3">notify_join_context()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.Join.notify_join_context](https://pytorch.org/docs/2.10/distributed.algorithms.join.html#torch.distributed.algorithms.Join.notify_join_context)

**Support Status**:

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id6 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.algorithms.Joinable

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.Joinable](https://pytorch.org/docs/2.10/distributed.algorithms.join.html#torch.distributed.algorithms.Joinable)

**Support Status**:

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id9 -->

> <font size="3">join_device()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.Joinable.join_device](https://pytorch.org/docs/2.10/distributed.algorithms.join.html#torch.distributed.algorithms.Joinable.join_device)

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

> <font size="3">join_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.Joinable.join_hook](https://pytorch.org/docs/2.10/distributed.algorithms.join.html#torch.distributed.algorithms.Joinable.join_hook)

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

> <font size="3">join_process_group()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.Joinable.join_process_group](https://pytorch.org/docs/2.10/distributed.algorithms.join.html#torch.distributed.algorithms.Joinable.join_process_group)

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

</div>

### <code><i>class</i></code> torch.distributed.algorithms.JoinHook

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.JoinHook](https://pytorch.org/docs/2.10/distributed.algorithms.join.html#torch.distributed.algorithms.JoinHook)

**Support Status**:

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id21 -->

> <font size="3">main_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.JoinHook.main_hook](https://pytorch.org/docs/2.10/distributed.algorithms.join.html#torch.distributed.algorithms.JoinHook.main_hook)

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id24 -->

</div>

> <font size="3">post_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.algorithms.JoinHook.post_hook](https://pytorch.org/docs/2.10/distributed.algorithms.join.html#torch.distributed.algorithms.JoinHook.post_hook)

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id27 -->

</div>

</div>
