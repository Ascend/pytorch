# torch.distributed.optim

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.9/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.9/distributed.optim.html).

<div style="display:none;">

## &#8203;torch.distributed.optim

</div>

### <code><i>class</i></code> torch.distributed.optim.PostLocalSGDOptimizer

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.optim.PostLocalSGDOptimizer](https://pytorch.org/docs/2.9/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer)

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

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.optim.PostLocalSGDOptimizer.load_state_dict](https://pytorch.org/docs/2.9/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer.load_state_dict)

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

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.optim.PostLocalSGDOptimizer.state_dict](https://pytorch.org/docs/2.9/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer.state_dict)

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

> <font size="3">step()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.optim.PostLocalSGDOptimizer.step](https://pytorch.org/docs/2.9/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer.step)

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

</div>

### <code><i>class</i></code> torch.distributed.optim.ZeroRedundancyOptimizer

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.optim.ZeroRedundancyOptimizer](https://pytorch.org/docs/2.9/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer)

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

**Constraints & Notes**:

- The supported input type is a `torch.nn.Optimizer` object
- Does not support NPU fused optimizer objects.

> <font size="3">add_param_group()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.optim.ZeroRedundancyOptimizer.add_param_group](https://pytorch.org/docs/2.9/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.add_param_group)

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

> <font size="3">consolidate_state_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.optim.ZeroRedundancyOptimizer.consolidate_state_dict](https://pytorch.org/docs/2.9/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.consolidate_state_dict)

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

<!-- npu="910b,A3" id45 -->
> <font size="3">join_device()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.optim.ZeroRedundancyOptimizer.join_device](https://pytorch.org/docs/2.9/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.join_device)

**Support Status**:
<!-- end id45 -->

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->

<!-- npu="910b,A3" id46 -->
</div>
<!-- end id46 -->

<!-- npu="910b,A3" id47 -->
> <font size="3">join_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.optim.ZeroRedundancyOptimizer.join_hook](https://pytorch.org/docs/2.9/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.join_hook)

**Support Status**:
<!-- end id47 -->

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->

<!-- npu="910b,A3" id48 -->
</div>
<!-- end id48 -->

<!-- npu="910b,A3" id49 -->
> <font size="3">join_process_group()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.optim.ZeroRedundancyOptimizer.join_process_group](https://pytorch.org/docs/2.9/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.join_process_group)

**Support Status**:
<!-- end id49 -->

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->

<!-- npu="910b,A3" id50 -->
</div>
<!-- end id50 -->

<!-- npu="910b,A3" id51 -->
> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.optim.ZeroRedundancyOptimizer.load_state_dict](https://pytorch.org/docs/2.9/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.load_state_dict)

**Support Status**:
<!-- end id51 -->

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->

<!-- npu="910b,A3" id52 -->
</div>
<!-- end id52 -->

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.optim.ZeroRedundancyOptimizer.state_dict](https://pytorch.org/docs/2.9/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.state_dict)

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id42 -->

</div>

<!-- npu="910b,A3" id53 -->
> <font size="3">step()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.optim.ZeroRedundancyOptimizer.step](https://pytorch.org/docs/2.9/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer.step)

**Support Status**:
<!-- end id53 -->

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->

<!-- npu="910b,A3" id54 -->
</div>
<!-- end id54 -->

</div>