# torch.distributed.tensor.parallel

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.9/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.9/distributed.tensor.parallel.html).

<div style="display:none;">

## &#8203;torch.distributed.tensor.parallel

</div>

### torch.distributed.tensor.parallel.parallelize_module

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.tensor.parallel.parallelize_module](https://pytorch.org/docs/2.9/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.parallelize_module)

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

### <code><i>class</i></code> torch.distributed.tensor.parallel.ColwiseParallel

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.tensor.parallel.ColwiseParallel](https://pytorch.org/docs/2.9/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.ColwiseParallel)

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

**Constraints & Notes**: `input` only supports bf16, fp16, fp32

</div>

### <code><i>class</i></code> torch.distributed.tensor.parallel.RowwiseParallel

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.tensor.parallel.RowwiseParallel](https://pytorch.org/docs/2.9/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.RowwiseParallel)

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

### <code><i>class</i></code> torch.distributed.tensor.parallel.PrepareModuleInput

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.tensor.parallel.PrepareModuleInput](https://pytorch.org/docs/2.9/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.PrepareModuleInput)

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id12 -->

</div>

### <code><i>class</i></code> torch.distributed.tensor.parallel.PrepareModuleOutput

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.tensor.parallel.PrepareModuleOutput](https://pytorch.org/docs/2.9/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.PrepareModuleOutput)

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

### torch.distributed.tensor.parallel.loss_parallel

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributed.tensor.parallel.loss_parallel](https://pytorch.org/docs/2.9/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.loss_parallel)

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

**Constraints & Notes**:

- `input` only supports bf16, fp16, fp32, int64

<!-- npu="950,A3,910b" id19 -->
- For <term>Ascend 950DT products</term>, when the logits values are particularly large, precision may differ from that of <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>
<!-- end id19 -->

</div>
