# torch.amp

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.9/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.9/amp.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Autocasting](#autocasting)
- [Gradient Scaling](#gradient-scaling)

</div>

<div style="display:none;">

## &#8203;torch.amp

</div>

## Autocasting

### torch.cuda.amp.autocast

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.amp.autocast](https://pytorch.org/docs/2.9/amp.html#torch.cuda.amp.autocast)

**Version Notes**: This API has been deprecated; use `torch.amp.autocast("cuda", ...)`.

**NPU-Adapted Name**: torch_npu.npu.amp.autocast

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

### torch.amp.custom_bwd

<div style="margin-left: 2em">

**Native Documentation**: [torch.amp.custom_bwd](https://pytorch.org/docs/2.9/amp.html#torch.amp.custom_bwd)

**NPU Form Name**: torch_npu.npu.amp.custom_bwd

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

### torch.cuda.amp.custom_bwd

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.amp.custom_bwd](https://pytorch.org/docs/2.9/amp.html#torch.cuda.amp.custom_bwd)

**NPU-Adapted Name**: torch_npu.npu.amp.custom_bwd

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

### torch.cpu.amp.autocast

<div style="margin-left: 2em">

**Native Documentation**: [torch.cpu.amp.autocast](https://pytorch.org/docs/2.9/amp.html#torch.cpu.amp.autocast)

**Version Notes**: This API has been deprecated; use `torch.amp.autocast("cpu", ...)`.

**NPU-Adapted Name**: torch.cpu.amp.autocast

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

### torch.autocast

<div style="margin-left: 2em">

**Native Documentation**: [torch.autocast](https://pytorch.org/docs/2.9/amp.html#torch.autocast)

**NPU-Adapted Name**: torch.autocast

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

### torch.cuda.amp.custom_fwd

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.amp.custom_fwd](https://pytorch.org/docs/2.9/amp.html#torch.cuda.amp.custom_fwd)

**NPU-Adapted Name**: torch_npu.npu.amp.custom_fwd

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

## Gradient Scaling

### <code><i>class</i></code> torch.cuda.amp.GradScaler

<div style="margin-left: 2em">

**Native Documentation**: [torch.cuda.amp.GradScaler](https://pytorch.org/docs/2.9/amp.html#torch.cuda.amp.GradScaler)

**Version Notes**: This API has been deprecated; use `torch.amp.GradScaler("cuda", ...)`.

**NPU-Adapted Name**: torch_npu.npu.amp.GradScaler

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

### <code><i>class</i></code> torch.cpu.amp.GradScaler

<div style="margin-left: 2em">

**Native Documentation**: [torch.cpu.amp.GradScaler](https://pytorch.org/docs/2.9/amp.html#torch.cpu.amp.GradScaler)

**NPU Form Name**: torch.cpu.amp.GradScaler

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
