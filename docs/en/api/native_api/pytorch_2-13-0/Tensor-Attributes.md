# Tensor Attributes

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.13/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.13/tensor_attributes.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [torch.dtype](#torchdtype)
- [torch.device](#torchdevice)
- [torch.layout](#torchlayout)
- [torch.memory_format](#torchmemory_format)

</div>

<div style="display:none;">

## &#8203;Tensor Attributes

</div>

## torch.dtype

### torch.dtype

<div style="margin-left: 2em">

**Native Documentation**: [torch.dtype](https://pytorch.org/docs/2.13/tensor_attributes.html#torch.dtype)

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

## torch.device

### torch.device

<div style="margin-left: 2em">

**Native Documentation**: [torch.device](https://pytorch.org/docs/2.13/tensor_attributes.html#torch.device)

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

## torch.layout

### torch.layout

<div style="margin-left: 2em">

**Native Documentation**: [torch.layout](https://pytorch.org/docs/2.13/tensor_attributes.html#torch.layout)

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

## torch.memory_format

### <code><i>class</i></code> torch.memory_format

<div style="margin-left: 2em">

**Native Documentation**: [torch.memory_format](https://pytorch.org/docs/2.13/tensor_attributes.html#torch.memory_format)

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

**Constraints & Notes**: 

- On NPU, memory_format only supports `torch.contiguous_format` and `torch.preserve_format`; `torch.preserve_format` is used to preserve the input layout in operations such as `clone`/`empty_like`, and is not used in `torch.empty()`.
- `torch.channels_last` and `torch.channels_last_3d` are not supported on NPU.

</div>
