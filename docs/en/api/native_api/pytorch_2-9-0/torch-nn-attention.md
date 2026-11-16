# torch.nn.attention

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.9/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.9/nn.attention.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Submodules](#submodules)

</div>

<div style="display:none;">

## &#8203;torch.nn.attention

</div>

## Submodules

### torch.nn.attention.flex_attention

#### BlockMask

##### <code><i>class</i></code> torch.nn.attention.flex_attention.BlockMask

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.attention.flex_attention.BlockMask](https://docs.pytorch.org/docs/2.9/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask)

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

> <font size="3">kv_indices</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.attention.flex_attention.BlockMask.kv_indices](https://docs.pytorch.org/docs/2.9/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.kv_indices)

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

> <font size="3"><code><i>property</i></code> shape</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.attention.flex_attention.BlockMask.shape](https://docs.pytorch.org/docs/2.9/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.shape)

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

</div>

### <code><i>class</i></code> torch.nn.attention.flex\_attention.BlockMask

<div style="margin-left: 2em">

> <font size="3">from_kv_blocks()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.attention.flex\_attention.BlockMask.from\_kv\_blocks](https://docs.pytorch.org/docs/2.9/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.from_kv_blocks)

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

</div>

### torch.nn.attention.flex\_attention.noop\_mask

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.attention.flex\_attention.noop\_mask](https://docs.pytorch.org/docs/2.9/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.noop_mask)

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