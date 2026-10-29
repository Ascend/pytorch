# torch.nn.attention

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.12/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Utils](#utils)
- [Submodules](#submodules)

</div>

<div style="display:none;">

## &#8203;torch.nn.attention

</div>

## Utils

### <code><i>class</i></code> torch.nn.attention.SDPBackend

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.attention.SDPBackend](https://docs.pytorch.org/docs/2.12/generated/torch.nn.attention.SDPBackend.html)

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

> <font size="3">name</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.attention.SDPBackend.name](https://docs.pytorch.org/docs/2.12/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend.name)

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

</div>

## Submodules

### torch.nn.attention.flex_attention

#### BlockMask Utilities

##### torch.nn.attention.flex_attention.or_masks

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.attention.flex_attention.or_masks](https://docs.pytorch.org/docs/2.12/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.or_masks)

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

##### torch.nn.attention.flex_attention.noop_mask

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.attention.flex_attention.noop_mask](https://docs.pytorch.org/docs/2.12/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.noop_mask)

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

#### BlockMask

##### <code><i>class</i></code> torch.nn.attention.flex_attention.BlockMask

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.attention.flex_attention.BlockMask](https://docs.pytorch.org/docs/2.12/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask)

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

**Constraints & Notes**: -

> <font size="3">BLOCK_SIZE</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.attention.flex_attention.BlockMask.BLOCK_SIZE](https://docs.pytorch.org/docs/2.12/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask)

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

> <font size="3">kv_indices</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.attention.flex_attention.BlockMask.kv_indices](https://docs.pytorch.org/docs/2.12/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.kv_indices)

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

**Constraints & Notes**: -

</div>

> <font size="3">seq\_lengths</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.attention.flex_attention.BlockMask.seq_lengths](https://docs.pytorch.org/docs/2.12/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.seq_lengths)

**Support Status**:

<!-- npu="910b" id1017 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1017 -->
<!-- npu="A3" id1018 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1018 -->
<!-- npu="950" id1019 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1019 -->

</div>

> <font size="3"><code><i>property</i></code> shape</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.attention.flex_attention.BlockMask.shape](https://docs.pytorch.org/docs/2.12/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.shape)

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

> <font size="3">to()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.attention.flex_attention.BlockMask.to](https://docs.pytorch.org/docs/2.12/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.to)

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

> <font size="3">to_dense()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.attention.flex_attention.BlockMask.to_dense](https://docs.pytorch.org/docs/2.12/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.to_dense)

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

</div>

### torch.nn.attention.bias

#### torch.nn.attention.bias.causal_lower_right

<div style="margin-left: 2em">

**Native Documentation**: [torch.nn.attention.bias.causal_lower_right](https://pytorch.org/docs/2.12/generated/torch.nn.attention.bias.causal_lower_right.html)

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
