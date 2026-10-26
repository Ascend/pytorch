# torch.nn.attention

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.11/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Utils](#utils)
- [Submodules](#submodules)

</div>

<div style="display:none;">

## &#8203;torch.nn.attention

</div>

## Utils

### <code><i>class</i></code> torch.nn.attention.SDPBackend

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.SDPBackend](https://docs.pytorch.org/docs/2.11/generated/torch.nn.attention.SDPBackend.html)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id12 -->

> <font size="3">name</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.SDPBackend.name](https://docs.pytorch.org/docs/2.11/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend.name)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id15 -->

</div>

</div>

## Submodules

### torch.nn.attention.flex_attention

#### BlockMask Utilities

##### torch.nn.attention.flex_attention.or_masks

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.flex_attention.or_masks](https://docs.pytorch.org/docs/2.11/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.or_masks)

**产品支持情况**：

<!-- npu="910b" id1020 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1020 -->
<!-- npu="A3" id1021 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1021 -->
<!-- npu="950" id1022 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1022 -->

</div>

##### torch.nn.attention.flex_attention.noop_mask

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.flex_attention.noop_mask](https://docs.pytorch.org/docs/2.11/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.noop_mask)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id18 -->

</div>

#### BlockMask

##### <code><i>class</i></code> torch.nn.attention.flex_attention.BlockMask

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.flex_attention.BlockMask](https://docs.pytorch.org/docs/2.11/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id3 -->

> <font size="3">BLOCK_SIZE</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.flex_attention.BlockMask.BLOCK_SIZE](https://docs.pytorch.org/docs/2.11/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id6 -->

</div>

> <font size="3">kv_indices</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.flex_attention.BlockMask.kv_indices](https://docs.pytorch.org/docs/2.11/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.kv_indices)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id9 -->

</div>

> <font size="3">seq\_lengths</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.flex_attention.BlockMask.seq_lengths](https://docs.pytorch.org/docs/2.11/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.seq_lengths)

**产品支持情况**：

<!-- npu="910b" id1017 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1017 -->
<!-- npu="A3" id1018 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1018 -->
<!-- npu="950" id1019 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1019 -->

</div>

> <font size="3"><code><i>property</i></code> shape</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.flex_attention.BlockMask.shape](https://docs.pytorch.org/docs/2.11/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.shape)

**产品支持情况**：

<!-- npu="910b" id1023 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1023 -->
<!-- npu="A3" id1024 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1024 -->
<!-- npu="950" id1025 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1025 -->

</div>

> <font size="3">to()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.flex_attention.BlockMask.to](https://docs.pytorch.org/docs/2.11/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.to)

**产品支持情况**：

<!-- npu="910b" id1026 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1026 -->
<!-- npu="A3" id1027 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1027 -->
<!-- npu="950" id1028 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1028 -->

</div>

> <font size="3">to_dense()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.flex_attention.BlockMask.to_dense](https://docs.pytorch.org/docs/2.11/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.to_dense)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id21 -->

</div>

</div>

### torch.nn.attention.bias

#### torch.nn.attention.bias.causal_lower_right

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.bias.causal_lower_right](https://pytorch.org/docs/2.11/generated/torch.nn.attention.bias.causal_lower_right.html)

**产品支持情况**：

<!-- npu="910b" id1029 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1029 -->
<!-- npu="A3" id1030 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1030 -->
<!-- npu="950" id1031 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1031 -->

</div>
