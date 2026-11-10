# torch.nn.attention

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 本支持清单的范围以[PyTorch社区文档](https://pytorch.org/docs/2.10/)为准，社区文档中未列出的接口不在本清单范围内；已在社区文档中存在但未收录于本清单的API尚未验证，请谨慎使用，我们将在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://docs.pytorch.org/docs/2.10/nn.attention.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Utils](#utils)
- [Submodules](#submodules)

</div>

## Utils

### <code><i>class</i></code> torch.nn.attention.SDPBackend

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.SDPBackend](https://docs.pytorch.org/docs/2.10/generated/torch.nn.attention.SDPBackend.html)

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

<a class="method_anchor" id="torch_nn_attention_SDPBackend_name"></a>

> <font size="3">name</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.SDPBackend.name](https://docs.pytorch.org/docs/2.10/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend.name)

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

##### torch.nn.attention.flex_attention.noop_mask

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.flex_attention.noop_mask](https://docs.pytorch.org/docs/2.10/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.noop_mask)

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

**原生文档**：[torch.nn.attention.flex_attention.BlockMask](https://docs.pytorch.org/docs/2.10/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask)

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

<a class="method_anchor" id="torch_nn_attention_flex_attention_BlockMask_BLOCK_SIZE"></a>

> <font size="3">BLOCK_SIZE</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.flex_attention.BlockMask.BLOCK_SIZE](https://docs.pytorch.org/docs/2.10/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask)

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

<a class="method_anchor" id="torch_nn_attention_flex_attention_BlockMask_kv_indices"></a>

> <font size="3">kv_indices</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.flex_attention.BlockMask.kv_indices](https://docs.pytorch.org/docs/2.10/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.kv_indices)

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

<a class="method_anchor" id="torch_nn_attention_flex_attention_BlockMask_seq_lengths"></a>

> <font size="3">seq\_lengths</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.flex_attention.BlockMask.seq_lengths](https://docs.pytorch.org/docs/2.10/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.seq_lengths)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id24 -->

</div>

<a class="method_anchor" id="torch_nn_attention_flex_attention_BlockMask_shape"></a>

> <font size="3"><code><i>property</i></code> shape</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.flex_attention.BlockMask.shape](https://docs.pytorch.org/docs/2.10/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.shape)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id27 -->

</div>

<a class="method_anchor" id="torch_nn_attention_flex_attention_BlockMask_to"></a>

> <font size="3">to()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.flex_attention.BlockMask.to](https://docs.pytorch.org/docs/2.10/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.to)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id30 -->

</div>

<a class="method_anchor" id="torch_nn_attention_flex_attention_BlockMask_to_dense"></a>

> <font size="3">to_dense()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.flex_attention.BlockMask.to_dense](https://docs.pytorch.org/docs/2.10/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.to_dense)

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

**原生文档**：[torch.nn.attention.bias.causal_lower_right](https://pytorch.org/docs/2.10/generated/torch.nn.attention.bias.causal_lower_right.html)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id33 -->

</div>

### <code><i>class</i></code> torch.nn.attention.flex\_attention.BlockMask

<div style="margin-left: 2em">

<a class="method_anchor" id="torch_nn_attention_flex_attention_BlockMask_from_kv_blocks"></a>

> <font size="3">from_kv_blocks()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.nn.attention.flex\_attention.BlockMask.from\_kv\_blocks](https://docs.pytorch.org/docs/2.10/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.from_kv_blocks)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id36 -->

</div>

</div>
