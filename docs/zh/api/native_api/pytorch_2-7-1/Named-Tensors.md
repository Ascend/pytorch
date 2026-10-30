# Named Tensors

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.7/named_tensor.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Named tensor API reference](#named-tensor-api-reference)

</div>

<div style="display:none;">

## &#8203;Named Tensors

</div>

## Named tensor API reference

<!-- npu="910b,A3,950" id25 -->
### <code><i>class</i></code> torch.Tensor
<!-- end id25 -->

<div style="margin-left: 2em">

<!-- npu="910b,A3" id26 -->
**原生文档**：[torch.Tensor](https://docs.pytorch.org/docs/2.7/named_tensor.html#named-tensors)

**产品支持情况**：
<!-- end id26 -->

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->

<!-- npu="910b,A3" id27 -->
**限制与说明**：`self`仅支持fp32
<!-- end id27 -->

<!-- npu="910b,A3" id28 -->
> <font size="3">names</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Tensor.names](https://pytorch.org/docs/2.7/named_tensor.html#torch.Tensor.names)

**产品支持情况**：
<!-- end id28 -->

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->

<!-- npu="910b,A3" id29 -->
**限制与说明**：`self`仅支持fp32

</div>
<!-- end id29 -->

<!-- npu="910b,A3" id30 -->
> <font size="3">rename()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Tensor.rename](https://pytorch.org/docs/2.7/named_tensor.html#torch.Tensor.rename)

**产品支持情况**：
<!-- end id30 -->

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->

<!-- npu="910b,A3" id31 -->
**限制与说明**：`self`仅支持fp32

</div>
<!-- end id31 -->

<!-- npu="910b,A3" id32 -->
> <font size="3">rename_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Tensor.rename_](https://pytorch.org/docs/2.7/named_tensor.html#torch.Tensor.rename_)

**产品支持情况**：
<!-- end id32 -->

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->

<!-- npu="910b,A3" id33 -->
**限制与说明**：`self`仅支持fp32

</div>
<!-- end id33 -->

<!-- npu="910b,A3" id34 -->
> <font size="3">align_to</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Tensor.align_to](https://pytorch.org/docs/2.7/named_tensor.html#torch.Tensor.align_to)

**产品支持情况**：
<!-- end id34 -->

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->

<!-- npu="910b,A3" id35 -->
**限制与说明**：`self`仅支持fp32

</div>
<!-- end id35 -->

<!-- npu="910b,A3" id36 -->
> <font size="3">refine_names()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Tensor.refine_names](https://pytorch.org/docs/2.7/named_tensor.html#torch.Tensor.refine_names)

**产品支持情况**：
<!-- end id36 -->

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->

<!-- npu="910b,A3" id37 -->
**限制与说明**：`self`仅支持fp32

</div>
<!-- end id37 -->

<!-- npu="910b,A3" id38 -->
> <font size="3">align_as()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Tensor.align_as](https://pytorch.org/docs/2.7/named_tensor.html#torch.Tensor.align_as)

**产品支持情况**：
<!-- end id38 -->

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->

<!-- npu="910b,A3" id39 -->
**限制与说明**：`self`仅支持fp32

</div>
<!-- end id39 -->

> <font size="3">flatten()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Tensor.flatten](https://docs.pytorch.org/docs/2.7/named_tensor.html#manipulating-dimensions)

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

**限制与说明**：`self`仅支持fp32

</div>

</div>
