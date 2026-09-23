# torch.nn.init

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.9/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.9/nn.init.html)。

<div style="display:none;">

## &#8203;torch.nn.init

</div>

### torch.nn.init.calculate_gain

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.calculate_gain](https://pytorch.org/docs/2.9/nn.init.html#torch.nn.init.calculate_gain)

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

</div>

### torch.nn.init.uniform_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.uniform_](https://pytorch.org/docs/2.9/nn.init.html#torch.nn.init.uniform_)

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

**限制与说明**：`tensor`仅支持fp32

</div>

### torch.nn.init.normal_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.normal_](https://pytorch.org/docs/2.9/nn.init.html#torch.nn.init.normal_)

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

**限制与说明**：`tensor`仅支持fp32

</div>

### torch.nn.init.constant_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.constant_](https://pytorch.org/docs/2.9/nn.init.html#torch.nn.init.constant_)

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

**限制与说明**：`tensor`仅支持fp32

</div>

### torch.nn.init.ones_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.ones_](https://pytorch.org/docs/2.9/nn.init.html#torch.nn.init.ones_)

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

**限制与说明**：`tensor`仅支持fp32

</div>

### torch.nn.init.zeros_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.zeros_](https://pytorch.org/docs/2.9/nn.init.html#torch.nn.init.zeros_)

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

**限制与说明**：`tensor`仅支持fp32

</div>

### torch.nn.init.eye_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.eye_](https://pytorch.org/docs/2.9/nn.init.html#torch.nn.init.eye_)

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

**限制与说明**：`tensor`仅支持fp32

</div>

### torch.nn.init.dirac_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.dirac_](https://pytorch.org/docs/2.9/nn.init.html#torch.nn.init.dirac_)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id24 -->

**限制与说明**：`tensor`仅支持fp32

</div>

### torch.nn.init.xavier_uniform_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.xavier_uniform_](https://pytorch.org/docs/2.9/nn.init.html#torch.nn.init.xavier_uniform_)

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

**限制与说明**：`tensor`仅支持fp32

</div>

### torch.nn.init.xavier_normal_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.xavier_normal_](https://pytorch.org/docs/2.9/nn.init.html#torch.nn.init.xavier_normal_)

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

**限制与说明**：`tensor`仅支持fp32

</div>

### torch.nn.init.kaiming_uniform_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.kaiming_uniform_](https://pytorch.org/docs/2.9/nn.init.html#torch.nn.init.kaiming_uniform_)

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

**限制与说明**：`tensor`仅支持fp32

</div>

### torch.nn.init.kaiming_normal_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.kaiming_normal_](https://pytorch.org/docs/2.9/nn.init.html#torch.nn.init.kaiming_normal_)

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

**限制与说明**：`tensor`仅支持fp32

</div>

### torch.nn.init.trunc_normal_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.trunc_normal_](https://pytorch.org/docs/2.9/nn.init.html#torch.nn.init.trunc_normal_)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id39 -->

</div>

### torch.nn.init.orthogonal_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.orthogonal_](https://pytorch.org/docs/2.9/nn.init.html#torch.nn.init.orthogonal_)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id42 -->

**限制与说明**：`tensor`仅支持fp32

</div>

### torch.nn.init.sparse_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.sparse_](https://pytorch.org/docs/2.9/nn.init.html#torch.nn.init.sparse_)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id45 -->

**限制与说明**：`tensor`仅支持fp32

</div>
