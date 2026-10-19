# torch.export

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.7/export.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [API Reference](#api-reference)

</div>

<div style="display:none;">

## &#8203;torch.export

</div>

## API Reference

### <code><i>class</i></code> torch.export.Constraint

<div style="margin-left: 2em">

**原生文档**：[torch.export.Constraint](https://pytorch.org/docs/2.7/export.html#torch.export.Constraint)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id3 -->

</div>

### <code><i>class</i></code> torch.export.ExportBackwardSignature

<div style="margin-left: 2em">

**原生文档**：[torch.export.ExportBackwardSignature](https://pytorch.org/docs/2.7/export.html#torch.export.ExportBackwardSignature)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id6 -->

</div>

### <code><i>class</i></code> torch.export.ModuleCallEntry

<div style="margin-left: 2em">

**原生文档**：[torch.export.ModuleCallEntry](https://pytorch.org/docs/2.7/export.html#torch.export.ModuleCallEntry)

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

### torch.export.export

<div style="margin-left: 2em">

**原生文档**：[torch.export.export](https://pytorch.org/docs/2.7/export.html#torch.export.export)

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

</div>

### torch.export.save

<div style="margin-left: 2em">

**原生文档**：[torch.export.save](https://pytorch.org/docs/2.7/export.html#torch.export.save)

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

### torch.export.load

<div style="margin-left: 2em">

**原生文档**：[torch.export.load](https://pytorch.org/docs/2.7/export.html#torch.export.load)

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

### torch.export.register_dataclass

<div style="margin-left: 2em">

**原生文档**：[torch.export.register_dataclass](https://pytorch.org/docs/2.7/export.html#torch.export.register_dataclass)

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

### <code><i>class</i></code> torch.export.decomp_utils.CustomDecompTable

<div style="margin-left: 2em">

**原生文档**：[torch.export.decomp_utils.CustomDecompTable](https://pytorch.org/docs/2.7/export.html#torch.export.decomp_utils.CustomDecompTable)

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

> <font size="3">copy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.export.decomp_utils.CustomDecompTable.copy](https://pytorch.org/docs/2.7/export.html#torch.export.decomp_utils.CustomDecompTable.copy)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id39 -->

</div>

> <font size="3">items()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.export.decomp_utils.CustomDecompTable.items](https://pytorch.org/docs/2.7/export.html#torch.export.decomp_utils.CustomDecompTable.items)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id42 -->

</div>

> <font size="3">keys()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.export.decomp_utils.CustomDecompTable.keys](https://pytorch.org/docs/2.7/export.html#torch.export.decomp_utils.CustomDecompTable.keys)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id45 -->

</div>

> <font size="3">materialize()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.export.decomp_utils.CustomDecompTable.materialize](https://pytorch.org/docs/2.7/export.html#torch.export.decomp_utils.CustomDecompTable.materialize)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id48 -->

</div>

> <font size="3">pop()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.export.decomp_utils.CustomDecompTable.pop](https://pytorch.org/docs/2.7/export.html#torch.export.decomp_utils.CustomDecompTable.pop)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id51 -->

</div>

> <font size="3">update()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.export.decomp_utils.CustomDecompTable.update](https://pytorch.org/docs/2.7/export.html#torch.export.decomp_utils.CustomDecompTable.update)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id54 -->

</div>

</div>

### <code><i>class</i></code> torch.export.graph_signature.InputKind

<div style="margin-left: 2em">

**原生文档**：[torch.export.graph_signature.InputKind](https://pytorch.org/docs/2.7/export.html#torch.export.graph_signature.InputKind)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id57 -->

</div>

### <code><i>class</i></code> torch.export.graph_signature.InputSpec

<div style="margin-left: 2em">

**原生文档**：[torch.export.graph_signature.InputSpec](https://pytorch.org/docs/2.7/export.html#torch.export.graph_signature.InputSpec)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id60 -->

</div>

### <code><i>class</i></code> torch.export.graph_signature.OutputKind

<div style="margin-left: 2em">

**原生文档**：[torch.export.graph_signature.OutputKind](https://pytorch.org/docs/2.7/export.html#torch.export.graph_signature.OutputKind)

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id63 -->

</div>

### <code><i>class</i></code> torch.export.graph_signature.OutputSpec

<div style="margin-left: 2em">

**原生文档**：[torch.export.graph_signature.OutputSpec](https://pytorch.org/docs/2.7/export.html#torch.export.graph_signature.OutputSpec)

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id66 -->

</div>

### <code><i>class</i></code> torch.export.graph_signature.SymBoolArgument

<div style="margin-left: 2em">

**原生文档**：[torch.export.graph_signature.SymBoolArgument](https://pytorch.org/docs/2.7/export.html#torch.export.graph_signature.SymBoolArgument)

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id69 -->

</div>

### <code><i>class</i></code> torch.export.graph_signature.SymFloatArgument

<div style="margin-left: 2em">

**原生文档**：[torch.export.graph_signature.SymFloatArgument](https://pytorch.org/docs/2.7/export.html#torch.export.graph_signature.SymFloatArgument)

**产品支持情况**：

<!-- npu="910b" id70 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id72 -->

</div>

### <code><i>class</i></code> torch.export.graph_signature.SymIntArgument

<div style="margin-left: 2em">

**原生文档**：[torch.export.graph_signature.SymIntArgument](https://pytorch.org/docs/2.7/export.html#torch.export.graph_signature.SymIntArgument)

**产品支持情况**：

<!-- npu="910b" id73 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id75 -->

</div>

### torch.export.passes.move_to_device_pass

<div style="margin-left: 2em">

**原生文档**：[torch.export.passes.move_to_device_pass](https://pytorch.org/docs/2.7/export.html#torch.export.passes.move_to_device_pass)

**产品支持情况**：

<!-- npu="910b" id76 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id78 -->

</div>

### <code><i>class</i></code> torch.export.ExportedProgram

<div style="margin-left: 2em">

**原生文档**：[torch.export.ExportedProgram](https://pytorch.org/docs/2.7/export.html#torch.export.ExportedProgram)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id21 -->

> <font size="3">module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.export.ExportedProgram.module](https://pytorch.org/docs/2.7/export.html#torch.export.ExportedProgram.module)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id24 -->

</div>

</div>

### <code><i>class</i></code> torch.export.ExportGraphSignature

<div style="margin-left: 2em">

**原生文档**：[torch.export.ExportGraphSignature](https://pytorch.org/docs/2.7/export.html#torch.export.ExportGraphSignature)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id27 -->

</div>

### <code><i>class</i></code> torch.export.ModuleCallSignature

<div style="margin-left: 2em">

**原生文档**：[torch.export.ModuleCallSignature](https://pytorch.org/docs/2.7/export.html#torch.export.ModuleCallSignature)

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
