# torch.overrides

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.10/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.10/torch.overrides.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Functions](#functions)

</div>

<div style="display:none;">

## &#8203;torch.overrides

</div>

## Functions

### torch.overrides.get_ignored_functions

<div style="margin-left: 2em">

**原生文档**：[torch.overrides.get_ignored_functions](https://pytorch.org/docs/2.10/torch.overrides.html#torch.overrides.get_ignored_functions)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id3 -->

</div>

### torch.overrides.get_overridable_functions

<div style="margin-left: 2em">

**原生文档**：[torch.overrides.get_overridable_functions](https://pytorch.org/docs/2.10/torch.overrides.html#torch.overrides.get_overridable_functions)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id6 -->

</div>

### torch.overrides.resolve_name

<div style="margin-left: 2em">

**原生文档**：[torch.overrides.resolve_name](https://pytorch.org/docs/2.10/torch.overrides.html#torch.overrides.resolve_name)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id9 -->

</div>

### torch.overrides.get_testing_overrides

<div style="margin-left: 2em">

**原生文档**：[torch.overrides.get_testing_overrides](https://pytorch.org/docs/2.10/torch.overrides.html#torch.overrides.get_testing_overrides)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id12 -->

</div>

### torch.overrides.has_torch_function

<div style="margin-left: 2em">

**原生文档**：[torch.overrides.has_torch_function](https://pytorch.org/docs/2.10/torch.overrides.html#torch.overrides.has_torch_function)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id15 -->

</div>

### torch.overrides.is_tensor_method_or_property

<div style="margin-left: 2em">

**原生文档**：[torch.overrides.is_tensor_method_or_property](https://pytorch.org/docs/2.10/torch.overrides.html#torch.overrides.is_tensor_method_or_property)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id18 -->

</div>

### torch.overrides.wrap_torch_function

<div style="margin-left: 2em">

**原生文档**：[torch.overrides.wrap_torch_function](https://pytorch.org/docs/2.10/torch.overrides.html#torch.overrides.wrap_torch_function)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id21 -->

</div>

### torch.overrides.handle_torch_function

<div style="margin-left: 2em">

**原生文档**：[torch.overrides.handle_torch_function](https://pytorch.org/docs/2.10/torch.overrides.html#torch.overrides.handle_torch_function)

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

</div>

### torch.overrides.is_tensor_like

<div style="margin-left: 2em">

**原生文档**：[torch.overrides.is_tensor_like](https://pytorch.org/docs/2.10/torch.overrides.html#torch.overrides.is_tensor_like)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id27 -->

</div>
