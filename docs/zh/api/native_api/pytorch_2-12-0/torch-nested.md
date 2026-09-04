# torch.nested

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.12/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.12/nested.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Detailed Docs for Construction and Conversion Functions](#detailed-docs-for-construction-and-conversion-functions)

</div>

<div style="display:none;">

## &#8203;torch.nested

</div>

## Detailed Docs for Construction and Conversion Functions

### torch.nested.nested_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.nested.nested_tensor](https://pytorch.org/docs/2.12/nested.html#torch.nested.nested_tensor)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**：目前嵌套张量只支持创建，不支持其他操作

</div>

### torch.nested.as_nested_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.nested.as_nested_tensor](https://pytorch.org/docs/2.12/nested.html#torch.nested.as_nested_tensor)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**：目前嵌套张量只支持创建，不支持其他操作

</div>

### torch.nested.to_padded_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.nested.to_padded_tensor](https://pytorch.org/docs/2.12/nested.html#torch.nested.to_padded_tensor)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.nested.nested_tensor_from_jagged

<div style="margin-left: 2em">

**原生文档**：[torch.nested.nested_tensor_from_jagged](https://pytorch.org/docs/2.12/nested.html#torch.nested.nested_tensor_from_jagged)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.nested.masked_select

<div style="margin-left: 2em">

**原生文档**：[torch.nested.masked_select](https://pytorch.org/docs/2.12/nested.html#torch.nested.masked_select)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.nested.narrow

<div style="margin-left: 2em">

**原生文档**：[torch.nested.narrow](https://pytorch.org/docs/2.12/nested.html#torch.nested.narrow)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**： <term>Ascend 950DT</term>：不支持complex64，complex128

</div>
