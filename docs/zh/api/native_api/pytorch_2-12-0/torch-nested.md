# torch.nested

> [!NOTE]
>
> - API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.12/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。

## 目录

- [base API](#base-api)

## base API

### torch.nested.nested_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.nested.nested_tensor](https://pytorch.org/docs/2.12/nested.html#torch.nested.nested_tensor)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**：目前嵌套张量只支持创建，不支持其他操作

</div>

### torch.nested.as_nested_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.nested.as_nested_tensor](https://pytorch.org/docs/2.12/nested.html#torch.nested.as_nested_tensor)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**：目前嵌套张量只支持创建，不支持其他操作

</div>

### torch.nested.to_padded_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.nested.to_padded_tensor](https://pytorch.org/docs/2.12/nested.html#torch.nested.to_padded_tensor)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.nested.nested_tensor_from_jagged

<div style="margin-left: 2em">

**原生文档**：[torch.nested.nested_tensor_from_jagged](https://pytorch.org/docs/2.12/nested.html#torch.nested.nested_tensor_from_jagged)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.nested.masked_select

<div style="margin-left: 2em">

**原生文档**：[torch.nested.masked_select](https://pytorch.org/docs/2.12/nested.html#torch.nested.masked_select)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.nested.narrow

<div style="margin-left: 2em">

**原生文档**：[torch.nested.narrow](https://pytorch.org/docs/2.12/nested.html#torch.nested.narrow)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： <term>Ascend 950DT</term>：不支持complex64，complex128

</div>
