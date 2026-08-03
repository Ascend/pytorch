# torch.nested

> [!NOTE]
>
> API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>

## 目录

- [base API](#base-api)

## base API

### torch.nested.nested_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.nested.nested_tensor](https://pytorch.org/docs/2.7/nested.html#torch.nested.nested_tensor)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**： 目前嵌套张量只支持创建，不支持其他操作

</div>

### torch.nested.as_nested_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.nested.as_nested_tensor](https://pytorch.org/docs/2.7/nested.html#torch.nested.as_nested_tensor)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

**限制与说明**： 目前嵌套张量只支持创建，不支持其他操作

</div>

### torch.nested.to_padded_tensor

<div style="margin-left: 2em">

**原生文档**：[torch.nested.to_padded_tensor](https://pytorch.org/docs/2.7/nested.html#torch.nested.to_padded_tensor)

**产品支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.nested.nested_tensor_from_jagged

<div style="margin-left: 2em">

**原生文档**：[torch.nested.nested_tensor_from_jagged](https://pytorch.org/docs/2.7/nested.html#torch.nested.nested_tensor_from_jagged)

**产品支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.nested.masked_select

<div style="margin-left: 2em">

**原生文档**：[torch.nested.masked_select](https://pytorch.org/docs/2.7/nested.html#torch.nested.masked_select)

**产品支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.nested.narrow

<div style="margin-left: 2em">

**原生文档**：[torch.nested.narrow](https://pytorch.org/docs/2.7/nested.html#torch.nested.narrow)

**产品支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**： <term>Ascend 950DT</term>：不支持complex64，complex128

</div>
