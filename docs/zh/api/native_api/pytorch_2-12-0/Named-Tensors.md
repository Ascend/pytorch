# Named Tensors

> [!NOTE]
>
> - API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.12/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。

## 目录

- [base API](#base-api)
- [Named dimensions](#named-dimensions)
- [Name propagation semantics](#name-propagation-semantics)
- [Explicit alignment by names](#explicit-alignment-by-names)
- [Manipulating dimensions](#manipulating-dimensions)

## base API

### names

<div style="margin-left: 2em">

**原生文档**：[names](https://pytorch.org/docs/2.12/named_tensor.html#torch.Tensor.names)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：`self`仅支持fp32

</div>

### rename

<div style="margin-left: 2em">

**原生文档**：[rename](https://pytorch.org/docs/2.12/named_tensor.html#torch.Tensor.rename)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：`self`仅支持fp32

</div>

### rename_

<div style="margin-left: 2em">

**原生文档**：[rename_](https://pytorch.org/docs/2.12/named_tensor.html#torch.Tensor.rename_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：`self`仅支持fp32

</div>

### align_to

<div style="margin-left: 2em">

**原生文档**：[align_to](https://pytorch.org/docs/2.12/named_tensor.html#torch.Tensor.align_to)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：`self`仅支持fp32

</div>

## Named dimensions

### <code><i>class</i></code> torch.Tensor

<div style="margin-left: 2em">

**原生文档**：[torch.Tensor](https://pytorch.org/docs/2.12/named_tensor.html#torch.Tensor)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：`self`仅支持fp32

</div>

## Name propagation semantics

### refine_names

<div style="margin-left: 2em">

**原生文档**：[refine_names](https://pytorch.org/docs/2.12/named_tensor.html#torch.Tensor.refine_names)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：`self`仅支持fp32

</div>

## Explicit alignment by names

### align_as

<div style="margin-left: 2em">

**原生文档**：[align_as](https://pytorch.org/docs/2.12/named_tensor.html#torch.Tensor.align_as)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：`self`仅支持fp32

</div>

## Manipulating dimensions

### flatten

<div style="margin-left: 2em">

**原生文档**：[flatten](https://pytorch.org/docs/2.12/named_tensor.html#torch.Tensor.flatten)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

**限制与说明**：`self`仅支持fp32

</div>
