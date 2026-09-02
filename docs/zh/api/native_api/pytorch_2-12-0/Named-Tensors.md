# Named Tensors

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.12/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.12/named_tensor.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Named tensor API reference](#named-tensor-api-reference)

</div>

<div style="display:none;">

## &#8203;Named Tensors

</div>

## Named tensor API reference

### <code><i>class</i></code> torch.Tensor

<div style="margin-left: 2em">

**原生文档**：[torch.Tensor](https://pytorch.org/docs/2.12/named_tensor.html#torch.Tensor)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`self`仅支持fp32

> <font size="3">names</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Tensor.names](https://pytorch.org/docs/2.12/named_tensor.html#torch.Tensor.names)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`self`仅支持fp32

</div>

> <font size="3">rename()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Tensor.rename](https://pytorch.org/docs/2.12/named_tensor.html#torch.Tensor.rename)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`self`仅支持fp32

</div>

> <font size="3">rename_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Tensor.rename_](https://pytorch.org/docs/2.12/named_tensor.html#torch.Tensor.rename_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`self`仅支持fp32

</div>

> <font size="3">align_to()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Tensor.align_to](https://pytorch.org/docs/2.12/named_tensor.html#torch.Tensor.align_to)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`self`仅支持fp32

</div>

> <font size="3">refine_names()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Tensor.refine_names](https://pytorch.org/docs/2.12/named_tensor.html#torch.Tensor.refine_names)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`self`仅支持fp32

</div>

> <font size="3">align_as()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Tensor.align_as](https://pytorch.org/docs/2.12/named_tensor.html#torch.Tensor.align_as)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`self`仅支持fp32

</div>

> <font size="3">flatten()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.Tensor.flatten](https://pytorch.org/docs/2.12/named_tensor.html#torch.Tensor.flatten)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`self`仅支持fp32

</div>

</div>
