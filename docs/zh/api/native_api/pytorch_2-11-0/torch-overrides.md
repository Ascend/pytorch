# torch.overrides

> [!NOTE]
>
> - API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.11/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。

## 目录

- [base API](#base-api)
- [Functions](#functions)

## base API

### torch.overrides.get_ignored_functions

<div style="margin-left: 2em">

**原生文档**：[torch.overrides.get_ignored_functions](https://pytorch.org/docs/2.11/torch.overrides.html#torch.overrides.get_ignored_functions)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.overrides.get_overridable_functions

<div style="margin-left: 2em">

**原生文档**：[torch.overrides.get_overridable_functions](https://pytorch.org/docs/2.11/torch.overrides.html#torch.overrides.get_overridable_functions)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.overrides.resolve_name

<div style="margin-left: 2em">

**原生文档**：[torch.overrides.resolve_name](https://pytorch.org/docs/2.11/torch.overrides.html#torch.overrides.resolve_name)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.overrides.get_testing_overrides

<div style="margin-left: 2em">

**原生文档**：[torch.overrides.get_testing_overrides](https://pytorch.org/docs/2.11/torch.overrides.html#torch.overrides.get_testing_overrides)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.overrides.has_torch_function

<div style="margin-left: 2em">

**原生文档**：[torch.overrides.has_torch_function](https://pytorch.org/docs/2.11/torch.overrides.html#torch.overrides.has_torch_function)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.overrides.is_tensor_method_or_property

<div style="margin-left: 2em">

**原生文档**：[torch.overrides.is_tensor_method_or_property](https://pytorch.org/docs/2.11/torch.overrides.html#torch.overrides.is_tensor_method_or_property)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.overrides.wrap_torch_function

<div style="margin-left: 2em">

**原生文档**：[torch.overrides.wrap_torch_function](https://pytorch.org/docs/2.11/torch.overrides.html#torch.overrides.wrap_torch_function)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

## Functions

### torch.overrides.handle_torch_function

<div style="margin-left: 2em">

**原生文档**：[torch.overrides.handle_torch_function](https://pytorch.org/docs/2.11/torch.overrides.html#torch.overrides.handle_torch_function)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.overrides.is_tensor_like

<div style="margin-left: 2em">

**原生文档**：[torch.overrides.is_tensor_like](https://pytorch.org/docs/2.11/torch.overrides.html#torch.overrides.is_tensor_like)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>
