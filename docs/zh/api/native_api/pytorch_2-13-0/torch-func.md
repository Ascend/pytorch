# torch.func

> [!NOTE]
>
> - API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.13/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。

## 目录

- [base API](#base-api)
- [Why composable function transforms?](#why-composable-function-transforms)

## base API

### torch.func.grad_and_value

<div style="margin-left: 2em">

**原生文档**：[torch.func.grad_and_value](https://pytorch.org/docs/2.13/generated/torch.func.grad_and_value.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### torch.func.jvp

<div style="margin-left: 2em">

**原生文档**：[torch.func.jvp](https://pytorch.org/docs/2.13/generated/torch.func.jvp.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.func.linearize

<div style="margin-left: 2em">

**原生文档**：[torch.func.linearize](https://pytorch.org/docs/2.13/generated/torch.func.linearize.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.func.jacrev

<div style="margin-left: 2em">

**原生文档**：[torch.func.jacrev](https://pytorch.org/docs/2.13/generated/torch.func.jacrev.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.func.jacfwd

<div style="margin-left: 2em">

**原生文档**：[torch.func.jacfwd](https://pytorch.org/docs/2.13/generated/torch.func.jacfwd.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.func.hessian

<div style="margin-left: 2em">

**原生文档**：[torch.func.hessian](https://pytorch.org/docs/2.13/generated/torch.func.hessian.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.func.functionalize

<div style="margin-left: 2em">

**原生文档**：[torch.func.functionalize](https://pytorch.org/docs/2.13/generated/torch.func.functionalize.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.func.functional_call

<div style="margin-left: 2em">

**原生文档**：[torch.func.functional_call](https://pytorch.org/docs/2.13/generated/torch.func.functional_call.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

</div>

### torch.func.stack_module_state

<div style="margin-left: 2em">

**原生文档**：[torch.func.stack_module_state](https://pytorch.org/docs/2.13/generated/torch.func.stack_module_state.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.func.replace_all_batch_norm_modules_

<div style="margin-left: 2em">

**原生文档**：[torch.func.replace_all_batch_norm_modules_](https://pytorch.org/docs/2.13/generated/torch.func.replace_all_batch_norm_modules_.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

## Why composable function transforms?

### torch.func.vmap

<div style="margin-left: 2em">

**原生文档**：[torch.func.vmap](https://pytorch.org/docs/2.13/generated/torch.func.vmap.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.func.grad

<div style="margin-left: 2em">

**原生文档**：[torch.func.grad](https://pytorch.org/docs/2.13/generated/torch.func.grad.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

### torch.func.vjp

<div style="margin-left: 2em">

**原生文档**：[torch.func.vjp](https://pytorch.org/docs/2.13/generated/torch.func.vjp.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10007; |
| <term>Atlas A3 训练系列产品</term> | &#10007; |
| <term>Ascend 950DT</term> | &#10007; |

</div>
