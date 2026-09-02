# torch.func

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.11/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://docs.pytorch.org/docs/2.11/func.api.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Function Transforms](#function-transforms)
- [Utilities for working with torch.nn.Modules](#utilities-for-working-with-torchnnmodules)

</div>

<div style="display:none;">

## &#8203;torch.func

</div>

## Function Transforms

### torch.func.grad_and_value

<div style="margin-left: 2em">

**原生文档**：[torch.func.grad_and_value](https://pytorch.org/docs/2.11/generated/torch.func.grad_and_value.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### torch.func.jvp

<div style="margin-left: 2em">

**原生文档**：[torch.func.jvp](https://pytorch.org/docs/2.11/generated/torch.func.jvp.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.func.linearize

<div style="margin-left: 2em">

**原生文档**：[torch.func.linearize](https://pytorch.org/docs/2.11/generated/torch.func.linearize.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.func.jacrev

<div style="margin-left: 2em">

**原生文档**：[torch.func.jacrev](https://pytorch.org/docs/2.11/generated/torch.func.jacrev.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.func.jacfwd

<div style="margin-left: 2em">

**原生文档**：[torch.func.jacfwd](https://pytorch.org/docs/2.11/generated/torch.func.jacfwd.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.func.hessian

<div style="margin-left: 2em">

**原生文档**：[torch.func.hessian](https://pytorch.org/docs/2.11/generated/torch.func.hessian.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.func.functionalize

<div style="margin-left: 2em">

**原生文档**：[torch.func.functionalize](https://pytorch.org/docs/2.11/generated/torch.func.functionalize.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.func.vmap

<div style="margin-left: 2em">

**原生文档**：[torch.func.vmap](https://pytorch.org/docs/2.11/generated/torch.func.vmap.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.func.grad

<div style="margin-left: 2em">

**原生文档**：[torch.func.grad](https://pytorch.org/docs/2.11/generated/torch.func.grad.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.func.vjp

<div style="margin-left: 2em">

**原生文档**：[torch.func.vjp](https://pytorch.org/docs/2.11/generated/torch.func.vjp.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

## Utilities for working with torch.nn.Modules

### torch.func.functional_call

<div style="margin-left: 2em">

**原生文档**：[torch.func.functional_call](https://pytorch.org/docs/2.11/generated/torch.func.functional_call.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### torch.func.stack_module_state

<div style="margin-left: 2em">

**原生文档**：[torch.func.stack_module_state](https://pytorch.org/docs/2.11/generated/torch.func.stack_module_state.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.func.replace_all_batch_norm_modules_

<div style="margin-left: 2em">

**原生文档**：[torch.func.replace_all_batch_norm_modules_](https://pytorch.org/docs/2.11/generated/torch.func.replace_all_batch_norm_modules_.html)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>
