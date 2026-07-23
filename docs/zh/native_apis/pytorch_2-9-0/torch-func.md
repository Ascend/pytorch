# torch.func

> [!NOTE]  
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)
- [Why composable function transforms?](#why-composable-function-transforms)

## base API

### torch.func.grad_and_value

<div style="margin-left: 2em">

**原生文档**：[torch.func.grad_and_value](https://pytorch.org/docs/2.9/generated/torch.func.grad_and_value.html)

**是否支持**：是

</div>

### torch.func.jvp

<div style="margin-left: 2em">

**原生文档**：[torch.func.jvp](https://pytorch.org/docs/2.9/generated/torch.func.jvp.html)

**是否支持**：否

</div>

### torch.func.linearize

<div style="margin-left: 2em">

**原生文档**：[torch.func.linearize](https://pytorch.org/docs/2.9/generated/torch.func.linearize.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.func.jacrev

<div style="margin-left: 2em">

**原生文档**：[torch.func.jacrev](https://pytorch.org/docs/2.9/generated/torch.func.jacrev.html)

**是否支持**：否

</div>

### torch.func.jacfwd

<div style="margin-left: 2em">

**原生文档**：[torch.func.jacfwd](https://pytorch.org/docs/2.9/generated/torch.func.jacfwd.html)

**是否支持**：否

</div>

### torch.func.hessian

<div style="margin-left: 2em">

**原生文档**：[torch.func.hessian](https://pytorch.org/docs/2.9/generated/torch.func.hessian.html)

**是否支持**：否

</div>

### torch.func.functionalize

<div style="margin-left: 2em">

**原生文档**：[torch.func.functionalize](https://pytorch.org/docs/2.9/generated/torch.func.functionalize.html)

**是否支持**：否

</div>

### torch.func.functional_call

<div style="margin-left: 2em">

**原生文档**：[torch.func.functional_call](https://pytorch.org/docs/2.9/generated/torch.func.functional_call.html)

**是否支持**：是

</div>

### torch.func.stack_module_state

<div style="margin-left: 2em">

**原生文档**：[torch.func.stack_module_state](https://pytorch.org/docs/2.9/generated/torch.func.stack_module_state.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.func.replace_all_batch_norm_modules_

<div style="margin-left: 2em">

**原生文档**：[torch.func.replace_all_batch_norm_modules_](https://pytorch.org/docs/2.9/generated/torch.func.replace_all_batch_norm_modules_.html)

**是否支持**：否

</div>

## Why composable function transforms?

### torch.func.vmap

<div style="margin-left: 2em">

**原生文档**：[torch.func.vmap](https://pytorch.org/docs/2.9/generated/torch.func.vmap.html)

**是否支持**：否

</div>

### torch.func.grad

<div style="margin-left: 2em">

**原生文档**：[torch.func.grad](https://pytorch.org/docs/2.9/generated/torch.func.grad.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.func.vjp

<div style="margin-left: 2em">

**原生文档**：[torch.func.vjp](https://pytorch.org/docs/2.9/generated/torch.func.vjp.html)

**是否支持**：否

</div>
