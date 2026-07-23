# torch.autograd

> [!NOTE]  
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)
- [Context method mixins](#context-method-mixins)
- [Profiler](#profiler)
- [Default gradient layouts](#default-gradient-layouts)
- [Tensor autograd functions](#tensor-autograd-functions)
- [Forward-mode Automatic Differentiation](#forward-mode-automatic-differentiation)
- [Functional higher level API](#functional-higher-level-api)
- [Numerical gradient checking](#numerical-gradient-checking)
- [Debugging and anomaly detection](#debugging-and-anomaly-detection)

## base API

### torch.autograd.detect_anomaly

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.detect_anomaly](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.detect_anomaly)

**是否支持**：是

</div>

### torch.autograd.set_detect_anomaly

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.set_detect_anomaly](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.set_detect_anomaly)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.autograd.graph.saved_tensors_hooks

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.graph.saved_tensors_hooks](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.graph.saved_tensors_hooks)

**是否支持**：是

</div>

### torch.autograd.graph.save_on_cpu

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.graph.save_on_cpu](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.graph.save_on_cpu)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.autograd.graph.disable_saved_tensors_hooks

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.graph.disable_saved_tensors_hooks](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.graph.disable_saved_tensors_hooks)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.autograd.graph.register_multi_grad_hook

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.graph.register_multi_grad_hook](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.graph.register_multi_grad_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.autograd.graph.allow_mutation_on_saved_tensors

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.graph.allow_mutation_on_saved_tensors](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.graph.allow_mutation_on_saved_tensors)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### _`class`_ torch.autograd.FunctionCtx

<div style="margin-left: 2em">

> <font size="3">mark_dirty()</font>

<div style="margin-left: 2em">

**原生文档**：[FunctionCtx.mark_dirty](https://pytorch.org/docs/2.11/generated/torch.autograd.function.FunctionCtx.mark_dirty.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mark_non_differentiable()</font>

<div style="margin-left: 2em">

**原生文档**：[FunctionCtx.mark_non_differentiable](https://pytorch.org/docs/2.11/generated/torch.autograd.function.FunctionCtx.mark_non_differentiable.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">save_for_backward()</font>

<div style="margin-left: 2em">

**原生文档**：[FunctionCtx.save_for_backward](https://pytorch.org/docs/2.11/generated/torch.autograd.function.FunctionCtx.save_for_backward.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_materialize_grads()</font>

<div style="margin-left: 2em">

**原生文档**：[FunctionCtx.set_materialize_grads](https://pytorch.org/docs/2.11/generated/torch.autograd.function.FunctionCtx.set_materialize_grads.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.autograd.Node

<div style="margin-left: 2em">

> <font size="3">name()</font>

<div style="margin-left: 2em">

**原生文档**：[Node.name](https://pytorch.org/docs/2.11/generated/torch.autograd.graph.Node.name.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">metadata()</font>

<div style="margin-left: 2em">

**原生文档**：[Node.metadata](https://pytorch.org/docs/2.11/generated/torch.autograd.graph.Node.metadata.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">next_functions()</font>

<div style="margin-left: 2em">

**原生文档**：[Node.next_functions](https://pytorch.org/docs/2.11/generated/torch.autograd.graph.Node.next_functions.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[Node.register_hook](https://pytorch.org/docs/2.11/generated/torch.autograd.graph.Node.register_hook.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">register_prehook()</font>

<div style="margin-left: 2em">

**原生文档**：[Node.register_prehook](https://pytorch.org/docs/2.11/generated/torch.autograd.graph.Node.register_prehook.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

## Context method mixins

### _`class`_ torch.autograd.Function

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.Function](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.Function)

**是否支持**：是

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[Function.forward](https://pytorch.org/docs/2.11/generated/torch.autograd.Function.forward.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">backward()</font>

<div style="margin-left: 2em">

**原生文档**：[Function.backward](https://pytorch.org/docs/2.11/generated/torch.autograd.Function.backward.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">jvp()</font>

<div style="margin-left: 2em">

**原生文档**：[Function.jvp](https://pytorch.org/docs/2.11/generated/torch.autograd.Function.jvp.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">vmap()</font>

<div style="margin-left: 2em">

**原生文档**：[Function.vmap](https://pytorch.org/docs/2.11/generated/torch.autograd.Function.vmap.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

## Profiler

### torch.autograd.profiler.profile

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.profiler.profile](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.profiler.profile)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 采集NPU上的profiling数据时，“use_device”需设置为“npu”

</div>

### torch.autograd.profiler.emit_nvtx

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.profiler.emit_nvtx](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.profiler.emit_nvtx)

**是否支持**：否

</div>

### torch.autograd.profiler.emit_itt

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.profiler.emit_itt](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.profiler.emit_itt)

**是否支持**：否

</div>

### profile.export_chrome_trace

<div style="margin-left: 2em">

**原生文档**：[profile.export_chrome_trace](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.profile.export_chrome_trace.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### profile.key_averages

<div style="margin-left: 2em">

**原生文档**：[profile.key_averages](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.profile.key_averages.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### profile.self_cpu_time_total

<div style="margin-left: 2em">

**原生文档**：[profile.self_cpu_time_total](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.profile.self_cpu_time_total.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### profile.total_average

<div style="margin-left: 2em">

**原生文档**：[profile.total_average](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.profile.total_average.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.autograd.profiler.load_nvprof

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.profiler.load_nvprof](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.load_nvprof.html)

**是否支持**：否

</div>

## Default gradient layouts

### torch.autograd.backward

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.backward](https://pytorch.org/docs/2.11/generated/torch.autograd.backward.html)

**是否支持**：是

**限制与说明**：

- 支持bf16，fp16，fp32，fp64
- 不支持稀疏张量

</div>

## Tensor autograd functions

### torch.autograd.grad

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.grad](https://pytorch.org/docs/2.11/generated/torch.autograd.grad.html)

**是否支持**：是

</div>

## Forward-mode Automatic Differentiation

### torch.autograd.forward_ad.dual_level

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.forward_ad.dual_level](https://pytorch.org/docs/2.11/generated/torch.autograd.forward_ad.dual_level.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.autograd.forward_ad.make_dual

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.forward_ad.make_dual](https://pytorch.org/docs/2.11/generated/torch.autograd.forward_ad.make_dual.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.autograd.forward_ad.unpack_dual

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.forward_ad.unpack_dual](https://pytorch.org/docs/2.11/generated/torch.autograd.forward_ad.unpack_dual.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

## Functional higher level API

### torch.autograd.functional.jacobian

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.functional.jacobian](https://pytorch.org/docs/2.11/generated/torch.autograd.functional.jacobian.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.autograd.functional.hessian

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.functional.hessian](https://pytorch.org/docs/2.11/generated/torch.autograd.functional.hessian.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.autograd.functional.vjp

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.functional.vjp](https://pytorch.org/docs/2.11/generated/torch.autograd.functional.vjp.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.autograd.functional.jvp

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.functional.jvp](https://pytorch.org/docs/2.11/generated/torch.autograd.functional.jvp.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.autograd.functional.vhp

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.functional.vhp](https://pytorch.org/docs/2.11/generated/torch.autograd.functional.vhp.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.autograd.functional.hvp

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.functional.hvp](https://pytorch.org/docs/2.11/generated/torch.autograd.functional.hvp.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

## Numerical gradient checking

### torch.autograd.gradcheck.gradcheck

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.gradcheck.gradcheck](https://pytorch.org/docs/2.11/generated/torch.autograd.gradcheck.gradcheck.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.autograd.gradcheck.gradgradcheck

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.gradcheck.gradgradcheck](https://pytorch.org/docs/2.11/generated/torch.autograd.gradcheck.gradgradcheck.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## Debugging and anomaly detection

### torch.autograd.grad_mode.set_multithreading_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.grad_mode.set_multithreading_enabled](https://pytorch.org/docs/2.11/generated/torch.autograd.grad_mode.set_multithreading_enabled.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>
