# torch.autograd

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.11/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.11/autograd.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Forward-mode Automatic Differentiation](#forward-mode-automatic-differentiation)
- [Functional higher level API](#functional-higher-level-api)
- [Function](#function)
- [Context method mixins](#context-method-mixins)
- [Numerical gradient checking](#numerical-gradient-checking)
- [Profiler](#profiler)
- [Debugging and anomaly detection](#debugging-and-anomaly-detection)
- [Autograd graph](#autograd-graph)

</div>

<div style="display:none;">

## &#8203;torch.autograd

</div>

### torch.autograd.backward

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.backward](https://pytorch.org/docs/2.11/generated/torch.autograd.backward.html)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id3 -->

**限制与说明**：

- `tensors`仅支持bf16，fp16，fp32，fp64
- 不支持稀疏张量

</div>

### torch.autograd.grad

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.grad](https://pytorch.org/docs/2.11/generated/torch.autograd.grad.html)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id6 -->

</div>

## Forward-mode Automatic Differentiation

### torch.autograd.forward_ad.dual_level

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.forward_ad.dual_level](https://pytorch.org/docs/2.11/generated/torch.autograd.forward_ad.dual_level.html)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id9 -->

</div>

### torch.autograd.forward_ad.make_dual

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.forward_ad.make_dual](https://pytorch.org/docs/2.11/generated/torch.autograd.forward_ad.make_dual.html)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id12 -->

**限制与说明**： `tensor`仅支持fp32

</div>

### torch.autograd.forward_ad.unpack_dual

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.forward_ad.unpack_dual](https://pytorch.org/docs/2.11/generated/torch.autograd.forward_ad.unpack_dual.html)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id15 -->

**限制与说明**： `tensor`仅支持fp32

</div>

## Functional higher level API

### <code><i>class</i></code> torch.autograd.function.FunctionCtx

<div style="margin-left: 2em">

> <font size="3">mark_dirty()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.function.FunctionCtx.mark_dirty](https://pytorch.org/docs/2.11/generated/torch.autograd.function.FunctionCtx.mark_dirty.html)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id18 -->

</div>

> <font size="3">mark_non_differentiable()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.function.FunctionCtx.mark_non_differentiable](https://pytorch.org/docs/2.11/generated/torch.autograd.function.FunctionCtx.mark_non_differentiable.html)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id21 -->

</div>

> <font size="3">save_for_backward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.function.FunctionCtx.save_for_backward](https://pytorch.org/docs/2.11/generated/torch.autograd.function.FunctionCtx.save_for_backward.html)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id24 -->

</div>

> <font size="3">set_materialize_grads()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.function.FunctionCtx.set_materialize_grads](https://pytorch.org/docs/2.11/generated/torch.autograd.function.FunctionCtx.set_materialize_grads.html)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id27 -->

</div>

</div>

### <code><i>class</i></code> torch.autograd.graph.Node

<div style="margin-left: 2em">

> <font size="3">name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.graph.Node.name](https://pytorch.org/docs/2.11/generated/torch.autograd.graph.Node.name.html)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id30 -->

</div>

> <font size="3">metadata()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.graph.Node.metadata](https://pytorch.org/docs/2.11/generated/torch.autograd.graph.Node.metadata.html)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id33 -->

</div>

> <font size="3">next_functions()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.graph.Node.next_functions](https://pytorch.org/docs/2.11/generated/torch.autograd.graph.Node.next_functions.html)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id36 -->

</div>

> <font size="3">register_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.graph.Node.register_hook](https://pytorch.org/docs/2.11/generated/torch.autograd.graph.Node.register_hook.html)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id39 -->

</div>

> <font size="3">register_prehook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.graph.Node.register_prehook](https://pytorch.org/docs/2.11/generated/torch.autograd.graph.Node.register_prehook.html)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id42 -->

</div>

</div>

### profile.export_chrome_trace

<div style="margin-left: 2em">

**原生文档**：[profile.export_chrome_trace](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.profile.export_chrome_trace.html)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id45 -->

</div>

### profile.key_averages

<div style="margin-left: 2em">

**原生文档**：[profile.key_averages](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.profile.key_averages.html)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id48 -->

</div>

### profile.self_cpu_time_total

<div style="margin-left: 2em">

**原生文档**：[profile.self_cpu_time_total](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.profile.self_cpu_time_total.html)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id51 -->

</div>

### profile.total_average

<div style="margin-left: 2em">

**原生文档**：[profile.total_average](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.profile.total_average.html)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id54 -->

</div>

### torch.autograd.functional.jacobian

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.functional.jacobian](https://pytorch.org/docs/2.11/generated/torch.autograd.functional.jacobian.html)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id57 -->

**限制与说明**： `inputs`仅支持fp32

</div>

### torch.autograd.functional.hessian

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.functional.hessian](https://pytorch.org/docs/2.11/generated/torch.autograd.functional.hessian.html)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id60 -->

**限制与说明**： `inputs`仅支持fp32

</div>

### torch.autograd.functional.vjp

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.functional.vjp](https://pytorch.org/docs/2.11/generated/torch.autograd.functional.vjp.html)

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id63 -->

**限制与说明**： `inputs`仅支持fp32

</div>

### torch.autograd.functional.jvp

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.functional.jvp](https://pytorch.org/docs/2.11/generated/torch.autograd.functional.jvp.html)

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id66 -->

**限制与说明**： `inputs`仅支持fp32

</div>

### torch.autograd.functional.vhp

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.functional.vhp](https://pytorch.org/docs/2.11/generated/torch.autograd.functional.vhp.html)

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id69 -->

**限制与说明**： `inputs`仅支持fp32

</div>

### torch.autograd.functional.hvp

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.functional.hvp](https://pytorch.org/docs/2.11/generated/torch.autograd.functional.hvp.html)

**产品支持情况**：

<!-- npu="910b" id70 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id72 -->

**限制与说明**： `inputs`仅支持fp32

</div>

## Function

### <code><i>class</i></code> torch.autograd.Function

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.Function](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.Function)

**产品支持情况**：

<!-- npu="910b" id73 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id75 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.Function.forward](https://pytorch.org/docs/2.11/generated/torch.autograd.Function.forward.html)

**产品支持情况**：

<!-- npu="910b" id76 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id78 -->

</div>

> <font size="3">backward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.Function.backward](https://pytorch.org/docs/2.11/generated/torch.autograd.Function.backward.html)

**产品支持情况**：

<!-- npu="910b" id79 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id81 -->

</div>

> <font size="3">jvp()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.Function.jvp](https://pytorch.org/docs/2.11/generated/torch.autograd.Function.jvp.html)

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id84 -->

</div>

> <font size="3">vmap()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.Function.vmap](https://pytorch.org/docs/2.11/generated/torch.autograd.Function.vmap.html)

**产品支持情况**：

<!-- npu="910b" id85 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id87 -->

</div>

</div>

## Context method mixins

### <code><i>class</i></code> torch.autograd.function.FunctionCtx

<div style="margin-left: 2em">

> <font size="3">mark_dirty()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.function.FunctionCtx.mark_dirty](https://pytorch.org/docs/2.11/generated/torch.autograd.function.FunctionCtx.mark_dirty.html)

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id90 -->

</div>

> <font size="3">mark_non_differentiable()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.function.FunctionCtx.mark_non_differentiable](https://pytorch.org/docs/2.11/generated/torch.autograd.function.FunctionCtx.mark_non_differentiable.html)

**产品支持情况**：

<!-- npu="910b" id91 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id93 -->

</div>

> <font size="3">save_for_backward()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.function.FunctionCtx.save_for_backward](https://pytorch.org/docs/2.11/generated/torch.autograd.function.FunctionCtx.save_for_backward.html)

**产品支持情况**：

<!-- npu="910b" id94 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id96 -->

</div>

> <font size="3">set_materialize_grads()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.function.FunctionCtx.set_materialize_grads](https://pytorch.org/docs/2.11/generated/torch.autograd.function.FunctionCtx.set_materialize_grads.html)

**产品支持情况**：

<!-- npu="910b" id97 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id99 -->

</div>

</div>

## Numerical gradient checking

### torch.autograd.gradcheck.gradcheck

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.gradcheck.gradcheck](https://pytorch.org/docs/2.11/generated/torch.autograd.gradcheck.gradcheck.html)

**产品支持情况**：

<!-- npu="910b" id100 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id102 -->

</div>

### torch.autograd.gradcheck.gradgradcheck

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.gradcheck.gradgradcheck](https://pytorch.org/docs/2.11/generated/torch.autograd.gradcheck.gradgradcheck.html)

**产品支持情况**：

<!-- npu="910b" id103 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id105 -->

</div>

### <code><i>class</i></code> torch.autograd.function.NestedIOFunction

<div style="margin-left: 2em">

**原生文档**：[NestedIOFunction](https://pytorch.org/docs/2.11/generated/torch.autograd.function.NestedIOFunction.html)

**产品支持情况**：

<!-- npu="910b" id106 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id108 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**原生文档**：[NestedIOFunction.forward](https://pytorch.org/docs/2.11/generated/torch.autograd.function.NestedIOFunction.html#torch.autograd.function.NestedIOFunction.forward)

**产品支持情况**：

<!-- npu="910b" id109 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id111 -->

</div>

> <font size="3">forward_extended()</font>

<div style="margin-left: 2em">

**原生文档**：[NestedIOFunction.forward_extended](https://pytorch.org/docs/2.11/generated/torch.autograd.function.NestedIOFunction.html#torch.autograd.function.NestedIOFunction.forward_extended)

**产品支持情况**：

<!-- npu="910b" id112 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id114 -->

</div>

> <font size="3">vmap()</font>

<div style="margin-left: 2em">

**原生文档**：[NestedIOFunction.vmap](https://pytorch.org/docs/2.11/generated/torch.autograd.function.NestedIOFunction.html#torch.autograd.function.NestedIOFunction.vmap)

**产品支持情况**：

<!-- npu="910b" id115 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id117 -->

</div>

</div>

## Profiler

### <code><i>class</i></code> torch.autograd.profiler.EnforceUnique

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.profiler.EnforceUnique](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.EnforceUnique.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.autograd.profiler_util.MemRecordsAcc.in_interval

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.profiler_util.MemRecordsAcc.in_interval](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler_util.MemRecordsAcc.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.autograd.profiler.profile

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.profiler.profile](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.profiler.profile)

**产品支持情况**：

<!-- npu="910b" id118 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id120 -->

**限制与说明**： 采集NPU上的profiling数据时，“`use_device`”需设置为“npu”，例如：

```python
with torch.autograd.profiler.profile(use_device="npu") as prof:
    ...
```

</div>

### profile.export_chrome_trace

<div style="margin-left: 2em">

**原生文档**：[profile.export_chrome_trace](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.profile.export_chrome_trace.html)

**产品支持情况**：

<!-- npu="910b" id121 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id123 -->

</div>

### profile.key_averages

<div style="margin-left: 2em">

**原生文档**：[profile.key_averages](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.profile.key_averages.html)

**产品支持情况**：

<!-- npu="910b" id124 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id126 -->

</div>

### profile.self_cpu_time_total

<div style="margin-left: 2em">

**原生文档**：[profile.self_cpu_time_total](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.profile.self_cpu_time_total.html)

**产品支持情况**：

<!-- npu="910b" id127 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id129 -->

</div>

### profile.total_average

<div style="margin-left: 2em">

**原生文档**：[profile.total_average](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.profile.total_average.html)

**产品支持情况**：

<!-- npu="910b" id130 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id132 -->

</div>

### torch.autograd.profiler.emit_nvtx

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.profiler.emit_nvtx](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.profiler.emit_nvtx)

**产品支持情况**：

<!-- npu="910b" id133 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id135 -->

</div>

### torch.autograd.profiler.emit_itt

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.profiler.emit_itt](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.profiler.emit_itt)

**产品支持情况**：

<!-- npu="910b" id136 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id138 -->

</div>

### torch.autograd.profiler.load_nvprof

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.profiler.load_nvprof](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.load_nvprof.html)

**产品支持情况**：

<!-- npu="910b" id139 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id141 -->

</div>

### torch.autograd.profiler.parse_nvprof_trace

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.profiler.parse_nvprof_trace](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.parse_nvprof_trace.html)

**产品支持情况**：

<!-- npu="910b" id142 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id143 -->
<!-- npu="950" id144 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id144 -->

</div>

### <code><i>class</i></code> torch.autograd.profiler.KinetoStepTracker

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.profiler.KinetoStepTracker](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.KinetoStepTracker.html#torch.autograd.profiler.KinetoStepTracker)

**产品支持情况**：

<!-- npu="910b" id145 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id147 -->

> <font size="3">increment_step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.profiler.KinetoStepTracker.increment_step](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.KinetoStepTracker.html#torch.autograd.profiler.KinetoStepTracker.increment_step)

**产品支持情况**：

<!-- npu="910b" id148 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id150 -->

</div>

> <font size="3">current_step()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.profiler.KinetoStepTracker.current_step](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler.KinetoStepTracker.html#torch.autograd.profiler.KinetoStepTracker.current_step)

**产品支持情况**：

<!-- npu="910b" id151 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id153 -->

</div>

</div>

### <code><i>class</i></code> torch.autograd.profiler_util.StringTable

<div style="margin-left: 2em">

> <font size="3">values()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.profiler_util.StringTable.values](https://pytorch.org/docs/2.11/generated/torch.autograd.profiler_util.StringTable.html#torch.autograd.profiler_util.StringTable.values)

**产品支持情况**：

<!-- npu="910b" id154 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id156 -->

</div>

</div>

## Debugging and anomaly detection

### torch.autograd.detect_anomaly

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.detect_anomaly](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.detect_anomaly)

**产品支持情况**：

<!-- npu="910b" id157 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id159 -->

</div>

### torch.autograd.set_detect_anomaly

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.set_detect_anomaly](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.set_detect_anomaly)

**产品支持情况**：

<!-- npu="910b" id160 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id162 -->

</div>

### torch.autograd.grad_mode.set_multithreading_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.grad_mode.set_multithreading_enabled](https://pytorch.org/docs/2.11/generated/torch.autograd.grad_mode.set_multithreading_enabled.html)

**产品支持情况**：

<!-- npu="910b" id163 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id165 -->

</div>

## Autograd graph

### <code><i>class</i></code> torch.autograd.graph.Node

<div style="margin-left: 2em">

> <font size="3">name()</font>

<div style="margin-left: 2em">

**原生文档**：[Node.name](https://pytorch.org/docs/2.11/generated/torch.autograd.graph.Node.name.html)

**产品支持情况**：

<!-- npu="910b" id166 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id168 -->

</div>

> <font size="3">metadata()</font>

<div style="margin-left: 2em">

**原生文档**：[Node.metadata](https://pytorch.org/docs/2.11/generated/torch.autograd.graph.Node.metadata.html)

**产品支持情况**：

<!-- npu="910b" id169 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id171 -->

</div>

> <font size="3">next_functions()</font>

<div style="margin-left: 2em">

**原生文档**：[Node.next_functions](https://pytorch.org/docs/2.11/generated/torch.autograd.graph.Node.next_functions.html)

**产品支持情况**：

<!-- npu="910b" id172 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id174 -->

</div>

> <font size="3">register_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[Node.register_hook](https://pytorch.org/docs/2.11/generated/torch.autograd.graph.Node.register_hook.html)

**产品支持情况**：

<!-- npu="910b" id175 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id177 -->

</div>

> <font size="3">register_prehook()</font>

<div style="margin-left: 2em">

**原生文档**：[Node.register_prehook](https://pytorch.org/docs/2.11/generated/torch.autograd.graph.Node.register_prehook.html)

**产品支持情况**：

<!-- npu="910b" id178 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id180 -->

</div>

</div>

### torch.autograd.graph.saved_tensors_hooks

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.graph.saved_tensors_hooks](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.graph.saved_tensors_hooks)

**产品支持情况**：

<!-- npu="910b" id181 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id183 -->

</div>

### torch.autograd.graph.save_on_cpu

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.graph.save_on_cpu](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.graph.save_on_cpu)

**产品支持情况**：

<!-- npu="910b" id184 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id186 -->

</div>

### torch.autograd.graph.disable_saved_tensors_hooks

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.graph.disable_saved_tensors_hooks](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.graph.disable_saved_tensors_hooks)

**产品支持情况**：

<!-- npu="910b" id187 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id189 -->

</div>

### torch.autograd.graph.register_multi_grad_hook

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.graph.register_multi_grad_hook](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.graph.register_multi_grad_hook)

**产品支持情况**：

<!-- npu="910b" id190 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id192 -->

</div>

### torch.autograd.graph.allow_mutation_on_saved_tensors

<div style="margin-left: 2em">

**原生文档**：[torch.autograd.graph.allow_mutation_on_saved_tensors](https://pytorch.org/docs/2.11/autograd.html#torch.autograd.graph.allow_mutation_on_saved_tensors)

**产品支持情况**：

<!-- npu="910b" id193 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id195 -->

**限制与说明**： `input`仅支持fp32

</div>
