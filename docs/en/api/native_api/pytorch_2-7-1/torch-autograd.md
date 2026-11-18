# torch.autograd

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.7/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.7/autograd.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

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

**Native Documentation**: [torch.autograd.backward](https://pytorch.org/docs/2.7/generated/torch.autograd.backward.html)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id3 -->

**Constraints & Notes**:

- `tensors` only supports bf16, fp16, fp32, fp64
- Does not support sparse tensors.

</div>

### torch.autograd.grad

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.grad](https://pytorch.org/docs/2.7/generated/torch.autograd.grad.html)

**Support Status**:

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id6 -->

</div>

## Forward-mode Automatic Differentiation

### torch.autograd.forward_ad.dual_level

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.forward_ad.dual_level](https://pytorch.org/docs/2.7/generated/torch.autograd.forward_ad.dual_level.html)

**Support Status**:

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id9 -->

</div>

### torch.autograd.forward_ad.make_dual

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.forward_ad.make_dual](https://pytorch.org/docs/2.7/generated/torch.autograd.forward_ad.make_dual.html)

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id12 -->

**Constraints & Notes**: `tensor` only supports fp32.

</div>

### torch.autograd.forward_ad.unpack_dual

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.forward_ad.unpack_dual](https://pytorch.org/docs/2.7/generated/torch.autograd.forward_ad.unpack_dual.html)

**Support Status**:

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id15 -->

**Constraints & Notes**: `tensor` only supports fp32.

</div>

## Functional higher level API

### torch.autograd.functional.jacobian

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.functional.jacobian](https://pytorch.org/docs/2.7/generated/torch.autograd.functional.jacobian.html)

**Support Status**:

<!-- npu="910b" id55 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id57 -->

**Constraints & Notes**: `inputs` only supports fp32.

</div>

### torch.autograd.functional.hessian

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.functional.hessian](https://pytorch.org/docs/2.7/generated/torch.autograd.functional.hessian.html)

**Support Status**:

<!-- npu="910b" id58 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id60 -->

**Constraints & Notes**: `inputs` only supports fp32.

</div>

### torch.autograd.functional.vjp

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.functional.vjp](https://pytorch.org/docs/2.7/generated/torch.autograd.functional.vjp.html)

**Support Status**:

<!-- npu="910b" id61 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id63 -->

**Constraints & Notes**: `inputs` only supports fp32.

</div>

<!-- npu="910b,A3" id206 -->
### torch.autograd.functional.jvp

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.functional.jvp](https://pytorch.org/docs/2.7/generated/torch.autograd.functional.jvp.html)

**Support Status**:
<!-- end id206 -->

<!-- npu="910b" id64 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id65 -->

<!-- npu="910b,A3" id207 -->
**Constraints & Notes**: `inputs` only supports fp32.

</div>
<!-- end id207 -->

<!-- npu="910b,A3" id208 -->
### torch.autograd.functional.vhp

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.functional.vhp](https://pytorch.org/docs/2.7/generated/torch.autograd.functional.vhp.html)

**Support Status**:
<!-- end id208 -->

<!-- npu="910b" id67 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id68 -->

<!-- npu="910b,A3" id209 -->
**Constraints & Notes**: `inputs` only supports fp32.

</div>
<!-- end id209 -->

### torch.autograd.functional.hvp

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.functional.hvp](https://pytorch.org/docs/2.7/generated/torch.autograd.functional.hvp.html)

**Support Status**:

<!-- npu="910b" id70 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id72 -->

**Constraints & Notes**: `inputs` only supports fp32.

</div>

## Function

### <code><i>class</i></code> torch.autograd.Function

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.Function](https://pytorch.org/docs/2.7/autograd.html#torch.autograd.Function)

**Support Status**:

<!-- npu="910b" id73 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id75 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.Function.forward](https://pytorch.org/docs/2.7/generated/torch.autograd.Function.forward.html)

**Support Status**:

<!-- npu="910b" id76 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id78 -->

</div>

> <font size="3">backward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.Function.backward](https://pytorch.org/docs/2.7/generated/torch.autograd.Function.backward.html)

**Support Status**:

<!-- npu="910b" id79 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id81 -->

</div>

<!-- npu="910b,A3" id210 -->
> <font size="3">jvp()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.Function.jvp](https://pytorch.org/docs/2.7/generated/torch.autograd.Function.jvp.html)

**Support Status**:
<!-- end id210 -->

<!-- npu="910b" id82 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id83 -->

<!-- npu="910b,A3" id211 -->
</div>
<!-- end id211 -->

> <font size="3">vmap()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.Function.vmap](https://pytorch.org/docs/2.7/generated/torch.autograd.Function.vmap.html)

**Support Status**:

<!-- npu="910b" id85 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id87 -->

</div>

</div>

## Context method mixins

### <code><i>class</i></code> torch.autograd.function.FunctionCtx

<div style="margin-left: 2em">

> <font size="3">mark_dirty()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.function.FunctionCtx.mark_dirty](https://pytorch.org/docs/2.7/generated/torch.autograd.function.FunctionCtx.mark_dirty.html)

**Support Status**:

<!-- npu="910b" id88 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id90 -->

</div>

> <font size="3">mark_non_differentiable()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.function.FunctionCtx.mark_non_differentiable](https://pytorch.org/docs/2.7/generated/torch.autograd.function.FunctionCtx.mark_non_differentiable.html)

**Support Status**:

<!-- npu="910b" id91 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id93 -->

</div>

> <font size="3">save_for_backward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.function.FunctionCtx.save_for_backward](https://pytorch.org/docs/2.7/generated/torch.autograd.function.FunctionCtx.save_for_backward.html)

**Support Status**:

<!-- npu="910b" id94 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id96 -->

</div>

> <font size="3">set_materialize_grads()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.function.FunctionCtx.set_materialize_grads](https://pytorch.org/docs/2.7/generated/torch.autograd.function.FunctionCtx.set_materialize_grads.html)

**Support Status**:

<!-- npu="910b" id97 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id99 -->

</div>

</div>

## Numerical gradient checking

### <code><i>class</i></code> torch.autograd.function.NestedIOFunction

<div style="margin-left: 2em">

**Native Documentation**: [NestedIOFunction](https://pytorch.org/docs/2.7/generated/torch.autograd.function.NestedIOFunction.html)

**Support Status**:

<!-- npu="910b" id106 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id108 -->

> <font size="3">forward()</font>

<div style="margin-left: 2em">

**Native Documentation**: [NestedIOFunction.forward](https://pytorch.org/docs/2.7/generated/torch.autograd.function.NestedIOFunction.html#torch.autograd.function.NestedIOFunction.forward)

**Support Status**:

<!-- npu="910b" id109 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id111 -->

</div>

> <font size="3">forward_extended()</font>

<div style="margin-left: 2em">

**Native Documentation**: [NestedIOFunction.forward_extended](https://pytorch.org/docs/2.7/generated/torch.autograd.function.NestedIOFunction.html#torch.autograd.function.NestedIOFunction.forward_extended)

**Support Status**:

<!-- npu="910b" id112 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id114 -->

</div>

> <font size="3">mark_dirty()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.function.NestedIOFunction.mark\_dirty](https://docs.pytorch.org/docs/2.7/generated/torch.autograd.function.NestedIOFunction.html#torch.autograd.function.NestedIOFunction.mark_dirty)

**Support Status**:

<!-- npu="910b" id230 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id230 -->
<!-- npu="A3" id231 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id231 -->
<!-- npu="950" id232 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id232 -->

</div>

> <font size="3">vmap()</font>

<div style="margin-left: 2em">

**Native Documentation**: [NestedIOFunction.vmap](https://pytorch.org/docs/2.7/generated/torch.autograd.function.NestedIOFunction.html#torch.autograd.function.NestedIOFunction.vmap)

**Support Status**:

<!-- npu="910b" id115 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id117 -->

</div>

</div>

### torch.autograd.gradcheck.gradcheck

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.gradcheck.gradcheck](https://pytorch.org/docs/2.7/generated/torch.autograd.gradcheck.gradcheck.html)

**Support Status**:

<!-- npu="910b" id100 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id102 -->

</div>

<!-- npu="910b,A3" id212 -->
### torch.autograd.gradcheck.gradgradcheck

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.gradcheck.gradgradcheck](https://pytorch.org/docs/2.7/generated/torch.autograd.gradcheck.gradgradcheck.html)

**Support Status**:
<!-- end id212 -->

<!-- npu="910b" id103 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id104 -->

<!-- npu="910b,A3" id213 -->
</div>
<!-- end id213 -->

## Profiler

### torch.autograd.profiler.profile

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.profiler.profile](https://pytorch.org/docs/2.7/autograd.html#torch.autograd.profiler.profile)

**Support Status**:

<!-- npu="910b" id118 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id120 -->

**Constraints & Notes**: When collecting profiling data on the NPU, set "`use_device`" to "npu", for example:

```python
with torch.autograd.profiler.profile(use_device="npu") as prof:
    ...
```

</div>

### profile.export_chrome_trace

<div style="margin-left: 2em">

**Native Documentation**: [profile.export_chrome_trace](https://pytorch.org/docs/2.7/generated/torch.autograd.profiler.profile.export_chrome_trace.html)

**Support Status**:

<!-- npu="910b" id121 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id123 -->

</div>

### profile.key_averages

<div style="margin-left: 2em">

**Native Documentation**: [profile.key_averages](https://pytorch.org/docs/2.7/generated/torch.autograd.profiler.profile.key_averages.html)

**Support Status**:

<!-- npu="910b" id124 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id126 -->

</div>

### profile.self_cpu_time_total

<div style="margin-left: 2em">

**Native Documentation**: [profile.self_cpu_time_total](https://pytorch.org/docs/2.7/generated/torch.autograd.profiler.profile.self_cpu_time_total.html)

**Support Status**:

<!-- npu="910b" id127 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id129 -->

</div>

### profile.total_average

<div style="margin-left: 2em">

**Native Documentation**: [profile.total_average](https://pytorch.org/docs/2.7/generated/torch.autograd.profiler.profile.total_average.html)

**Support Status**:

<!-- npu="910b" id130 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id132 -->

</div>

### torch.autograd.profiler.parse_nvprof_trace

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.profiler.parse_nvprof_trace](https://pytorch.org/docs/2.7/generated/torch.autograd.profiler.parse_nvprof_trace.html)

**Support Status**:

<!-- npu="910b" id142 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id143 -->
<!-- npu="950" id144 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id144 -->

</div>

### <code><i>class</i></code> torch.autograd.profiler.KinetoStepTracker

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.profiler.KinetoStepTracker](https://pytorch.org/docs/2.7/generated/torch.autograd.profiler.KinetoStepTracker.html#torch.autograd.profiler.KinetoStepTracker)

**Support Status**:

<!-- npu="910b" id145 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id147 -->

> <font size="3">increment_step()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.profiler.KinetoStepTracker.increment_step](https://pytorch.org/docs/2.7/generated/torch.autograd.profiler.KinetoStepTracker.html#torch.autograd.profiler.KinetoStepTracker.increment_step)

**Support Status**:

<!-- npu="910b" id148 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id150 -->

</div>

> <font size="3">current_step()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.profiler.KinetoStepTracker.current_step](https://pytorch.org/docs/2.7/generated/torch.autograd.profiler.KinetoStepTracker.html#torch.autograd.profiler.KinetoStepTracker.current_step)

**Support Status**:

<!-- npu="910b" id151 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id153 -->

</div>

</div>

### <code><i>class</i></code> torch.autograd.profiler_util.StringTable

<div style="margin-left: 2em">

> <font size="3">values()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.profiler_util.StringTable.values](https://pytorch.org/docs/2.7/generated/torch.autograd.profiler_util.StringTable.html#torch.autograd.profiler_util.StringTable.values)

**Support Status**:

<!-- npu="910b" id154 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id156 -->

</div>

</div>

## Debugging and anomaly detection

### torch.autograd.detect_anomaly

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.detect_anomaly](https://pytorch.org/docs/2.7/autograd.html#torch.autograd.detect_anomaly)

**Support Status**:

<!-- npu="910b" id157 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id159 -->

</div>

### torch.autograd.set_detect_anomaly

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.set_detect_anomaly](https://pytorch.org/docs/2.7/autograd.html#torch.autograd.set_detect_anomaly)

**Support Status**:

<!-- npu="910b" id160 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id162 -->

</div>

<!-- npu="910b,A3" id214 -->
### torch.autograd.grad_mode.set_multithreading_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.grad_mode.set_multithreading_enabled](https://pytorch.org/docs/2.7/generated/torch.autograd.grad_mode.set_multithreading_enabled.html)

**Support Status**:
<!-- end id214 -->

<!-- npu="910b" id163 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id164 -->

<!-- npu="910b,A3" id215 -->
</div>
<!-- end id215 -->

## Autograd graph

<!-- npu="910b,A3" id226 -->
### <code><i>class</i></code> torch.autograd.graph.Node

<div style="margin-left: 2em">

<!-- npu="910b,A3" id216 -->
> <font size="3">name()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Node.name](https://pytorch.org/docs/2.7/generated/torch.autograd.graph.Node.name.html)

**Support Status**:
<!-- end id216 -->

<!-- npu="910b" id166 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id167 -->

<!-- npu="910b,A3" id217 -->
</div>
<!-- end id217 -->

<!-- npu="910b,A3" id218 -->
> <font size="3">metadata()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Node.metadata](https://pytorch.org/docs/2.7/generated/torch.autograd.graph.Node.metadata.html)

**Support Status**:
<!-- end id218 -->

<!-- npu="910b" id169 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id170 -->

<!-- npu="910b,A3" id219 -->
</div>
<!-- end id219 -->

<!-- npu="910b,A3" id220 -->
> <font size="3">next_functions()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Node.next_functions](https://pytorch.org/docs/2.7/generated/torch.autograd.graph.Node.next_functions.html)

**Support Status**:
<!-- end id220 -->

<!-- npu="910b" id172 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id173 -->

<!-- npu="910b,A3" id221 -->
</div>
<!-- end id221 -->

<!-- npu="910b,A3" id222 -->
> <font size="3">register_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Node.register_hook](https://pytorch.org/docs/2.7/generated/torch.autograd.graph.Node.register_hook.html)

**Support Status**:
<!-- end id222 -->

<!-- npu="910b" id175 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id176 -->

<!-- npu="910b,A3" id223 -->
</div>
<!-- end id223 -->

<!-- npu="910b,A3" id224 -->
> <font size="3">register_prehook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [Node.register_prehook](https://pytorch.org/docs/2.7/generated/torch.autograd.graph.Node.register_prehook.html)

**Support Status**:
<!-- end id224 -->

<!-- npu="910b" id178 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id179 -->

<!-- npu="910b,A3" id225 -->
</div>
<!-- end id225 -->

</div>
<!-- end id226 -->

### torch.autograd.graph.increment\_version

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.graph.increment\_version](https://docs.pytorch.org/docs/2.7/generated/torch.autograd.graph.increment_version.html#torch.autograd.graph.increment_version)

**Support Status**:

<!-- npu="910b" id227 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id227 -->
<!-- npu="A3" id228 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id228 -->
<!-- npu="950" id229 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id229 -->

</div>

### torch.autograd.graph.saved_tensors_hooks

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.graph.saved_tensors_hooks](https://pytorch.org/docs/2.7/autograd.html#torch.autograd.graph.saved_tensors_hooks)

**Support Status**:

<!-- npu="910b" id181 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id183 -->

</div>

### torch.autograd.graph.save_on_cpu

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.graph.save_on_cpu](https://pytorch.org/docs/2.7/autograd.html#torch.autograd.graph.save_on_cpu)

**Support Status**:

<!-- npu="910b" id184 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id186 -->

</div>

### torch.autograd.graph.disable_saved_tensors_hooks

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.graph.disable_saved_tensors_hooks](https://pytorch.org/docs/2.7/autograd.html#torch.autograd.graph.disable_saved_tensors_hooks)

**Support Status**:

<!-- npu="910b" id187 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id189 -->

</div>

### torch.autograd.graph.register_multi_grad_hook

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.graph.register_multi_grad_hook](https://pytorch.org/docs/2.7/autograd.html#torch.autograd.graph.register_multi_grad_hook)

**Support Status**:

<!-- npu="910b" id190 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id192 -->

</div>

### torch.autograd.graph.allow_mutation_on_saved_tensors

<div style="margin-left: 2em">

**Native Documentation**: [torch.autograd.graph.allow_mutation_on_saved_tensors](https://pytorch.org/docs/2.7/autograd.html#torch.autograd.graph.allow_mutation_on_saved_tensors)

**Support Status**:

<!-- npu="910b" id193 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id195 -->

**Constraints & Notes**: `input` only supports fp32.

</div>
