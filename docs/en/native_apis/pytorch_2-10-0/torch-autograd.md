# torch.autograd

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-14T07:54:31.428Z pushedAt=2026-06-14T09:16:34.722Z -->

> [!NOTE]
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.autograd.Function](https://pytorch.org/docs/2.10/autograd.html#torch.autograd.Function)|Yes|-|
|[torch.autograd.profiler.profile](https://pytorch.org/docs/2.10/autograd.html#torch.autograd.profiler.profile)|Yes|When collecting profiling data on NPU, "use_device" must be set to "npu"|
|[torch.autograd.profiler.emit_nvtx](https://pytorch.org/docs/2.10/autograd.html#torch.autograd.profiler.emit_nvtx)|No|-|
|[torch.autograd.profiler.emit_itt](https://pytorch.org/docs/2.10/autograd.html#torch.autograd.profiler.emit_itt)|No|-|
|[torch.autograd.detect_anomaly](https://pytorch.org/docs/2.10/autograd.html#torch.autograd.detect_anomaly)|Yes|-|
|[torch.autograd.set_detect_anomaly](https://pytorch.org/docs/2.10/autograd.html#torch.autograd.set_detect_anomaly)|Yes|-|
|[torch.autograd.graph.saved_tensors_hooks](https://pytorch.org/docs/2.10/autograd.html#torch.autograd.graph.saved_tensors_hooks)|Yes|-|
|[torch.autograd.graph.save_on_cpu](https://pytorch.org/docs/2.10/autograd.html#torch.autograd.graph.save_on_cpu)|Yes|-|
|[torch.autograd.graph.disable_saved_tensors_hooks](https://pytorch.org/docs/2.10/autograd.html#torch.autograd.graph.disable_saved_tensors_hooks)|Yes|-|
|[torch.autograd.graph.register_multi_grad_hook](https://pytorch.org/docs/2.10/autograd.html#torch.autograd.graph.register_multi_grad_hook)|Yes|-|
|[torch.autograd.graph.allow_mutation_on_saved_tensors](https://pytorch.org/docs/2.10/autograd.html#torch.autograd.graph.allow_mutation_on_saved_tensors)|Yes|Supports fp32|
|[torch.autograd.backward](https://pytorch.org/docs/2.10/generated/torch.autograd.backward.html)|Yes|Supports bf16, fp16, fp32, fp64<br>Does not support sparse tensors|
|[torch.autograd.grad](https://pytorch.org/docs/2.10/generated/torch.autograd.grad.html)|Yes|-|
|[torch.autograd.forward_ad.dual_level](https://pytorch.org/docs/2.10/generated/torch.autograd.forward_ad.dual_level.html)|Yes|-|
|[torch.autograd.forward_ad.make_dual](https://pytorch.org/docs/2.10/generated/torch.autograd.forward_ad.make_dual.html)|Yes|Supports fp32|
|[torch.autograd.forward_ad.unpack_dual](https://pytorch.org/docs/2.10/generated/torch.autograd.forward_ad.unpack_dual.html)|Yes|Supports fp32|
|[torch.autograd.functional.jacobian](https://pytorch.org/docs/2.10/generated/torch.autograd.functional.jacobian.html)|Yes|Supports fp32|
|[torch.autograd.functional.hessian](https://pytorch.org/docs/2.10/generated/torch.autograd.functional.hessian.html)|Yes|Supports fp32|
|[torch.autograd.functional.vjp](https://pytorch.org/docs/2.10/generated/torch.autograd.functional.vjp.html)|Yes|Supports fp32|
|[torch.autograd.functional.jvp](https://pytorch.org/docs/2.10/generated/torch.autograd.functional.jvp.html)|Yes|Supports fp32|
|[torch.autograd.functional.vhp](https://pytorch.org/docs/2.10/generated/torch.autograd.functional.vhp.html)|Yes|Supports fp32|
|[torch.autograd.functional.hvp](https://pytorch.org/docs/2.10/generated/torch.autograd.functional.hvp.html)|Yes|Supports fp32|
|[Function.forward](https://pytorch.org/docs/2.10/generated/torch.autograd.Function.forward.html)|Yes|-|
|[Function.backward](https://pytorch.org/docs/2.10/generated/torch.autograd.Function.backward.html)|Yes|-|
|[Function.jvp](https://pytorch.org/docs/2.10/generated/torch.autograd.Function.jvp.html)|Yes|-|
|[Function.vmap](https://pytorch.org/docs/2.10/generated/torch.autograd.Function.vmap.html)|Yes|-|
|[FunctionCtx.mark_dirty](https://pytorch.org/docs/2.10/generated/torch.autograd.function.FunctionCtx.mark_dirty.html)|Yes|-|
|[FunctionCtx.mark_non_differentiable](https://pytorch.org/docs/2.10/generated/torch.autograd.function.FunctionCtx.mark_non_differentiable.html)|Yes|-|
|[FunctionCtx.save_for_backward](https://pytorch.org/docs/2.10/generated/torch.autograd.function.FunctionCtx.save_for_backward.html)|Yes|-|
|[FunctionCtx.set_materialize_grads](https://pytorch.org/docs/2.10/generated/torch.autograd.function.FunctionCtx.set_materialize_grads.html)|Yes|-|
|[torch.autograd.gradcheck.gradcheck](https://pytorch.org/docs/2.10/generated/torch.autograd.gradcheck.gradcheck.html)|Yes|-|
|[torch.autograd.gradcheck.gradgradcheck](https://pytorch.org/docs/2.10/generated/torch.autograd.gradcheck.gradgradcheck.html)|Yes|-|
|[profile.export_chrome_trace](https://pytorch.org/docs/2.10/generated/torch.autograd.profiler.profile.export_chrome_trace.html)|Yes|-|
|[profile.key_averages](https://pytorch.org/docs/2.10/generated/torch.autograd.profiler.profile.key_averages.html)|Yes|-|
|[profile.self_cpu_time_total](https://pytorch.org/docs/2.10/generated/torch.autograd.profiler.profile.self_cpu_time_total.html)|Yes|-|
|[profile.total_average](https://pytorch.org/docs/2.10/generated/torch.autograd.profiler.profile.total_average.html)|Yes|-|
|[torch.autograd.profiler.load_nvprof](https://pytorch.org/docs/2.10/generated/torch.autograd.profiler.load_nvprof.html)|No|-|
|[torch.autograd.grad_mode.set_multithreading_enabled](https://pytorch.org/docs/2.10/generated/torch.autograd.grad_mode.set_multithreading_enabled.html)|Yes|-|
|[Node.name](https://pytorch.org/docs/2.10/generated/torch.autograd.graph.Node.name.html)|Yes|-|
|[Node.metadata](https://pytorch.org/docs/2.10/generated/torch.autograd.graph.Node.metadata.html)|Yes|-|
|[Node.next_functions](https://pytorch.org/docs/2.10/generated/torch.autograd.graph.Node.next_functions.html)|Yes|-|
|[Node.register_hook](https://pytorch.org/docs/2.10/generated/torch.autograd.graph.Node.register_hook.html)|Yes|-|
|[Node.register_prehook](https://pytorch.org/docs/2.10/generated/torch.autograd.graph.Node.register_prehook.html)|Yes|-|
