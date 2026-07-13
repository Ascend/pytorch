# torch.autograd

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-14T07:54:31.428Z pushedAt=2026-06-14T09:16:34.722Z -->

> [!NOTE]
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.autograd.Function|Yes|-|
|torch.autograd.profiler.profile|Yes|When collecting profiling data on NPU, "use_device" must be set to "npu"|
|torch.autograd.profiler.emit_nvtx|No|-|
|torch.autograd.profiler.emit_itt|No|-|
|torch.autograd.detect_anomaly|Yes|-|
|torch.autograd.set_detect_anomaly|Yes|-|
|torch.autograd.graph.saved_tensors_hooks|Yes|-|
|torch.autograd.graph.save_on_cpu|Yes|-|
|torch.autograd.graph.disable_saved_tensors_hooks|Yes|-|
|torch.autograd.graph.register_multi_grad_hook|Yes|-|
|torch.autograd.graph.allow_mutation_on_saved_tensors|Yes|Supports fp32|
|torch.autograd.backward|Yes|Supports bf16, fp16, fp32, fp64<br>Does not support sparse tensors|
|torch.autograd.grad|Yes|-|
|torch.autograd.forward_ad.dual_level|Yes|-|
|torch.autograd.forward_ad.make_dual|Yes|Supports fp32|
|torch.autograd.forward_ad.unpack_dual|Yes|Supports fp32|
|torch.autograd.functional.jacobian|Yes|Supports fp32|
|torch.autograd.functional.hessian|Yes|Supports fp32|
|torch.autograd.functional.vjp|Yes|Supports fp32|
|torch.autograd.functional.jvp|Yes|Supports fp32|
|torch.autograd.functional.vhp|Yes|Supports fp32|
|torch.autograd.functional.hvp|Yes|Supports fp32|
|Function.forward|Yes|-|
|Function.backward|Yes|-|
|Function.jvp|Yes|-|
|Function.vmap|Yes|-|
|FunctionCtx.mark_dirty|Yes|-|
|FunctionCtx.mark_non_differentiable|Yes|-|
|FunctionCtx.save_for_backward|Yes|-|
|FunctionCtx.set_materialize_grads|Yes|-|
|torch.autograd.gradcheck.gradcheck|Yes|-|
|torch.autograd.gradcheck.gradgradcheck|Yes|-|
|profile.export_chrome_trace|Yes|-|
|profile.key_averages|Yes|-|
|profile.self_cpu_time_total|Yes|-|
|profile.total_average|Yes|-|
|torch.autograd.profiler.load_nvprof|No|-|
|torch.autograd.grad_mode.set_multithreading_enabled|Yes|-|
|Node.name|Yes|-|
|Node.metadata|Yes|-|
|Node.next_functions|Yes|-|
|Node.register_hook|Yes|-|
|Node.register_prehook|Yes|-|
