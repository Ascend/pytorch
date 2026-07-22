# torch.autograd

> [!NOTE]  
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|[torch.autograd.Function](https://pytorch.org/docs/2.12/autograd.html#torch.autograd.Function)|是|-|
|[torch.autograd.profiler.profile](https://pytorch.org/docs/2.12/autograd.html#torch.autograd.profiler.profile)|是<br>暂不支持<term>Ascend 950DT</term>|采集NPU上的profiling数据时，“use_device”需设置为“npu”|
|[torch.autograd.profiler.emit_nvtx](https://pytorch.org/docs/2.12/autograd.html#torch.autograd.profiler.emit_nvtx)|否|-|
|[torch.autograd.profiler.emit_itt](https://pytorch.org/docs/2.12/autograd.html#torch.autograd.profiler.emit_itt)|否|-|
|[torch.autograd.detect_anomaly](https://pytorch.org/docs/2.12/autograd.html#torch.autograd.detect_anomaly)|是|-|
|[torch.autograd.set_detect_anomaly](https://pytorch.org/docs/2.12/autograd.html#torch.autograd.set_detect_anomaly)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.autograd.graph.saved_tensors_hooks](https://pytorch.org/docs/2.12/autograd.html#torch.autograd.graph.saved_tensors_hooks)|是|-|
|[torch.autograd.graph.save_on_cpu](https://pytorch.org/docs/2.12/autograd.html#torch.autograd.graph.save_on_cpu)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.autograd.graph.disable_saved_tensors_hooks](https://pytorch.org/docs/2.12/autograd.html#torch.autograd.graph.disable_saved_tensors_hooks)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.autograd.graph.register_multi_grad_hook](https://pytorch.org/docs/2.12/autograd.html#torch.autograd.graph.register_multi_grad_hook)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.autograd.graph.allow_mutation_on_saved_tensors](https://pytorch.org/docs/2.12/autograd.html#torch.autograd.graph.allow_mutation_on_saved_tensors)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.autograd.backward](https://pytorch.org/docs/2.12/generated/torch.autograd.backward.html)|是|支持bf16，fp16，fp32，fp64<br>不支持稀疏张量|
|[torch.autograd.grad](https://pytorch.org/docs/2.12/generated/torch.autograd.grad.html)|是|-|
|[torch.autograd.forward_ad.dual_level](https://pytorch.org/docs/2.12/generated/torch.autograd.forward_ad.dual_level.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.autograd.forward_ad.make_dual](https://pytorch.org/docs/2.12/generated/torch.autograd.forward_ad.make_dual.html)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.autograd.forward_ad.unpack_dual](https://pytorch.org/docs/2.12/generated/torch.autograd.forward_ad.unpack_dual.html)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.autograd.functional.jacobian](https://pytorch.org/docs/2.12/generated/torch.autograd.functional.jacobian.html)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.autograd.functional.hessian](https://pytorch.org/docs/2.12/generated/torch.autograd.functional.hessian.html)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.autograd.functional.vjp](https://pytorch.org/docs/2.12/generated/torch.autograd.functional.vjp.html)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.autograd.functional.jvp](https://pytorch.org/docs/2.12/generated/torch.autograd.functional.jvp.html)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.autograd.functional.vhp](https://pytorch.org/docs/2.12/generated/torch.autograd.functional.vhp.html)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.autograd.functional.hvp](https://pytorch.org/docs/2.12/generated/torch.autograd.functional.hvp.html)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[Function.forward](https://pytorch.org/docs/2.12/generated/torch.autograd.Function.forward.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[Function.backward](https://pytorch.org/docs/2.12/generated/torch.autograd.Function.backward.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[Function.jvp](https://pytorch.org/docs/2.12/generated/torch.autograd.Function.jvp.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[Function.vmap](https://pytorch.org/docs/2.12/generated/torch.autograd.Function.vmap.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[FunctionCtx.mark_dirty](https://pytorch.org/docs/2.12/generated/torch.autograd.function.FunctionCtx.mark_dirty.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[FunctionCtx.mark_non_differentiable](https://pytorch.org/docs/2.12/generated/torch.autograd.function.FunctionCtx.mark_non_differentiable.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[FunctionCtx.save_for_backward](https://pytorch.org/docs/2.12/generated/torch.autograd.function.FunctionCtx.save_for_backward.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[FunctionCtx.set_materialize_grads](https://pytorch.org/docs/2.12/generated/torch.autograd.function.FunctionCtx.set_materialize_grads.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.autograd.gradcheck.gradcheck](https://pytorch.org/docs/2.12/generated/torch.autograd.gradcheck.gradcheck.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.autograd.gradcheck.gradgradcheck](https://pytorch.org/docs/2.12/generated/torch.autograd.gradcheck.gradgradcheck.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[profile.export_chrome_trace](https://pytorch.org/docs/2.12/generated/torch.autograd.profiler.profile.export_chrome_trace.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[profile.key_averages](https://pytorch.org/docs/2.12/generated/torch.autograd.profiler.profile.key_averages.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[profile.self_cpu_time_total](https://pytorch.org/docs/2.12/generated/torch.autograd.profiler.profile.self_cpu_time_total.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[profile.total_average](https://pytorch.org/docs/2.12/generated/torch.autograd.profiler.profile.total_average.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.autograd.profiler.load_nvprof](https://pytorch.org/docs/2.12/generated/torch.autograd.profiler.load_nvprof.html)|否|-|
|[torch.autograd.grad_mode.set_multithreading_enabled](https://pytorch.org/docs/2.12/generated/torch.autograd.grad_mode.set_multithreading_enabled.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[Node.name](https://pytorch.org/docs/2.12/generated/torch.autograd.graph.Node.name.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[Node.metadata](https://pytorch.org/docs/2.12/generated/torch.autograd.graph.Node.metadata.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[Node.next_functions](https://pytorch.org/docs/2.12/generated/torch.autograd.graph.Node.next_functions.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[Node.register_hook](https://pytorch.org/docs/2.12/generated/torch.autograd.graph.Node.register_hook.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[Node.register_prehook](https://pytorch.org/docs/2.12/generated/torch.autograd.graph.Node.register_prehook.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
