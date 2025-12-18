# torch.autograd

> [!NOTE]  
> 若API“是否支持“为“是“，“限制与说明“为“-“，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.autograd.Function|是|-|
|torch.autograd.profiler.profile|是|采集NPU上的profiling数据时，“use_device”需设置为“npu”|
|torch.autograd.profiler.emit_nvtx|否|-|
|torch.autograd.profiler.emit_itt|否|-|
|torch.autograd.detect_anomaly|是|-|
|torch.autograd.set_detect_anomaly|是|-|
|torch.autograd.graph.saved_tensors_hooks|是|-|
|torch.autograd.graph.save_on_cpu|是|-|
|torch.autograd.graph.disable_saved_tensors_hooks|是|-|
|torch.autograd.graph.register_multi_grad_hook|是|-|
|torch.autograd.graph.allow_mutation_on_saved_tensors|是|支持fp32|
|torch.autograd.backward|是|支持bf16，fp16，fp32，fp64<br>不支持稀疏张量|
|torch.autograd.grad|是|-|
|torch.autograd.forward_ad.dual_level|是|-|
|torch.autograd.forward_ad.make_dual|是|支持fp32|
|torch.autograd.forward_ad.unpack_dual|是|支持fp32|
|torch.autograd.functional.jacobian|是|支持fp32|
|torch.autograd.functional.hessian|是|支持fp32|
|torch.autograd.functional.vjp|是|支持fp32|
|torch.autograd.functional.jvp|是|支持fp32|
|torch.autograd.functional.vhp|是|支持fp32|
|torch.autograd.functional.hvp|是|支持fp32|
|Function.forward|是|-|
|Function.backward|是|-|
|Function.jvp|是|-|
|Function.vmap|是|-|
|FunctionCtx.mark_dirty|是|-|
|FunctionCtx.mark_non_differentiable|是|-|
|FunctionCtx.save_for_backward|是|-|
|FunctionCtx.set_materialize_grads|是|-|
|torch.autograd.gradcheck.gradcheck|是|-|
|torch.autograd.gradcheck.gradgradcheck|是|-|
|profile.export_chrome_trace|是|-|
|profile.key_averages|是|-|
|profile.self_cpu_time_total|是|-|
|profile.total_average|是|-|
|torch.autograd.profiler.load_nvprof|否|-|
|torch.autograd.grad_mode.set_multithreading_enabled|是|-|
|Node.name|是|-|
|Node.metadata|是|-|
|Node.next_functions|是|-|
|Node.register_hook|是|-|
|Node.register_prehook|是|-|


