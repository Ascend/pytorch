# torch.jit

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T02:17:29.310Z pushedAt=2026-06-15T03:25:49.184Z -->

> [!NOTE]  
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.jit.script](https://pytorch.org/docs/2.9/generated/torch.jit.script.html)|Yes|-|
|[torch.jit.trace](https://pytorch.org/docs/2.9/generated/torch.jit.trace.html)|Yes|FP32 Supported|
|[torch.jit.script_if_tracing](https://pytorch.org/docs/2.9/generated/torch.jit.script_if_tracing.html)|Yes|FP32 Supported|
|[torch.jit.trace_module](https://pytorch.org/docs/2.9/generated/torch.jit.trace_module.html)|Yes|FP32 Supported|
|[torch.jit.fork](https://pytorch.org/docs/2.9/generated/torch.jit.fork.html)|Yes|-|
|[torch.jit.wait](https://pytorch.org/docs/2.9/generated/torch.jit.wait.html)|Yes|-|
|[torch.jit.ScriptModule](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.add_module](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.add_module)|No|-|
|[torch.jit.ScriptModule.apply](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.apply)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.bfloat16](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.bfloat16)|No|-|
|[torch.jit.ScriptModule.buffers](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.buffers)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.children](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.children)|Yes|-|
|[torch.jit.ScriptModule.code](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.code)|No|-|
|[torch.jit.ScriptModule.code_with_constants](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.code_with_constants)|No|-|
|[torch.jit.ScriptModule.compile](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.compile)|No|-|
|[torch.jit.ScriptModule.cpu](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.cpu)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.cuda](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.cuda)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.double](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.double)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.eval](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.eval)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.extra_repr](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.extra_repr)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.float](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.float)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.get_buffer](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_buffer)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.get_extra_state](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_extra_state)|No|-|
|[torch.jit.ScriptModule.get_parameter](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_parameter)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.get_submodule](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_submodule)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.graph](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.graph)|No|-|
|[torch.jit.ScriptModule.half](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.half)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.inlined_graph](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.inlined_graph)|No|-|
|[torch.jit.ScriptModule.ipu](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.ipu)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.load_state_dict)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.modules](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.modules)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.named_buffers](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_buffers)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.named_children](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_children)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.named_modules](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_modules)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.named_parameters](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_parameters)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.parameters](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.parameters)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.register_backward_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_backward_hook)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.register_buffer](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_buffer)|No|-|
|[torch.jit.ScriptModule.register_forward_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_forward_hook)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.register_forward_pre_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_forward_pre_hook)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.register_full_backward_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_full_backward_hook)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.register_full_backward_pre_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_full_backward_pre_hook)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_load_state_dict_pre_hook)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_load_state_dict_post_hook)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.register_module](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_module)|No|-|
|[torch.jit.ScriptModule.register_parameter](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_parameter)|No|-|
|[torch.jit.ScriptModule.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_state_dict_pre_hook)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.register_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_state_dict_post_hook)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.requires_grad_](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.requires_grad_)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.save](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.save)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.set_extra_state](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.set_extra_state)|No|-|
|[torch.jit.ScriptModule.share_memory](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.share_memory)|No|-|
|[torch.jit.ScriptModule.state_dict](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.state_dict)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.to](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.to)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.to_empty](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.to_empty)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.train](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.train)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.type](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.type)|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|[torch.jit.ScriptModule.xpu](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.xpu)|Yes|FP32 Supported|
|[torch.jit.ScriptModule.zero_grad](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.zero_grad)|Yes|FP32 Supported|
|[torch.jit.ScriptFunction](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptFunction.html)|Yes|FP32 Supported|
|[torch.jit.ScriptFunction.get_debug_state](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction.get_debug_state)|Yes|FP32 Supported|
|[torch.jit.ScriptFunction.save](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction.save)|Yes|FP32 Supported|
|[torch.jit.ScriptFunction.save_to_buffer](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction.save_to_buffer)|Yes|-|
|[torch.jit.freeze](https://pytorch.org/docs/2.9/generated/torch.jit.freeze.html)|Yes|-|
|[torch.jit.optimize_for_inference](https://pytorch.org/docs/2.9/generated/torch.jit.optimize_for_inference.html)|Yes|-|
|[torch.jit.enable_onednn_fusion](https://pytorch.org/docs/2.9/generated/torch.jit.enable_onednn_fusion.html)|Yes|-|
|[torch.jit.onednn_fusion_enabled](https://pytorch.org/docs/2.9/generated/torch.jit.onednn_fusion_enabled.html)|Yes|-|
|[torch.jit.set_fusion_strategy](https://pytorch.org/docs/2.9/generated/torch.jit.set_fusion_strategy.html)|Yes|-|
|[torch.jit.strict_fusion](https://pytorch.org/docs/2.9/generated/torch.jit.strict_fusion.html)|Yes|-|
|[torch.jit.save](https://pytorch.org/docs/2.9/generated/torch.jit.save.html)|Yes|-|
|[torch.jit.load](https://pytorch.org/docs/2.9/generated/torch.jit.load.html)|Yes|-|
|[torch.jit.ignore](https://pytorch.org/docs/2.9/generated/torch.jit.ignore.html)|Yes|-|
|[torch.jit.unused](https://pytorch.org/docs/2.9/generated/torch.jit.unused.html)|Yes|-|
|[torch.jit.isinstance](https://pytorch.org/docs/2.9/generated/torch.jit.isinstance.html)|Yes|-|
|[torch.jit.Attribute](https://pytorch.org/docs/2.9/generated/torch.jit.Attribute.html)|Yes|FP32 Supported|
|[torch.jit.Attribute.count](https://pytorch.org/docs/2.9/generated/torch.jit.Attribute.html#torch.jit.Attribute.count)|Yes|FP32 Supported|
|[torch.jit.Attribute.index](https://pytorch.org/docs/2.9/generated/torch.jit.Attribute.html#torch.jit.Attribute.index)|Yes|FP32 Supported|
|[torch.jit.Attribute.type](https://pytorch.org/docs/2.9/generated/torch.jit.Attribute.html#torch.jit.Attribute.type)|No|-|
|[torch.jit.Attribute.value](https://pytorch.org/docs/2.9/generated/torch.jit.Attribute.html#torch.jit.Attribute.value)|No|-|
|[torch.jit.annotate](https://pytorch.org/docs/2.9/generated/torch.jit.annotate.html)|Yes|FP32 Supported|
