# torch.jit

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T02:17:29.310Z pushedAt=2026-06-15T03:25:49.184Z -->

> [!NOTE]  
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.jit.script|Yes|-|
|torch.jit.trace|Yes|FP32 Supported|
|torch.jit.script_if_tracing|Yes|FP32 Supported|
|torch.jit.trace_module|Yes|FP32 Supported|
|torch.jit.fork|Yes|-|
|torch.jit.wait|Yes|-|
|torch.jit.ScriptModule|Yes|FP32 Supported|
|torch.jit.ScriptModule.add_module|No|-|
|torch.jit.ScriptModule.apply|Yes|FP32 Supported|
|torch.jit.ScriptModule.bfloat16|No|-|
|torch.jit.ScriptModule.buffers|Yes|FP32 Supported|
|torch.jit.ScriptModule.children|Yes|-|
|torch.jit.ScriptModule.code|No|-|
|torch.jit.ScriptModule.code_with_constants|No|-|
|torch.jit.ScriptModule.compile|No|-|
|torch.jit.ScriptModule.cpu|Yes|FP32 Supported|
|torch.jit.ScriptModule.cuda|Yes|FP32 Supported|
|torch.jit.ScriptModule.double|Yes|FP32 Supported|
|torch.jit.ScriptModule.eval|Yes|FP32 Supported|
|torch.jit.ScriptModule.extra_repr|Yes|FP32 Supported|
|torch.jit.ScriptModule.float|Yes|FP32 Supported|
|torch.jit.ScriptModule.get_buffer|Yes|FP32 Supported|
|torch.jit.ScriptModule.get_extra_state|No|-|
|torch.jit.ScriptModule.get_parameter|Yes|FP32 Supported|
|torch.jit.ScriptModule.get_submodule|Yes|FP32 Supported|
|torch.jit.ScriptModule.graph|No|-|
|torch.jit.ScriptModule.half|Yes|FP32 Supported|
|torch.jit.ScriptModule.inlined_graph|No|-|
|torch.jit.ScriptModule.ipu|Yes|FP32 Supported|
|torch.jit.ScriptModule.load_state_dict|Yes|FP32 Supported|
|torch.jit.ScriptModule.modules|Yes|FP32 Supported|
|torch.jit.ScriptModule.named_buffers|Yes|FP32 Supported|
|torch.jit.ScriptModule.named_children|Yes|FP32 Supported|
|torch.jit.ScriptModule.named_modules|Yes|FP32 Supported|
|torch.jit.ScriptModule.named_parameters|Yes|FP32 Supported|
|torch.jit.ScriptModule.parameters|Yes|FP32 Supported|
|torch.jit.ScriptModule.register_backward_hook|Yes|FP32 Supported|
|torch.jit.ScriptModule.register_buffer|No|-|
|torch.jit.ScriptModule.register_forward_hook|Yes|FP32 Supported|
|torch.jit.ScriptModule.register_forward_pre_hook|Yes|FP32 Supported|
|torch.jit.ScriptModule.register_full_backward_hook|Yes|FP32 Supported|
|torch.jit.ScriptModule.register_full_backward_pre_hook|Yes|FP32 Supported|
|torch.jit.ScriptModule.register_load_state_dict_pre_hook|Yes|FP32 Supported|
|torch.jit.ScriptModule.register_load_state_dict_post_hook|Yes|FP32 Supported|
|torch.jit.ScriptModule.register_module|No|-|
|torch.jit.ScriptModule.register_parameter|No|-|
|torch.jit.ScriptModule.register_state_dict_pre_hook|Yes|FP32 Supported|
|torch.jit.ScriptModule.register_state_dict_post_hook|Yes|FP32 Supported|
|torch.jit.ScriptModule.requires_grad_|Yes|FP32 Supported|
|torch.jit.ScriptModule.save|Yes|FP32 Supported|
|torch.jit.ScriptModule.set_extra_state|No|-|
|torch.jit.ScriptModule.share_memory|No|-|
|torch.jit.ScriptModule.state_dict|Yes|FP32 Supported|
|torch.jit.ScriptModule.to|Yes|FP32 Supported|
|torch.jit.ScriptModule.to_empty|Yes|FP32 Supported|
|torch.jit.ScriptModule.train|Yes|FP32 Supported|
|torch.jit.ScriptModule.type|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.jit.ScriptModule.xpu|Yes|FP32 Supported|
|torch.jit.ScriptModule.zero_grad|Yes|FP32 Supported|
|torch.jit.ScriptFunction|Yes|FP32 Supported|
|torch.jit.ScriptFunction.get_debug_state|Yes|FP32 Supported|
|torch.jit.ScriptFunction.save|Yes|FP32 Supported|
|torch.jit.ScriptFunction.save_to_buffer|Yes|-|
|torch.jit.freeze|Yes|-|
|torch.jit.optimize_for_inference|Yes|-|
|torch.jit.enable_onednn_fusion|Yes|-|
|torch.jit.onednn_fusion_enabled|Yes|-|
|torch.jit.set_fusion_strategy|Yes|-|
|torch.jit.strict_fusion|Yes|-|
|torch.jit.save|Yes|-|
|torch.jit.load|Yes|-|
|torch.jit.ignore|Yes|-|
|torch.jit.unused|Yes|-|
|torch.jit.isinstance|Yes|-|
|torch.jit.Attribute|Yes|FP32 Supported|
|torch.jit.Attribute.count|Yes|FP32 Supported|
|torch.jit.Attribute.index|Yes|FP32 Supported|
|torch.jit.Attribute.type|No|-|
|torch.jit.Attribute.value|No|-|
|torch.jit.annotate|Yes|FP32 Supported|
