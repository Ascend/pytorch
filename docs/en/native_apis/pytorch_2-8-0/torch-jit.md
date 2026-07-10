# torch.jit

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:31:53.807Z pushedAt=2026-07-09T08:44:08.288Z -->

> [!NOTE]  
> If the API "Supported" column shows "Yes" and "Restrictions and Notes" shows "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.jit.export|Yes|-|
|torch.jit.is_scripting|Yes|-|
|torch.jit.is_tracing|Yes|Supports fp32|
|torch.jit.script|Yes|-|
|torch.jit.trace|Yes|Supports fp32|
|torch.jit.script_if_tracing|Yes|Supports fp32|
|torch.jit.trace_module|Yes|Supports fp32|
|torch.jit.fork|Yes|-|
|torch.jit.wait|Yes|-|
|torch.jit.ScriptModule|Yes|Supports fp32|
|torch.jit.ScriptModule.add_module|No|-|
|torch.jit.ScriptModule.apply|Yes|Supports fp32|
|torch.jit.ScriptModule.bfloat16|No|-|
|torch.jit.ScriptModule.buffers|Yes|Supports fp32|
|torch.jit.ScriptModule.children|Yes|-|
|torch.jit.ScriptModule.code|No|-|
|torch.jit.ScriptModule.code_with_constants|No|-|
|torch.jit.ScriptModule.compile|No|-|
|torch.jit.ScriptModule.cpu|Yes|Supports fp32|
|torch.jit.ScriptModule.cuda|Yes|Supports fp32|
|torch.jit.ScriptModule.double|Yes|Supports fp32|
|torch.jit.ScriptModule.eval|Yes|Supports fp32|
|torch.jit.ScriptModule.extra_repr|Yes|Supports fp32|
|torch.jit.ScriptModule.float|Yes|Supports fp32|
|torch.jit.ScriptModule.get_buffer|Yes|Supports fp32|
|torch.jit.ScriptModule.get_extra_state|No|-|
|torch.jit.ScriptModule.get_parameter|Yes|Supports fp32|
|torch.jit.ScriptModule.get_submodule|Yes|Supports fp32|
|torch.jit.ScriptModule.graph|No|-|
|torch.jit.ScriptModule.half|Yes|Supports fp32|
|torch.jit.ScriptModule.inlined_graph|No|-|
|torch.jit.ScriptModule.ipu|Yes|Supports fp32|
|torch.jit.ScriptModule.load_state_dict|Yes|Supports fp32|
|torch.jit.ScriptModule.modules|Yes|Supports fp32|
|torch.jit.ScriptModule.named_buffers|Yes|Supports fp32|
|torch.jit.ScriptModule.named_children|Yes|Supports fp32|
|torch.jit.ScriptModule.named_modules|Yes|Supports fp32|
|torch.jit.ScriptModule.named_parameters|Yes|Supports fp32|
|torch.jit.ScriptModule.parameters|Yes|Supports fp32|
|torch.jit.ScriptModule.register_backward_hook|Yes|Supports fp32|
|torch.jit.ScriptModule.register_buffer|No|-|
|torch.jit.ScriptModule.register_forward_hook|Yes|Supports fp32|
|torch.jit.ScriptModule.register_forward_pre_hook|Yes|Supports fp32|
|torch.jit.ScriptModule.register_full_backward_hook|Yes|Supports fp32|
|torch.jit.ScriptModule.register_full_backward_pre_hook|Yes|Supports fp32|
|torch.jit.ScriptModule.register_load_state_dict_post_hook|Yes|Supports fp32|
|torch.jit.ScriptModule.register_module|No|-|
|torch.jit.ScriptModule.register_parameter|No|-|
|torch.jit.ScriptModule.register_state_dict_pre_hook|Yes|Supports fp32|
|torch.jit.ScriptModule.requires_grad_|Yes|Supports fp32|
|torch.jit.ScriptModule.save|Yes|Supports fp32|
|torch.jit.ScriptModule.set_extra_state|No|-|
|torch.jit.ScriptModule.share_memory|No|-|
|torch.jit.ScriptModule.state_dict|Yes|Supports fp32|
|torch.jit.ScriptModule.to|Yes|Supports fp32|
|torch.jit.ScriptModule.to_empty|Yes|Supports fp32|
|torch.jit.ScriptModule.train|Yes|Supports fp32|
|torch.jit.ScriptModule.type|Yes|Supports fp16, fp32, uint8, int8, int16, int32, int64, bool|
|torch.jit.ScriptModule.xpu|Yes|Supports fp32|
|torch.jit.ScriptModule.zero_grad|Yes|Supports fp32|
|torch.jit.ScriptFunction|Yes|Supports fp32|
|torch.jit.ScriptFunction.get_debug_state|Yes|Supports fp32|
|torch.jit.ScriptFunction.save|Yes|Supports fp32|
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
|torch.jit.Attribute|Yes|Supports fp32|
|torch.jit.Attribute.count|Yes|Supports fp32|
|torch.jit.Attribute.index|Yes|Supports fp32|
|torch.jit.Attribute.type|No|-|
|torch.jit.Attribute.value|No|-|
|torch.jit.annotate|Yes|Supports fp32|
