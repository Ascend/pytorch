# torch.jit

> [!NOTE]  
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.jit.export|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.jit.is_scripting|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.jit.is_tracing|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.jit.script|是|-|
|torch.jit.trace|是|支持fp32|
|torch.jit.script_if_tracing|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.trace_module|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.fork|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.jit.wait|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.jit.ScriptModule|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.add_module|否|-|
|torch.jit.ScriptModule.apply|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.bfloat16|否|-|
|torch.jit.ScriptModule.buffers|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.children|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.jit.ScriptModule.code|否|-|
|torch.jit.ScriptModule.code_with_constants|否|-|
|torch.jit.ScriptModule.compile|否|-|
|torch.jit.ScriptModule.cpu|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.cuda|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.double|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.eval|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.extra_repr|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.float|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.get_buffer|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.get_extra_state|否|-|
|torch.jit.ScriptModule.get_parameter|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.get_submodule|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.graph|否|-|
|torch.jit.ScriptModule.half|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.inlined_graph|否|-|
|torch.jit.ScriptModule.ipu|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.load_state_dict|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.modules|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.named_buffers|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.named_children|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.named_modules|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.named_parameters|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.parameters|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.register_backward_hook|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.register_buffer|否|-|
|torch.jit.ScriptModule.register_forward_hook|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.register_forward_pre_hook|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.register_full_backward_hook|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.register_full_backward_pre_hook|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.register_load_state_dict_pre_hook|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.register_load_state_dict_post_hook|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.register_module|否|-|
|torch.jit.ScriptModule.register_parameter|否|-|
|torch.jit.ScriptModule.register_state_dict_pre_hook|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.register_state_dict_post_hook|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.requires_grad_|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.save|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.set_extra_state|否|-|
|torch.jit.ScriptModule.share_memory|否|-|
|torch.jit.ScriptModule.state_dict|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.to|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.to_empty|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.train|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.type|是<br>暂不支持<term>Ascend 950DT</term>|支持fp16，fp32，uint8，int8，int16，int32，int64，bool|
|torch.jit.ScriptModule.xpu|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptModule.zero_grad|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptFunction|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptFunction.get_debug_state|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptFunction.save|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.ScriptFunction.save_to_buffer|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.jit.freeze|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.jit.optimize_for_inference|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.jit.enable_onednn_fusion|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.jit.onednn_fusion_enabled|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.jit.set_fusion_strategy|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.jit.strict_fusion|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.jit.save|是|-|
|torch.jit.load|是|-|
|torch.jit.ignore|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.jit.unused|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.jit.isinstance|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.jit.Attribute|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.Attribute.count|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.Attribute.index|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.jit.Attribute.type|否|-|
|torch.jit.Attribute.value|否|-|
|torch.jit.annotate|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
