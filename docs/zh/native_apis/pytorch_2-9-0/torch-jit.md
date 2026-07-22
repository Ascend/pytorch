# torch.jit

> [!NOTE]  
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|[torch.jit.script](https://pytorch.org/docs/2.9/generated/torch.jit.script.html)|是|-|
|[torch.jit.trace](https://pytorch.org/docs/2.9/generated/torch.jit.trace.html)|是|支持fp32|
|[torch.jit.script_if_tracing](https://pytorch.org/docs/2.9/generated/torch.jit.script_if_tracing.html)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.trace_module](https://pytorch.org/docs/2.9/generated/torch.jit.trace_module.html)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.fork](https://pytorch.org/docs/2.9/generated/torch.jit.fork.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.jit.wait](https://pytorch.org/docs/2.9/generated/torch.jit.wait.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.jit.ScriptModule](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.add_module](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.add_module)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.jit.ScriptModule.apply](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.apply)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.bfloat16](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.bfloat16)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.jit.ScriptModule.buffers](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.buffers)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.children](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.children)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.jit.ScriptModule.code](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.code)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.jit.ScriptModule.code_with_constants](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.code_with_constants)|否|-|
|[torch.jit.ScriptModule.compile](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.compile)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.jit.ScriptModule.cpu](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.cpu)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.cuda](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.cuda)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.double](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.double)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.eval](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.eval)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.extra_repr](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.extra_repr)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.float](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.float)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.get_buffer](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_buffer)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.get_extra_state](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_extra_state)|否|-|
|[torch.jit.ScriptModule.get_parameter](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_parameter)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.get_submodule](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_submodule)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.graph](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.graph)|否|-|
|[torch.jit.ScriptModule.half](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.half)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.inlined_graph](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.inlined_graph)|否|-|
|[torch.jit.ScriptModule.ipu](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.ipu)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.load_state_dict)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.modules](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.modules)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.named_buffers](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_buffers)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.named_children](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_children)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.named_modules](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_modules)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.named_parameters](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_parameters)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.parameters](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.parameters)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.register_backward_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_backward_hook)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.register_buffer](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_buffer)|否|-|
|[torch.jit.ScriptModule.register_forward_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_forward_hook)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.register_forward_pre_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_forward_pre_hook)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.register_full_backward_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_full_backward_hook)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.register_full_backward_pre_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_full_backward_pre_hook)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_load_state_dict_pre_hook)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_load_state_dict_post_hook)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.register_module](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_module)|否|-|
|[torch.jit.ScriptModule.register_parameter](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_parameter)|否|-|
|[torch.jit.ScriptModule.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_state_dict_pre_hook)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.register_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_state_dict_post_hook)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.requires_grad_](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.requires_grad_)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.save](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.save)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.set_extra_state](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.set_extra_state)|否|-|
|[torch.jit.ScriptModule.set_submodule](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.set_submodule)|否|-|
|[torch.jit.ScriptModule.share_memory](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.share_memory)|否|-|
|[torch.jit.ScriptModule.state_dict](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.state_dict)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.to](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.to)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.to_empty](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.to_empty)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.train](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.train)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.type](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.type)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp16，fp32，uint8，int8，int16，int32，int64，bool|
|[torch.jit.ScriptModule.xpu](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.xpu)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptModule.zero_grad](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.zero_grad)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptFunction](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptFunction.html)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptFunction.get_debug_state](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction.get_debug_state)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptFunction.save](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction.save)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.ScriptFunction.save_to_buffer](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction.save_to_buffer)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.jit.freeze](https://pytorch.org/docs/2.9/generated/torch.jit.freeze.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.jit.optimize_for_inference](https://pytorch.org/docs/2.9/generated/torch.jit.optimize_for_inference.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.jit.enable_onednn_fusion](https://pytorch.org/docs/2.9/generated/torch.jit.enable_onednn_fusion.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.jit.onednn_fusion_enabled](https://pytorch.org/docs/2.9/generated/torch.jit.onednn_fusion_enabled.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.jit.set_fusion_strategy](https://pytorch.org/docs/2.9/generated/torch.jit.set_fusion_strategy.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.jit.strict_fusion](https://pytorch.org/docs/2.9/generated/torch.jit.strict_fusion.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.jit.save](https://pytorch.org/docs/2.9/generated/torch.jit.save.html)|是|-|
|[torch.jit.load](https://pytorch.org/docs/2.9/generated/torch.jit.load.html)|是|-|
|[torch.jit.ignore](https://pytorch.org/docs/2.9/generated/torch.jit.ignore.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.jit.unused](https://pytorch.org/docs/2.9/generated/torch.jit.unused.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.jit.isinstance](https://pytorch.org/docs/2.9/generated/torch.jit.isinstance.html)|是<br>暂不支持<term>Ascend 950DT</term>|-|
|[torch.jit.Attribute](https://pytorch.org/docs/2.9/generated/torch.jit.Attribute.html)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.Attribute.count](https://pytorch.org/docs/2.9/generated/torch.jit.Attribute.html#torch.jit.Attribute.count)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.Attribute.index](https://pytorch.org/docs/2.9/generated/torch.jit.Attribute.html#torch.jit.Attribute.index)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|[torch.jit.Attribute.type](https://pytorch.org/docs/2.9/generated/torch.jit.Attribute.html#torch.jit.Attribute.type)|否|-|
|[torch.jit.Attribute.value](https://pytorch.org/docs/2.9/generated/torch.jit.Attribute.html#torch.jit.Attribute.value)|否|-|
|[torch.jit.annotate](https://pytorch.org/docs/2.9/generated/torch.jit.annotate.html)|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
