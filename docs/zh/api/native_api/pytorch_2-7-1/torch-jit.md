# torch.jit

> [!NOTE]  
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)
- [Creating TorchScript Code](#creating-torchscript-code)

## base API

### torch.jit.export

<div style="margin-left: 2em">

**原生文档**：[torch.jit.export](https://pytorch.org/docs/2.7/jit.html#torch.jit.export)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.jit.is_scripting

<div style="margin-left: 2em">

**原生文档**：[torch.jit.is_scripting](https://docs.pytorch.org/docs/2.7/jit_language_reference.html#torch.jit.is_scripting)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.jit.is_tracing

<div style="margin-left: 2em">

**原生文档**：[torch.jit.is_tracing](https://docs.pytorch.org/docs/2.7/jit_language_reference.html#torch.jit.is_tracing)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## Creating TorchScript Code

### torch.jit.script

<div style="margin-left: 2em">

**原生文档**：[torch.jit.script](https://pytorch.org/docs/2.7/generated/torch.jit.script.html)

**是否支持**：是

</div>

### torch.jit.trace

<div style="margin-left: 2em">

**原生文档**：[torch.jit.trace](https://pytorch.org/docs/2.7/generated/torch.jit.trace.html)

**是否支持**：是

**限制与说明**： 支持fp32

</div>

### torch.jit.script_if_tracing

<div style="margin-left: 2em">

**原生文档**：[torch.jit.script_if_tracing](https://pytorch.org/docs/2.7/generated/torch.jit.script_if_tracing.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.jit.trace_module

<div style="margin-left: 2em">

**原生文档**：[torch.jit.trace_module](https://pytorch.org/docs/2.7/generated/torch.jit.trace_module.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.jit.fork

<div style="margin-left: 2em">

**原生文档**：[torch.jit.fork](https://pytorch.org/docs/2.7/generated/torch.jit.fork.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.jit.wait

<div style="margin-left: 2em">

**原生文档**：[torch.jit.wait](https://pytorch.org/docs/2.7/generated/torch.jit.wait.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.jit.ScriptModule

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

> <font size="3">add_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.add_module](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.add_module)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.apply](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.apply)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">bfloat16()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.bfloat16](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.bfloat16)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">buffers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.buffers](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.buffers)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">children()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.children](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.children)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">code()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.code](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.code)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">code_with_constants()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.code_with_constants](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.code_with_constants)

**是否支持**：否

</div>

> <font size="3">compile()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.compile](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.compile)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">cpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.cpu](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.cpu)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">cuda()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.cuda](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.cuda)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">double()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.double](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.double)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">eval()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.eval](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.eval)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">extra_repr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.extra_repr](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.extra_repr)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">float()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.float](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.float)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">get_buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.get_buffer](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_buffer)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">get_extra_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.get_extra_state](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_extra_state)

**是否支持**：否

</div>

> <font size="3">get_parameter()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.get_parameter](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_parameter)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">get_submodule()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.get_submodule](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_submodule)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">graph()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.graph](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.graph)

**是否支持**：否

</div>

> <font size="3">half()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.half](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.half)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">inlined_graph()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.inlined_graph](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.inlined_graph)

**是否支持**：否

</div>

> <font size="3">ipu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.ipu](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.ipu)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.load_state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.modules](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.modules)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">named_buffers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.named_buffers](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_buffers)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">named_children()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.named_children](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_children)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">named_modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.named_modules](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_modules)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">named_parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.named_parameters](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_parameters)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.parameters](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.parameters)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">register_backward_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_backward_hook](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_backward_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">register_buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_buffer](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_buffer)

**是否支持**：否

</div>

> <font size="3">register_forward_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_forward_hook](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_forward_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">register_forward_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_forward_pre_hook](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_forward_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">register_full_backward_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_full_backward_hook](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_full_backward_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">register_full_backward_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_full_backward_pre_hook](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_full_backward_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_load_state_dict_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_load_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_load_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">register_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_module](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_module)

**是否支持**：否

</div>

> <font size="3">register_parameter()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_parameter](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_parameter)

**是否支持**：否

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_state_dict_pre_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_state_dict_post_hook)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">requires_grad_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.requires_grad_](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.requires_grad_)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">save()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.save](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.save)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">set_extra_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.set_extra_state](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.set_extra_state)

**是否支持**：否

</div>

> <font size="3">set_submodule()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.set_submodule](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.set_submodule)

**是否支持**：否

</div>

> <font size="3">share_memory()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.share_memory](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.share_memory)

**是否支持**：否

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.state_dict](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.state_dict)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">to()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.to](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.to)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">to_empty()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.to_empty](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.to_empty)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">train()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.train](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.train)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.type](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.type)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">xpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.xpu](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.xpu)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.zero_grad](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.zero_grad)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

</div>

### _`class`_ torch.jit.ScriptFunction

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptFunction](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptFunction.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

> <font size="3">get_debug_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptFunction.get_debug_state](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction.get_debug_state)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">save()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptFunction.save](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction.save)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">save_to_buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptFunction.save_to_buffer](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction.save_to_buffer)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### torch.jit.freeze

<div style="margin-left: 2em">

**原生文档**：[torch.jit.freeze](https://pytorch.org/docs/2.7/generated/torch.jit.freeze.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.jit.optimize_for_inference

<div style="margin-left: 2em">

**原生文档**：[torch.jit.optimize_for_inference](https://pytorch.org/docs/2.7/generated/torch.jit.optimize_for_inference.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.jit.enable_onednn_fusion

<div style="margin-left: 2em">

**原生文档**：[torch.jit.enable_onednn_fusion](https://pytorch.org/docs/2.7/generated/torch.jit.enable_onednn_fusion.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.jit.onednn_fusion_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.jit.onednn_fusion_enabled](https://pytorch.org/docs/2.7/generated/torch.jit.onednn_fusion_enabled.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.jit.set_fusion_strategy

<div style="margin-left: 2em">

**原生文档**：[torch.jit.set_fusion_strategy](https://pytorch.org/docs/2.7/generated/torch.jit.set_fusion_strategy.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.jit.strict_fusion

<div style="margin-left: 2em">

**原生文档**：[torch.jit.strict_fusion](https://pytorch.org/docs/2.7/generated/torch.jit.strict_fusion.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.jit.save

<div style="margin-left: 2em">

**原生文档**：[torch.jit.save](https://pytorch.org/docs/2.7/generated/torch.jit.save.html)

**是否支持**：是

</div>

### torch.jit.load

<div style="margin-left: 2em">

**原生文档**：[torch.jit.load](https://pytorch.org/docs/2.7/generated/torch.jit.load.html)

**是否支持**：是

</div>

### torch.jit.ignore

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ignore](https://pytorch.org/docs/2.7/generated/torch.jit.ignore.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.jit.unused

<div style="margin-left: 2em">

**原生文档**：[torch.jit.unused](https://pytorch.org/docs/2.7/generated/torch.jit.unused.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.jit.isinstance

<div style="margin-left: 2em">

**原生文档**：[torch.jit.isinstance](https://pytorch.org/docs/2.7/generated/torch.jit.isinstance.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.jit.Attribute

<div style="margin-left: 2em">

**原生文档**：[torch.jit.Attribute](https://pytorch.org/docs/2.7/generated/torch.jit.Attribute.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

> <font size="3">count()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.Attribute.count](https://pytorch.org/docs/2.7/generated/torch.jit.Attribute.html#torch.jit.Attribute.count)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">index()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.Attribute.index](https://pytorch.org/docs/2.7/generated/torch.jit.Attribute.html#torch.jit.Attribute.index)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.Attribute.type](https://pytorch.org/docs/2.7/generated/torch.jit.Attribute.html#torch.jit.Attribute.type)

**是否支持**：否

</div>

> <font size="3">value()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.Attribute.value](https://pytorch.org/docs/2.7/generated/torch.jit.Attribute.html#torch.jit.Attribute.value)

**是否支持**：否

</div>

</div>

### torch.jit.annotate

<div style="margin-left: 2em">

**原生文档**：[torch.jit.annotate](https://pytorch.org/docs/2.7/generated/torch.jit.annotate.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>
