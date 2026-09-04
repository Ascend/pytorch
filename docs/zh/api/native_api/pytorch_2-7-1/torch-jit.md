# torch.jit

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.7/jit.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Creating TorchScript Code](#creating-torchscript-code)
- [Appendix](#appendix)

</div>

<div style="display:none;">

## &#8203;torch.jit

</div>

### torch.jit.is_scripting

<div style="margin-left: 2em">

**原生文档**：[torch.jit.is_scripting](https://docs.pytorch.org/docs/2.7/jit_language_reference.html#torch.jit.is_scripting)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.jit.is_tracing

<div style="margin-left: 2em">

**原生文档**：[torch.jit.is_tracing](https://docs.pytorch.org/docs/2.7/jit_language_reference.html#torch.jit.is_tracing)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

## Creating TorchScript Code

### torch.jit.script

<div style="margin-left: 2em">

**原生文档**：[torch.jit.script](https://pytorch.org/docs/2.7/generated/torch.jit.script.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.jit.trace

<div style="margin-left: 2em">

**原生文档**：[torch.jit.trace](https://pytorch.org/docs/2.7/generated/torch.jit.trace.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

**限制与说明**：`input`仅支持fp32

</div>

### torch.jit.script_if_tracing

<div style="margin-left: 2em">

**原生文档**：[torch.jit.script_if_tracing](https://pytorch.org/docs/2.7/generated/torch.jit.script_if_tracing.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**： `input`仅支持fp32

</div>

### torch.jit.trace_module

<div style="margin-left: 2em">

**原生文档**：[torch.jit.trace_module](https://pytorch.org/docs/2.7/generated/torch.jit.trace_module.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

### torch.jit.fork

<div style="margin-left: 2em">

**原生文档**：[torch.jit.fork](https://pytorch.org/docs/2.7/generated/torch.jit.fork.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.jit.wait

<div style="margin-left: 2em">

**原生文档**：[torch.jit.wait](https://pytorch.org/docs/2.7/generated/torch.jit.wait.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.jit.ScriptModule

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

> <font size="3">add_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.add_module](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.add_module)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.apply](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.apply)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">bfloat16()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.bfloat16](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.bfloat16)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">buffers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.buffers](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.buffers)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">children()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.children](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.children)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">code()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.code](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.code)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">code_with_constants()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.code_with_constants](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.code_with_constants)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">compile()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.compile](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.compile)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">cpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.cpu](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.cpu)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">cuda()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.cuda](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.cuda)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">double()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.double](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.double)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">eval()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.eval](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.eval)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">extra_repr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.extra_repr](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.extra_repr)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">float()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.float](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.float)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">get_buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.get_buffer](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_buffer)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">get_extra_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.get_extra_state](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_extra_state)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">get_parameter()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.get_parameter](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_parameter)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">get_submodule()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.get_submodule](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_submodule)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">graph()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.graph](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.graph)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">half()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.half](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.half)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">inlined_graph()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.inlined_graph](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.inlined_graph)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">ipu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.ipu](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.ipu)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.load_state_dict](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.load_state_dict)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.modules](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.modules)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">named_buffers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.named_buffers](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_buffers)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">named_children()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.named_children](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_children)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">named_modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.named_modules](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_modules)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">named_parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.named_parameters](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_parameters)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">parameters()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.parameters](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.parameters)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">register_backward_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_backward_hook](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_backward_hook)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">register_buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_buffer](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_buffer)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">register_forward_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_forward_hook](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_forward_hook)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">register_forward_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_forward_pre_hook](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_forward_pre_hook)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">register_full_backward_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_full_backward_hook](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_full_backward_hook)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">register_full_backward_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_full_backward_pre_hook](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_full_backward_pre_hook)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_load_state_dict_pre_hook)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_load_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_load_state_dict_post_hook)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">register_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_module](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_module)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">register_parameter()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_parameter](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_parameter)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_state_dict_pre_hook](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_state_dict_pre_hook)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.register_state_dict_post_hook](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_state_dict_post_hook)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">requires_grad_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.requires_grad_](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.requires_grad_)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">save()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.save](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.save)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">set_extra_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.set_extra_state](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.set_extra_state)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_submodule()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.set_submodule](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.set_submodule)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">share_memory()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.share_memory](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.share_memory)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.state_dict](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.state_dict)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">to()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.to](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.to)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">to_empty()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.to_empty](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.to_empty)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">train()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.train](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.train)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.type](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.type)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp16，fp32，uint8，int8，int16，int32，int64，bool

</div>

> <font size="3">xpu()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.xpu](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.xpu)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptModule.zero_grad](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.zero_grad)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

</div>

### <code><i>class</i></code> torch.jit.ScriptFunction

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptFunction](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptFunction.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

> <font size="3">get_debug_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptFunction.get_debug_state](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction.get_debug_state)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">save()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptFunction.save](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction.save)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">save_to_buffer()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ScriptFunction.save_to_buffer](https://pytorch.org/docs/2.7/generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction.save_to_buffer)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### torch.jit.freeze

<div style="margin-left: 2em">

**原生文档**：[torch.jit.freeze](https://pytorch.org/docs/2.7/generated/torch.jit.freeze.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.jit.optimize_for_inference

<div style="margin-left: 2em">

**原生文档**：[torch.jit.optimize_for_inference](https://pytorch.org/docs/2.7/generated/torch.jit.optimize_for_inference.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.jit.enable_onednn_fusion

<div style="margin-left: 2em">

**原生文档**：[torch.jit.enable_onednn_fusion](https://pytorch.org/docs/2.7/generated/torch.jit.enable_onednn_fusion.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.jit.onednn_fusion_enabled

<div style="margin-left: 2em">

**原生文档**：[torch.jit.onednn_fusion_enabled](https://pytorch.org/docs/2.7/generated/torch.jit.onednn_fusion_enabled.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.jit.set_fusion_strategy

<div style="margin-left: 2em">

**原生文档**：[torch.jit.set_fusion_strategy](https://pytorch.org/docs/2.7/generated/torch.jit.set_fusion_strategy.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.jit.strict_fusion

<div style="margin-left: 2em">

**原生文档**：[torch.jit.strict_fusion](https://pytorch.org/docs/2.7/generated/torch.jit.strict_fusion.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.jit.save

<div style="margin-left: 2em">

**原生文档**：[torch.jit.save](https://pytorch.org/docs/2.7/generated/torch.jit.save.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.jit.load

<div style="margin-left: 2em">

**原生文档**：[torch.jit.load](https://pytorch.org/docs/2.7/generated/torch.jit.load.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：支持

</div>

### torch.jit.ignore

<div style="margin-left: 2em">

**原生文档**：[torch.jit.ignore](https://pytorch.org/docs/2.7/generated/torch.jit.ignore.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.jit.unused

<div style="margin-left: 2em">

**原生文档**：[torch.jit.unused](https://pytorch.org/docs/2.7/generated/torch.jit.unused.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.jit.isinstance

<div style="margin-left: 2em">

**原生文档**：[torch.jit.isinstance](https://pytorch.org/docs/2.7/generated/torch.jit.isinstance.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.jit.Attribute

<div style="margin-left: 2em">

**原生文档**：[torch.jit.Attribute](https://pytorch.org/docs/2.7/generated/torch.jit.Attribute.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

> <font size="3">count()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.Attribute.count](https://pytorch.org/docs/2.7/generated/torch.jit.Attribute.html#torch.jit.Attribute.count)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">index()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.Attribute.index](https://pytorch.org/docs/2.7/generated/torch.jit.Attribute.html#torch.jit.Attribute.index)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

> <font size="3">type()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.Attribute.type](https://pytorch.org/docs/2.7/generated/torch.jit.Attribute.html#torch.jit.Attribute.type)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">value()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.jit.Attribute.value](https://pytorch.org/docs/2.7/generated/torch.jit.Attribute.html#torch.jit.Attribute.value)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### torch.jit.annotate

<div style="margin-left: 2em">

**原生文档**：[torch.jit.annotate](https://pytorch.org/docs/2.7/generated/torch.jit.annotate.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**：`input`仅支持fp32

</div>

## Appendix

### torch.jit.export

<div style="margin-left: 2em">

**原生文档**：[torch.jit.export](https://pytorch.org/docs/2.7/jit.html#torch.jit.export)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>
