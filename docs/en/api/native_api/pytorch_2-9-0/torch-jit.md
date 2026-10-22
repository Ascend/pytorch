# torch.jit

> [!NOTE]
>
> - TorchScript is deprecated; prefer `torch.export` or `torch.compile`. This page retains the support information inherited from the previous version for reference by existing code.
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.9/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.9/jit.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [Creating TorchScript Code](#creating-torchscript-code)

</div>

<div style="display:none;">

## &#8203;torch.jit

</div>

## Creating TorchScript Code

### torch.jit.script

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.script](https://pytorch.org/docs/2.9/generated/torch.jit.script.html)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id3 -->

</div>

### torch.jit.trace

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.trace](https://pytorch.org/docs/2.9/generated/torch.jit.trace.html)

**Support Status**:

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id6 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.jit.script_if_tracing

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.script_if_tracing](https://pytorch.org/docs/2.9/generated/torch.jit.script_if_tracing.html)

**Support Status**:

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id9 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.jit.trace_module

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.trace_module](https://pytorch.org/docs/2.9/generated/torch.jit.trace_module.html)

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id12 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### torch.jit.fork

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.fork](https://pytorch.org/docs/2.9/generated/torch.jit.fork.html)

**Support Status**:

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id15 -->

</div>

### torch.jit.wait

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.wait](https://pytorch.org/docs/2.9/generated/torch.jit.wait.html)

**Support Status**:

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id18 -->

</div>

### <code><i>class</i></code> torch.jit.ScriptModule

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html)

**Support Status**:

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id21 -->

**Constraints & Notes**: `input` only supports fp32

> <font size="3">add_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.add_module](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.add_module)

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id24 -->

</div>

> <font size="3">apply()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.apply](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.apply)

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id27 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">bfloat16()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.bfloat16](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.bfloat16)

**Support Status**:

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id30 -->

</div>

> <font size="3">buffers()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.buffers](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.buffers)

**Support Status**:

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id33 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">children()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.children](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.children)

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id36 -->

</div>

> <font size="3">code()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.code](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.code)

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id39 -->

</div>

> <font size="3">code_with_constants()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.code_with_constants](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.code_with_constants)

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id42 -->

</div>

> <font size="3">compile()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.compile](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.compile)

**Support Status**:

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id45 -->

</div>

> <font size="3">cpu()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.cpu](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.cpu)

**Support Status**:

<!-- npu="910b" id46 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id48 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">cuda()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.cuda](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.cuda)

**Support Status**:

<!-- npu="910b" id49 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id51 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">double()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.double](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.double)

**Support Status**:

<!-- npu="910b" id52 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id54 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">eval()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.eval](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.eval)

**Support Status**:

<!-- npu="910b" id55 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id57 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">extra_repr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.extra_repr](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.extra_repr)

**Support Status**:

<!-- npu="910b" id58 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id60 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">float()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.float](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.float)

**Support Status**:

<!-- npu="910b" id61 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id63 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">get_buffer()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.get_buffer](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_buffer)

**Support Status**:

<!-- npu="910b" id64 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id66 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">get_extra_state()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.get_extra_state](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_extra_state)

**Support Status**:

<!-- npu="910b" id67 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id69 -->

</div>

> <font size="3">get_parameter()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.get_parameter](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_parameter)

**Support Status**:

<!-- npu="910b" id70 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id72 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">get_submodule()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.get_submodule](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.get_submodule)

**Support Status**:

<!-- npu="910b" id73 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id75 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">graph()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.graph](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.graph)

**Support Status**:

<!-- npu="910b" id76 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id78 -->

</div>

> <font size="3">half()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.half](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.half)

**Support Status**:

<!-- npu="910b" id79 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id81 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">inlined_graph()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.inlined_graph](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.inlined_graph)

**Support Status**:

<!-- npu="910b" id82 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id84 -->

</div>

> <font size="3">ipu()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.ipu](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.ipu)

**Support Status**:

<!-- npu="910b" id85 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id87 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">load_state_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.load_state_dict](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.load_state_dict)

**Support Status**:

<!-- npu="910b" id88 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id90 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">modules()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.modules](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.modules)

**Support Status**:

<!-- npu="910b" id91 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id93 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">named_buffers()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.named_buffers](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_buffers)

**Support Status**:

<!-- npu="910b" id94 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id96 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">named_children()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.named_children](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_children)

**Support Status**:

<!-- npu="910b" id97 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id99 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">named_modules()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.named_modules](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_modules)

**Support Status**:

<!-- npu="910b" id100 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id102 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">named_parameters()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.named_parameters](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.named_parameters)

**Support Status**:

<!-- npu="910b" id103 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id105 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">parameters()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.parameters](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.parameters)

**Support Status**:

<!-- npu="910b" id106 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id108 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">register_backward_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.register_backward_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_backward_hook)

**Support Status**:

<!-- npu="910b" id109 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id111 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">register_buffer()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.register_buffer](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_buffer)

**Support Status**:

<!-- npu="910b" id112 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id114 -->

</div>

> <font size="3">register_forward_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.register_forward_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_forward_hook)

**Support Status**:

<!-- npu="910b" id115 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id117 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">register_forward_pre_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.register_forward_pre_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_forward_pre_hook)

**Support Status**:

<!-- npu="910b" id118 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id120 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">register_full_backward_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.register_full_backward_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_full_backward_hook)

**Support Status**:

<!-- npu="910b" id121 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id123 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">register_full_backward_pre_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.register_full_backward_pre_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_full_backward_pre_hook)

**Support Status**:

<!-- npu="910b" id124 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id126 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">register_load_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.register_load_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_load_state_dict_pre_hook)

**Support Status**:

<!-- npu="910b" id127 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id129 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">register_load_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.register_load_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_load_state_dict_post_hook)

**Support Status**:

<!-- npu="910b" id130 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id132 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">register_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.register_module](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_module)

**Support Status**:

<!-- npu="910b" id133 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id135 -->

</div>

> <font size="3">register_parameter()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.register_parameter](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_parameter)

**Support Status**:

<!-- npu="910b" id136 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id138 -->

</div>

> <font size="3">register_state_dict_pre_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.register_state_dict_pre_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_state_dict_pre_hook)

**Support Status**:

<!-- npu="910b" id139 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id141 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">register_state_dict_post_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.register_state_dict_post_hook](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.register_state_dict_post_hook)

**Support Status**:

<!-- npu="910b" id142 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id143 -->
<!-- npu="950" id144 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id144 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">requires_grad_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.requires_grad_](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.requires_grad_)

**Support Status**:

<!-- npu="910b" id145 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id147 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">save()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.save](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.save)

**Support Status**:

<!-- npu="910b" id148 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id150 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">set_extra_state()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.set_extra_state](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.set_extra_state)

**Support Status**:

<!-- npu="910b" id151 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id153 -->

</div>

> <font size="3">set_submodule()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.set_submodule](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.set_submodule)

**Support Status**:

<!-- npu="910b" id154 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id156 -->

</div>

> <font size="3">share_memory()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.share_memory](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.share_memory)

**Support Status**:

<!-- npu="910b" id157 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id159 -->

</div>

> <font size="3">state_dict()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.state_dict](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.state_dict)

**Version Notes**: Passing `destination`, `prefix`, and `keep_vars` as positional arguments is deprecated; use keyword arguments.

**Support Status**:

<!-- npu="910b" id160 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id162 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">to()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.to](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.to)

**Support Status**:

<!-- npu="910b" id163 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id165 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">to_empty()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.to_empty](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.to_empty)

**Support Status**:

<!-- npu="910b" id166 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id168 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">train()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.train](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.train)

**Support Status**:

<!-- npu="910b" id169 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id171 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">type()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.type](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.type)

**Support Status**:

<!-- npu="910b" id172 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id174 -->

**Constraints & Notes**: `input` only supports fp16, fp32, uint8, int8, int16, int32, int64, bool

</div>

> <font size="3">xpu()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.xpu](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.xpu)

**Support Status**:

<!-- npu="910b" id175 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id177 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">zero_grad()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptModule.zero_grad](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule.zero_grad)

**Support Status**:

<!-- npu="910b" id178 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id180 -->

**Constraints & Notes**: `input` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.jit.ScriptFunction

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptFunction](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptFunction.html)

**Support Status**:

<!-- npu="910b" id181 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id183 -->

**Constraints & Notes**: `input` only supports fp32

> <font size="3">get_debug_state()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptFunction.get_debug_state](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction.get_debug_state)

**Support Status**:

<!-- npu="910b" id184 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id186 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">save()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptFunction.save](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction.save)

**Support Status**:

<!-- npu="910b" id187 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id189 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">save_to_buffer()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ScriptFunction.save_to_buffer](https://pytorch.org/docs/2.9/generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction.save_to_buffer)

**Support Status**:

<!-- npu="910b" id190 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id192 -->

</div>

</div>

### torch.jit.freeze

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.freeze](https://pytorch.org/docs/2.9/generated/torch.jit.freeze.html)

**Support Status**:

<!-- npu="910b" id193 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id195 -->

</div>

### torch.jit.optimize_for_inference

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.optimize_for_inference](https://pytorch.org/docs/2.9/generated/torch.jit.optimize_for_inference.html)

**Support Status**:

<!-- npu="910b" id196 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id197 -->
<!-- npu="950" id198 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id198 -->

</div>

### torch.jit.enable_onednn_fusion

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.enable_onednn_fusion](https://pytorch.org/docs/2.9/generated/torch.jit.enable_onednn_fusion.html)

**Support Status**:

<!-- npu="910b" id199 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id200 -->
<!-- npu="950" id201 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id201 -->

</div>

### torch.jit.onednn_fusion_enabled

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.onednn_fusion_enabled](https://pytorch.org/docs/2.9/generated/torch.jit.onednn_fusion_enabled.html)

**Support Status**:

<!-- npu="910b" id202 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id203 -->
<!-- npu="950" id204 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id204 -->

</div>

### torch.jit.set_fusion_strategy

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.set_fusion_strategy](https://pytorch.org/docs/2.9/generated/torch.jit.set_fusion_strategy.html)

**Support Status**:

<!-- npu="910b" id205 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id206 -->
<!-- npu="950" id207 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id207 -->

</div>

### torch.jit.strict_fusion

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.strict_fusion](https://pytorch.org/docs/2.9/generated/torch.jit.strict_fusion.html)

**Support Status**:

<!-- npu="910b" id208 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id209 -->
<!-- npu="950" id210 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id210 -->

</div>

### torch.jit.save

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.save](https://pytorch.org/docs/2.9/generated/torch.jit.save.html)

**Support Status**:

<!-- npu="910b" id211 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id212 -->
<!-- npu="950" id213 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id213 -->

</div>

### torch.jit.load

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.load](https://pytorch.org/docs/2.9/generated/torch.jit.load.html)

**Support Status**:

<!-- npu="910b" id214 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id215 -->
<!-- npu="950" id216 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id216 -->

</div>

### torch.jit.ignore

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.ignore](https://pytorch.org/docs/2.9/generated/torch.jit.ignore.html)

**Support Status**:

<!-- npu="910b" id217 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id218 -->
<!-- npu="950" id219 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id219 -->

</div>

### torch.jit.unused

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.unused](https://pytorch.org/docs/2.9/generated/torch.jit.unused.html)

**Support Status**:

<!-- npu="910b" id220 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id220 -->
<!-- npu="A3" id221 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id221 -->
<!-- npu="950" id222 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id222 -->

</div>

### torch.jit.isinstance

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.isinstance](https://pytorch.org/docs/2.9/generated/torch.jit.isinstance.html)

**Support Status**:

<!-- npu="910b" id223 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id223 -->
<!-- npu="A3" id224 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id224 -->
<!-- npu="950" id225 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id225 -->

</div>

### <code><i>class</i></code> torch.jit.Attribute

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.Attribute](https://pytorch.org/docs/2.9/generated/torch.jit.Attribute.html)

**Support Status**:

<!-- npu="910b" id226 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id226 -->
<!-- npu="A3" id227 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id227 -->
<!-- npu="950" id228 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id228 -->

**Constraints & Notes**: `input` only supports fp32

> <font size="3">count()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.Attribute.count](https://pytorch.org/docs/2.9/generated/torch.jit.Attribute.html#torch.jit.Attribute.count)

**Support Status**:

<!-- npu="910b" id229 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id229 -->
<!-- npu="A3" id230 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id230 -->
<!-- npu="950" id231 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id231 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">index()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.Attribute.index](https://pytorch.org/docs/2.9/generated/torch.jit.Attribute.html#torch.jit.Attribute.index)

**Support Status**:

<!-- npu="910b" id232 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id232 -->
<!-- npu="A3" id233 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id233 -->
<!-- npu="950" id234 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id234 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">type()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.Attribute.type](https://pytorch.org/docs/2.9/generated/torch.jit.Attribute.html#torch.jit.Attribute.type)

**Support Status**:

<!-- npu="910b" id235 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id235 -->
<!-- npu="A3" id236 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id236 -->
<!-- npu="950" id237 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id237 -->

</div>

> <font size="3">value()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.Attribute.value](https://pytorch.org/docs/2.9/generated/torch.jit.Attribute.html#torch.jit.Attribute.value)

**Support Status**:

<!-- npu="910b" id238 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id238 -->
<!-- npu="A3" id239 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id239 -->
<!-- npu="950" id240 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id240 -->

</div>

</div>

### torch.jit.annotate

<div style="margin-left: 2em">

**Native Documentation**: [torch.jit.annotate](https://pytorch.org/docs/2.9/generated/torch.jit.annotate.html)

**Support Status**:

<!-- npu="910b" id241 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id241 -->
<!-- npu="A3" id242 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id242 -->
<!-- npu="950" id243 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id243 -->

**Constraints & Notes**: `input` only supports fp32

</div>
