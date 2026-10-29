# torch.fx

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.11/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.11/fx.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [API Reference](#api-reference)

</div>

<div style="display:none;">

## &#8203;torch.fx

</div>

## API Reference

### torch.fx.symbolic_trace

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.symbolic_trace](https://pytorch.org/docs/2.11/fx.html#torch.fx.symbolic_trace)

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

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.fx.GraphModule

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.GraphModule](https://pytorch.org/docs/2.11/fx.html#torch.fx.GraphModule)

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

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.GraphModule.\_\_init\_\_](https://pytorch.org/docs/2.11/fx.html#torch.fx.GraphModule.__init__)

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

> <font size="3">add_submodule()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.GraphModule.add_submodule](https://pytorch.org/docs/2.11/fx.html#torch.fx.GraphModule.add_submodule)

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

> <font size="3">code()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.GraphModule.code](https://pytorch.org/docs/2.11/fx.html#torch.fx.GraphModule.code)

**Support Status**:

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id15 -->

</div>

> <font size="3">delete_all_unused_submodules()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.GraphModule.delete_all_unused_submodules](https://pytorch.org/docs/2.11/fx.html#torch.fx.GraphModule.delete_all_unused_submodules)

**Support Status**:

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id18 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">delete_submodule()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.GraphModule.delete_submodule](https://pytorch.org/docs/2.11/fx.html#torch.fx.GraphModule.delete_submodule)

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

</div>

> <font size="3">graph()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.GraphModule.graph](https://pytorch.org/docs/2.11/fx.html#torch.fx.GraphModule.graph)

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id24 -->

</div>

> <font size="3">print_readable()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.GraphModule.print_readable](https://pytorch.org/docs/2.11/fx.html#torch.fx.GraphModule.print_readable)

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

</div>

> <font size="3">recompile()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.GraphModule.recompile](https://pytorch.org/docs/2.11/fx.html#torch.fx.GraphModule.recompile)

**Support Status**:

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id30 -->

</div>

> <font size="3">to_folder()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.GraphModule.to_folder](https://pytorch.org/docs/2.11/fx.html#torch.fx.GraphModule.to_folder)

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

</div>

### <code><i>class</i></code> torch.fx.Graph

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph)

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

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.\_\_init\_\_](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.__init__)

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

> <font size="3">call_function()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.call_function](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.call_function)

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

> <font size="3">call_method()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.call_method](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.call_method)

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

> <font size="3">call_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.call_module](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.call_module)

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

</div>

> <font size="3">create_node()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.create_node](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.create_node)

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

</div>

> <font size="3">eliminate_dead_code()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.eliminate_dead_code](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.eliminate_dead_code)

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

</div>

> <font size="3">erase_node()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.erase_node](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.erase_node)

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

</div>

> <font size="3">get_attr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.get_attr](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.get_attr)

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

</div>

> <font size="3">graph_copy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.graph_copy](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.graph_copy)

**Support Status**:

<!-- npu="910b" id61 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id63 -->

</div>

> <font size="3">find_nodes()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.find_nodes](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.find_nodes)

**Support Status**:

<!-- npu="910b" id64 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id66 -->

</div>

> <font size="3">inserting_after()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.inserting_after](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.inserting_after)

**Support Status**:

<!-- npu="910b" id67 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id69 -->

</div>

> <font size="3">inserting_before()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.inserting_before](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.inserting_before)

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

</div>

> <font size="3">lint()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.lint](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.lint)

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

> <font size="3">node_copy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.node_copy](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.node_copy)

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

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">nodes()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.nodes](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.nodes)

**Support Status**:

<!-- npu="910b" id79 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id81 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">on_generate_code()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.on_generate_code](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.on_generate_code)

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

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">output()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.output](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.output)

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

</div>

> <font size="3">output_node()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.output_node](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.output_node)

**Support Status**:

<!-- npu="910b" id88 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id90 -->

</div>

> <font size="3">placeholder()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.placeholder](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.placeholder)

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

</div>

> <font size="3">print_tabular()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.print_tabular](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.print_tabular)

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

> <font size="3">process_inputs()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.process_inputs](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.process_inputs)

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

</div>

> <font size="3">process_outputs()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.process_outputs](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.process_outputs)

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

</div>

> <font size="3">python_code()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.python_code](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.python_code)

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

> <font size="3">set_codegen()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Graph.set_codegen](https://pytorch.org/docs/2.11/fx.html#torch.fx.Graph.set_codegen)

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

</div>

</div>

### <code><i>class</i></code> torch.fx.Tracer

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Tracer](https://pytorch.org/docs/2.11/fx.html#torch.fx.Tracer)

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

> <font size="3">call_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Tracer.call_module](https://pytorch.org/docs/2.11/fx.html#torch.fx.Tracer.call_module)

**Support Status**:

<!-- npu="910b" id112 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id114 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">create_arg()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Tracer.create_arg](https://pytorch.org/docs/2.11/fx.html#torch.fx.Tracer.create_arg)

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

</div>

> <font size="3">create_args_for_root()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Tracer.create_args_for_root](https://pytorch.org/docs/2.11/fx.html#torch.fx.Tracer.create_args_for_root)

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

> <font size="3">create_node()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Tracer.create_node](https://pytorch.org/docs/2.11/fx.html#torch.fx.Tracer.create_node)

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

</div>

> <font size="3">create_proxy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Tracer.create_proxy](https://pytorch.org/docs/2.11/fx.html#torch.fx.Tracer.create_proxy)

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

> <font size="3">getattr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Tracer.getattr](https://pytorch.org/docs/2.11/fx.html#torch.fx.Tracer.getattr)

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

</div>

> <font size="3">is_leaf_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Tracer.is_leaf_module](https://pytorch.org/docs/2.11/fx.html#torch.fx.Tracer.is_leaf_module)

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

</div>

> <font size="3">iter()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Tracer.iter](https://pytorch.org/docs/2.11/fx.html#torch.fx.Tracer.iter)

**Support Status**:

<!-- npu="910b" id133 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id135 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">keys()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Tracer.keys](https://pytorch.org/docs/2.11/fx.html#torch.fx.Tracer.keys)

**Support Status**:

<!-- npu="910b" id136 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id138 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">path_of_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Tracer.path_of_module](https://pytorch.org/docs/2.11/fx.html#torch.fx.Tracer.path_of_module)

**Support Status**:

<!-- npu="910b" id139 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id141 -->

</div>

> <font size="3">proxy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Tracer.proxy](https://pytorch.org/docs/2.11/fx.html#torch.fx.Tracer.proxy)

**Support Status**:

<!-- npu="910b" id142 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id143 -->
<!-- npu="950" id144 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id144 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">to_bool()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Tracer.to_bool](https://pytorch.org/docs/2.11/fx.html#torch.fx.Tracer.to_bool)

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

> <font size="3">trace()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Tracer.trace](https://pytorch.org/docs/2.11/fx.html#torch.fx.Tracer.trace)

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

</div>

</div>

### torch.fx.wrap

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.wrap](https://pytorch.org/docs/2.11/fx.html#torch.fx.wrap)

**Support Status**:

<!-- npu="910b" id151 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id153 -->

**Constraints & Notes**: `input` only supports fp32

</div>

### <code><i>class</i></code> torch.fx.Node

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Node](https://pytorch.org/docs/2.11/fx.html#torch.fx.Node)

**Support Status**:

<!-- npu="910b" id154 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id156 -->

**Constraints & Notes**: `input` only supports fp32

> <font size="3">all_input_nodes()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Node.all_input_nodes](https://pytorch.org/docs/2.11/fx.html#torch.fx.Node.all_input_nodes)

**Support Status**:

<!-- npu="910b" id157 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id159 -->

</div>

> <font size="3">append()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Node.append](https://pytorch.org/docs/2.11/fx.html#torch.fx.Node.append)

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

> <font size="3">args()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Node.args](https://pytorch.org/docs/2.11/fx.html#torch.fx.Node.args)

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

</div>

> <font size="3">format_node()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Node.format_node](https://pytorch.org/docs/2.11/fx.html#torch.fx.Node.format_node)

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

> <font size="3">is_impure()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Node.is_impure](https://pytorch.org/docs/2.11/fx.html#torch.fx.Node.is_impure)

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

> <font size="3">kwargs()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Node.kwargs](https://pytorch.org/docs/2.11/fx.html#torch.fx.Node.kwargs)

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

</div>

> <font size="3">next()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Node.next](https://pytorch.org/docs/2.11/fx.html#torch.fx.Node.next)

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

</div>

> <font size="3">normalized_arguments()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Node.normalized_arguments](https://pytorch.org/docs/2.11/fx.html#torch.fx.Node.normalized_arguments)

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

</div>

> <font size="3">prepend()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Node.prepend](https://pytorch.org/docs/2.11/fx.html#torch.fx.Node.prepend)

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

</div>

> <font size="3">prev()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Node.prev](https://pytorch.org/docs/2.11/fx.html#torch.fx.Node.prev)

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

</div>

> <font size="3">replace_all_uses_with()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Node.replace_all_uses_with](https://pytorch.org/docs/2.11/fx.html#torch.fx.Node.replace_all_uses_with)

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

</div>

> <font size="3">replace_input_with()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Node.replace_input_with](https://pytorch.org/docs/2.11/fx.html#torch.fx.Node.replace_input_with)

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

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">stack_trace()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Node.stack_trace](https://pytorch.org/docs/2.11/fx.html#torch.fx.Node.stack_trace)

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

> <font size="3">update_arg()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Node.update_arg](https://pytorch.org/docs/2.11/fx.html#torch.fx.Node.update_arg)

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

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">update_kwarg()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Node.update_kwarg](https://pytorch.org/docs/2.11/fx.html#torch.fx.Node.update_kwarg)

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

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">insert_arg()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Node.insert_arg](https://pytorch.org/docs/2.11/fx.html#torch.fx.Node.insert_arg)

**Support Status**:

<!-- npu="910b" id202 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id203 -->
<!-- npu="950" id204 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id204 -->

</div>

</div>

### <code><i>class</i></code> torch.fx.Proxy

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Proxy](https://pytorch.org/docs/2.11/fx.html#torch.fx.Proxy)

**Support Status**:

<!-- npu="910b" id205 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id206 -->
<!-- npu="950" id207 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id207 -->

</div>

### <code><i>class</i></code> torch.fx.Interpreter

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Interpreter](https://pytorch.org/docs/2.11/fx.html#torch.fx.Interpreter)

**Support Status**:

<!-- npu="910b" id208 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id209 -->
<!-- npu="950" id210 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id210 -->

> <font size="3">call_function()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Interpreter.call_function](https://pytorch.org/docs/2.11/fx.html#torch.fx.Interpreter.call_function)

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

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">call_method()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Interpreter.call_method](https://pytorch.org/docs/2.11/fx.html#torch.fx.Interpreter.call_method)

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

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">call_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Interpreter.call_module](https://pytorch.org/docs/2.11/fx.html#torch.fx.Interpreter.call_module)

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

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">fetch_args_kwargs_from_env()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Interpreter.fetch_args_kwargs_from_env](https://pytorch.org/docs/2.11/fx.html#torch.fx.Interpreter.fetch_args_kwargs_from_env)

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

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">fetch_attr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Interpreter.fetch_attr](https://pytorch.org/docs/2.11/fx.html#torch.fx.Interpreter.fetch_attr)

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

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">get_attr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Interpreter.get_attr](https://pytorch.org/docs/2.11/fx.html#torch.fx.Interpreter.get_attr)

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

</div>

> <font size="3">map_nodes_to_values()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Interpreter.map_nodes_to_values](https://pytorch.org/docs/2.11/fx.html#torch.fx.Interpreter.map_nodes_to_values)

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

> <font size="3">output()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Interpreter.output](https://pytorch.org/docs/2.11/fx.html#torch.fx.Interpreter.output)

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

> <font size="3">placeholder()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Interpreter.placeholder](https://pytorch.org/docs/2.11/fx.html#torch.fx.Interpreter.placeholder)

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

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">run()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Interpreter.run](https://pytorch.org/docs/2.11/fx.html#torch.fx.Interpreter.run)

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

> <font size="3">boxed_run()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Interpreter.boxed_run](https://pytorch.org/docs/2.11/fx.html#torch.fx.Interpreter.boxed_run)

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

</div>

> <font size="3">run_node()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Interpreter.run_node](https://pytorch.org/docs/2.11/fx.html#torch.fx.Interpreter.run_node)

**Support Status**:

<!-- npu="910b" id244 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id244 -->
<!-- npu="A3" id245 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id245 -->
<!-- npu="950" id246 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id246 -->

**Constraints & Notes**: `input` only supports fp32

</div>

</div>

### <code><i>class</i></code> torch.fx.Transformer

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Transformer](https://pytorch.org/docs/2.11/fx.html#torch.fx.Transformer)

**Support Status**:

<!-- npu="910b" id247 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id247 -->
<!-- npu="A3" id248 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id248 -->
<!-- npu="950" id249 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id249 -->

> <font size="3">call_function()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Transformer.call_function](https://pytorch.org/docs/2.11/fx.html#torch.fx.Transformer.call_function)

**Support Status**:

<!-- npu="910b" id250 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id250 -->
<!-- npu="A3" id251 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id251 -->
<!-- npu="950" id252 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id252 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">call_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Transformer.call_module](https://pytorch.org/docs/2.11/fx.html#torch.fx.Transformer.call_module)

**Support Status**:

<!-- npu="910b" id253 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id253 -->
<!-- npu="A3" id254 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id254 -->
<!-- npu="950" id255 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id255 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">get_attr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Transformer.get_attr](https://pytorch.org/docs/2.11/fx.html#torch.fx.Transformer.get_attr)

**Support Status**:

<!-- npu="910b" id256 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id256 -->
<!-- npu="A3" id257 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id257 -->
<!-- npu="950" id258 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id258 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">placeholder()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Transformer.placeholder](https://pytorch.org/docs/2.11/fx.html#torch.fx.Transformer.placeholder)

**Support Status**:

<!-- npu="910b" id259 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id259 -->
<!-- npu="A3" id260 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id260 -->
<!-- npu="950" id261 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id261 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">transform()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.Transformer.transform](https://pytorch.org/docs/2.11/fx.html#torch.fx.Transformer.transform)

**Support Status**:

<!-- npu="910b" id262 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id262 -->
<!-- npu="A3" id263 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id263 -->
<!-- npu="950" id264 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id264 -->

</div>

</div>

### torch.fx.replace_pattern

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.replace_pattern](https://pytorch.org/docs/2.11/fx.html#torch.fx.replace_pattern)

**Support Status**:

<!-- npu="910b" id265 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id265 -->
<!-- npu="A3" id266 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id266 -->
<!-- npu="950" id267 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id267 -->

</div>
