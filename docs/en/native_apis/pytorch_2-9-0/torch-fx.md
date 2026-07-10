# torch.fx

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:38:49.811Z pushedAt=2026-07-09T08:44:08.367Z -->

> [!NOTE]
> If the "Supported" column shows "Yes" and "Restrictions and Notes" shows "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|torch.fx.symbolic_trace|Yes|Supports fp32|
|torch.fx.wrap|Yes|Supports fp32|
|torch.fx.GraphModule|Yes|Supports fp32|
|torch.fx.GraphModule.\_\_init_\_|Yes|Supports fp32|
|torch.fx.GraphModule.add_submodule|Yes|Supports fp32|
|torch.fx.GraphModule.code|No|-|
|torch.fx.GraphModule.delete_all_unused_submodules|Yes|Supports fp32|
|torch.fx.GraphModule.delete_submodule|Yes|Supports fp32|
|torch.fx.GraphModule.graph|No|-|
|torch.fx.GraphModule.print_readable|Yes|-|
|torch.fx.GraphModule.recompile|Yes|-|
|torch.fx.GraphModule.to_folder|Yes|Supports fp32|
|torch.fx.Graph|Yes|-|
|torch.fx.Graph.\_\_init_\_|Yes|-|
|torch.fx.Graph.call_function|Yes|-|
|torch.fx.Graph.call_method|Yes|-|
|torch.fx.Graph.call_module|Yes|-|
|torch.fx.Graph.create_node|Yes|-|
|torch.fx.Graph.eliminate_dead_code|Yes|-|
|torch.fx.Graph.erase_node|Yes|-|
|torch.fx.Graph.get_attr|Yes|-|
|torch.fx.Graph.graph_copy|Yes|-|
|torch.fx.Graph.inserting_after|Yes|-|
|torch.fx.Graph.inserting_before|Yes|-|
|torch.fx.Graph.lint|Yes|Supports fp32|
|torch.fx.Graph.node_copy|Yes|Supports fp32|
|torch.fx.Graph.nodes|No|-|
|torch.fx.Graph.on_generate_code|Yes|Supports fp32|
|torch.fx.Graph.output|Yes|-|
|torch.fx.Graph.placeholder|Yes|-|
|torch.fx.Graph.print_tabular|Yes|Supports fp32|
|torch.fx.Graph.process_inputs|Yes|-|
|torch.fx.Graph.process_outputs|Yes|-|
|torch.fx.Graph.python_code|Yes|Supports fp32|
|torch.fx.Graph.set_codegen|Yes|-|
|torch.fx.Node|Yes|Supports fp32|
|torch.fx.Node.all_input_nodes|No|-|
|torch.fx.Node.append|Yes|Supports fp32|
|torch.fx.Node.args|No|-|
|torch.fx.Node.format_node|Yes|Supports fp32|
|torch.fx.Node.is_impure|Yes|Supports fp32|
|torch.fx.Node.kwargs|No|-|
|torch.fx.Node.next|No|-|
|torch.fx.Node.normalized_arguments|Yes|-|
|torch.fx.Node.prepend|Yes|-|
|torch.fx.Node.prev|No|-|
|torch.fx.Node.replace_all_uses_with|Yes|-|
|torch.fx.Node.replace_input_with|Yes|Supports fp32|
|torch.fx.Node.stack_trace|No|-|
|torch.fx.Node.update_arg|Yes|Supports fp32|
|torch.fx.Node.update_kwarg|Yes|Supports fp32|
|torch.fx.Tracer|Yes|-|
|torch.fx.Tracer.call_module|Yes|Supports fp32|
|torch.fx.Tracer.create_arg|Yes|-|
|torch.fx.Tracer.create_args_for_root|Yes|Supports fp32|
|torch.fx.Tracer.create_node|Yes|-|
|torch.fx.Tracer.create_proxy|Yes|Supports fp32|
|torch.fx.Tracer.getattr|Yes|-|
|torch.fx.Tracer.is_leaf_module|Yes|-|
|torch.fx.Tracer.iter|Yes|Supports fp32|
|torch.fx.Tracer.keys|Yes|Supports fp32|
|torch.fx.Tracer.path_of_module|Yes|-|
|torch.fx.Tracer.proxy|Yes|Supports fp32|
|torch.fx.Tracer.to_bool|Yes|Supports fp32|
|torch.fx.Tracer.trace|Yes|-|
|torch.fx.Proxy|Yes|-|
|torch.fx.Interpreter|Yes|-|
|torch.fx.Interpreter.call_function|Yes|Supports fp32|
|torch.fx.Interpreter.call_method|Yes|Supports fp32|
|torch.fx.Interpreter.call_module|Yes|Supports fp32|
|torch.fx.Interpreter.fetch_args_kwargs_from_env|Yes|Supports fp32|
|torch.fx.Interpreter.fetch_attr|Yes|Supports fp32|
|torch.fx.Interpreter.get_attr|Yes|Supports fp32|
|torch.fx.Interpreter.map_nodes_to_values|Yes|Supports fp32|
|torch.fx.Interpreter.output|Yes|Supports fp32|
|torch.fx.Interpreter.placeholder|Yes|Supports fp32|
|torch.fx.Interpreter.run|Yes|-|
|torch.fx.Interpreter.run_node|Yes|Supports fp32|
|torch.fx.Transformer|Yes|-|
|torch.fx.Transformer.call_function|Yes|Supports fp32|
|torch.fx.Transformer.call_module|Yes|Supports fp32|
|torch.fx.Transformer.get_attr|Yes|Supports fp32|
|torch.fx.Transformer.placeholder|Yes|Supports fp32|
|torch.fx.Transformer.transform|Yes|-|
|torch.fx.replace_pattern|Yes|-|
