# torch.fx

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-14T07:57:12.035Z pushedAt=2026-06-14T09:16:34.752Z -->

> [!NOTE]
> If the "Supported" column for an API is "Yes" and the "Restrictions and Notes" column is "-", it means the API support is consistent with the native API.

|API Name|Supported|Restrictions and Notes|
|--|--|--|
|[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbolic_sizes_strides_storage_offset](https://pytorch.org/docs/2.10/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbolic_sizes_strides_storage_offset.html)|Yes|-|
|[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symboolnode](https://pytorch.org/docs/2.10/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symboolnode.html)|Yes|-|
|[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symfloatnode](https://pytorch.org/docs/2.10/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symfloatnode.html)|Yes|-|
|[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symintnode](https://pytorch.org/docs/2.10/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symintnode.html)|Yes|-|
|[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unbacked_symbool](https://pytorch.org/docs/2.10/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unbacked_symbool.html)|Yes|-|
|[torch.fx.symbolic_trace](https://pytorch.org/docs/2.10/fx.html#torch.fx.symbolic_trace)|Yes|fp32 supported|
|[torch.fx.wrap](https://pytorch.org/docs/2.10/fx.html#torch.fx.wrap)|Yes|fp32 supported|
|[torch.fx.GraphModule](https://pytorch.org/docs/2.10/fx.html#torch.fx.GraphModule)|Yes|fp32 supported|
|[torch.fx.GraphModule.\_\_init_\_](https://pytorch.org/docs/2.10/fx.html#torch.fx.GraphModule.__init__)|Yes|fp32 supported|
|[torch.fx.GraphModule.add_submodule](https://pytorch.org/docs/2.10/fx.html#torch.fx.GraphModule.add_submodule)|Yes|fp32 supported|
|[torch.fx.GraphModule.code](https://pytorch.org/docs/2.10/fx.html#torch.fx.GraphModule.code)|No|-|
|[torch.fx.GraphModule.delete_all_unused_submodules](https://pytorch.org/docs/2.10/fx.html#torch.fx.GraphModule.delete_all_unused_submodules)|Yes|fp32 supported|
|[torch.fx.GraphModule.delete_submodule](https://pytorch.org/docs/2.10/fx.html#torch.fx.GraphModule.delete_submodule)|Yes|fp32 supported|
|[torch.fx.GraphModule.graph](https://pytorch.org/docs/2.10/fx.html#torch.fx.GraphModule.graph)|No|-|
|[torch.fx.GraphModule.print_readable](https://pytorch.org/docs/2.10/fx.html#torch.fx.GraphModule.print_readable)|Yes|-|
|[torch.fx.GraphModule.recompile](https://pytorch.org/docs/2.10/fx.html#torch.fx.GraphModule.recompile)|Yes|-|
|[torch.fx.GraphModule.to_folder](https://pytorch.org/docs/2.10/fx.html#torch.fx.GraphModule.to_folder)|Yes|fp32 supported|
|[torch.fx.Graph](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph)|Yes|-|
|[torch.fx.Graph.\_\_init_\_](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.__init__)|Yes|-|
|[torch.fx.Graph.call_function](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.call_function)|Yes|-|
|[torch.fx.Graph.call_method](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.call_method)|Yes|-|
|[torch.fx.Graph.call_module](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.call_module)|Yes|-|
|[torch.fx.Graph.create_node](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.create_node)|Yes|-|
|[torch.fx.Graph.eliminate_dead_code](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.eliminate_dead_code)|Yes|-|
|[torch.fx.Graph.erase_node](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.erase_node)|Yes|-|
|[torch.fx.Graph.get_attr](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.get_attr)|Yes|-|
|[torch.fx.Graph.graph_copy](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.graph_copy)|Yes|-|
|[torch.fx.Graph.find_nodes](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.find_nodes)|Yes|-|
|[torch.fx.Graph.inserting_after](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.inserting_after)|Yes|-|
|[torch.fx.Graph.inserting_before](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.inserting_before)|Yes|-|
|[torch.fx.Graph.lint](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.lint)|Yes|fp32 supported|
|[torch.fx.Graph.node_copy](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.node_copy)|Yes|fp32 supported|
|[torch.fx.Graph.nodes](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.nodes)|No|-|
|[torch.fx.Graph.on_generate_code](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.on_generate_code)|Yes|fp32 supported|
|[torch.fx.Graph.output](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.output)|Yes|-|
|[torch.fx.Graph.output_node](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.output_node)|Yes|-|
|[torch.fx.Graph.placeholder](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.placeholder)|Yes|-|
|[torch.fx.Graph.print_tabular](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.print_tabular)|Yes|fp32 supported|
|[torch.fx.Graph.process_inputs](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.process_inputs)|Yes|-|
|[torch.fx.Graph.process_outputs](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.process_outputs)|Yes|-|
|[torch.fx.Graph.python_code](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.python_code)|Yes|fp32 supported|
|[torch.fx.Graph.set_codegen](https://pytorch.org/docs/2.10/fx.html#torch.fx.Graph.set_codegen)|Yes|-|
|[torch.fx.Node](https://pytorch.org/docs/2.10/fx.html#torch.fx.Node)|Yes|fp32 supported|
|[torch.fx.Node.all_input_nodes](https://pytorch.org/docs/2.10/fx.html#torch.fx.Node.all_input_nodes)|No|-|
|[torch.fx.Node.append](https://pytorch.org/docs/2.10/fx.html#torch.fx.Node.append)|Yes|fp32 supported|
|[torch.fx.Node.args](https://pytorch.org/docs/2.10/fx.html#torch.fx.Node.args)|No|-|
|[torch.fx.Node.format_node](https://pytorch.org/docs/2.10/fx.html#torch.fx.Node.format_node)|Yes|fp32 supported|
|[torch.fx.Node.is_impure](https://pytorch.org/docs/2.10/fx.html#torch.fx.Node.is_impure)|Yes|fp32 supported|
|[torch.fx.Node.kwargs](https://pytorch.org/docs/2.10/fx.html#torch.fx.Node.kwargs)|No|-|
|[torch.fx.Node.next](https://pytorch.org/docs/2.10/fx.html#torch.fx.Node.next)|No|-|
|[torch.fx.Node.normalized_arguments](https://pytorch.org/docs/2.10/fx.html#torch.fx.Node.normalized_arguments)|Yes|-|
|[torch.fx.Node.prepend](https://pytorch.org/docs/2.10/fx.html#torch.fx.Node.prepend)|Yes|-|
|[torch.fx.Node.prev](https://pytorch.org/docs/2.10/fx.html#torch.fx.Node.prev)|No|-|
|[torch.fx.Node.replace_all_uses_with](https://pytorch.org/docs/2.10/fx.html#torch.fx.Node.replace_all_uses_with)|Yes|-|
|[torch.fx.Node.replace_input_with](https://pytorch.org/docs/2.10/fx.html#torch.fx.Node.replace_input_with)|Yes|fp32 supported|
|[torch.fx.Node.stack_trace](https://pytorch.org/docs/2.10/fx.html#torch.fx.Node.stack_trace)|Yes|-|
|[torch.fx.Node.update_arg](https://pytorch.org/docs/2.10/fx.html#torch.fx.Node.update_arg)|Yes|fp32 supported|
|[torch.fx.Node.update_kwarg](https://pytorch.org/docs/2.10/fx.html#torch.fx.Node.update_kwarg)|Yes|fp32 supported|
|[torch.fx.Tracer](https://pytorch.org/docs/2.10/fx.html#torch.fx.Tracer)|Yes|-|
|[torch.fx.Tracer.call_module](https://pytorch.org/docs/2.10/fx.html#torch.fx.Tracer.call_module)|Yes|fp32 supported|
|[torch.fx.Tracer.create_arg](https://pytorch.org/docs/2.10/fx.html#torch.fx.Tracer.create_arg)|Yes|-|
|[torch.fx.Tracer.create_args_for_root](https://pytorch.org/docs/2.10/fx.html#torch.fx.Tracer.create_args_for_root)|Yes|fp32 supported|
|[torch.fx.Tracer.create_node](https://pytorch.org/docs/2.10/fx.html#torch.fx.Tracer.create_node)|Yes|-|
|[torch.fx.Tracer.create_proxy](https://pytorch.org/docs/2.10/fx.html#torch.fx.Tracer.create_proxy)|Yes|fp32 supported|
|[torch.fx.Tracer.getattr](https://pytorch.org/docs/2.10/fx.html#torch.fx.Tracer.getattr)|Yes|-|
|[torch.fx.Tracer.is_leaf_module](https://pytorch.org/docs/2.10/fx.html#torch.fx.Tracer.is_leaf_module)|Yes|-|
|[torch.fx.Tracer.iter](https://pytorch.org/docs/2.10/fx.html#torch.fx.Tracer.iter)|Yes|fp32 supported|
|[torch.fx.Tracer.keys](https://pytorch.org/docs/2.10/fx.html#torch.fx.Tracer.keys)|Yes|fp32 supported|
|[torch.fx.Tracer.path_of_module](https://pytorch.org/docs/2.10/fx.html#torch.fx.Tracer.path_of_module)|Yes|-|
|[torch.fx.Tracer.proxy](https://pytorch.org/docs/2.10/fx.html#torch.fx.Tracer.proxy)|Yes|fp32 supported|
|[torch.fx.Tracer.to_bool](https://pytorch.org/docs/2.10/fx.html#torch.fx.Tracer.to_bool)|Yes|fp32 supported|
|[torch.fx.Tracer.trace](https://pytorch.org/docs/2.10/fx.html#torch.fx.Tracer.trace)|Yes|-|
|[torch.fx.Proxy](https://pytorch.org/docs/2.10/fx.html#torch.fx.Proxy)|Yes|-|
|[torch.fx.proxy.ParameterProxy](https://pytorch.org/docs/2.10/fx.html#torch.fx.proxy.ParameterProxy)|Yes|-|
|[torch.fx.passes.split_module.split_module](https://pytorch.org/docs/2.7/fx.html#torch.fx.passes.split_module)|Yes|-|
|[torch.fx.Interpreter](https://pytorch.org/docs/2.10/fx.html#torch.fx.Interpreter)|Yes|-|
|[torch.fx.Interpreter.call_function](https://pytorch.org/docs/2.10/fx.html#torch.fx.Interpreter.call_function)|Yes|fp32 supported|
|[torch.fx.Interpreter.call_method](https://pytorch.org/docs/2.10/fx.html#torch.fx.Interpreter.call_method)|Yes|fp32 supported|
|[torch.fx.Interpreter.call_module](https://pytorch.org/docs/2.10/fx.html#torch.fx.Interpreter.call_module)|Yes|fp32 supported|
|[torch.fx.Interpreter.fetch_args_kwargs_from_env](https://pytorch.org/docs/2.10/fx.html#torch.fx.Interpreter.fetch_args_kwargs_from_env)|Yes|fp32 supported|
|[torch.fx.Interpreter.fetch_attr](https://pytorch.org/docs/2.10/fx.html#torch.fx.Interpreter.fetch_attr)|Yes|fp32 supported|
|[torch.fx.Interpreter.get_attr](https://pytorch.org/docs/2.10/fx.html#torch.fx.Interpreter.get_attr)|Yes|fp32 supported|
|[torch.fx.Interpreter.map_nodes_to_values](https://pytorch.org/docs/2.10/fx.html#torch.fx.Interpreter.map_nodes_to_values)|Yes|fp32 supported|
|[torch.fx.Interpreter.output](https://pytorch.org/docs/2.10/fx.html#torch.fx.Interpreter.output)|Yes|fp32 supported|
|[torch.fx.Interpreter.placeholder](https://pytorch.org/docs/2.10/fx.html#torch.fx.Interpreter.placeholder)|Yes|fp32 supported|
|[torch.fx.Interpreter.run](https://pytorch.org/docs/2.10/fx.html#torch.fx.Interpreter.run)|Yes|-|
|[torch.fx.Interpreter.run_node](https://pytorch.org/docs/2.10/fx.html#torch.fx.Interpreter.run_node)|Yes|fp32 supported|
|[torch.fx.Transformer](https://pytorch.org/docs/2.10/fx.html#torch.fx.Transformer)|Yes|-|
|[torch.fx.Transformer.call_function](https://pytorch.org/docs/2.10/fx.html#torch.fx.Transformer.call_function)|Yes|fp32 supported|
|[torch.fx.Transformer.call_module](https://pytorch.org/docs/2.10/fx.html#torch.fx.Transformer.call_module)|Yes|fp32 supported|
|[torch.fx.Transformer.get_attr](https://pytorch.org/docs/2.10/fx.html#torch.fx.Transformer.get_attr)|Yes|fp32 supported|
|[torch.fx.Transformer.placeholder](https://pytorch.org/docs/2.10/fx.html#torch.fx.Transformer.placeholder)|Yes|fp32 supported|
|[torch.fx.Transformer.transform](https://pytorch.org/docs/2.10/fx.html#torch.fx.Transformer.transform)|Yes|-|
|[torch.fx.replace_pattern](https://pytorch.org/docs/2.10/fx.html#torch.fx.replace_pattern)|Yes|-|
|[torch.fx.passes.regional_inductor.regional_inductor](https://pytorch.org/docs/2.10/fx.html#torch.fx.passes.regional_inductor.regional_inductor)|Yes|-|
|[torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings](https://pytorch.org/docs/2.10/generated/torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings.html)|Yes|-|
|[torch.fx.experimental.symbolic_shapes.constrain_range](https://pytorch.org/docs/2.10/generated/torch.fx.experimental.symbolic_shapes.constrain_range.html)|Yes|Requires SymInt obtained via torch.compile|
|[torch.fx.experimental.symbolic_shapes.constrain_unify](https://pytorch.org/docs/2.10/generated/torch.fx.experimental.symbolic_shapes.constrain_unify.html)|Yes|Requires SymInt obtained via torch.compile|
|[torch.fx.experimental.symbolic_shapes.ConvertIntKey](https://pytorch.org/docs/2.10/generated/torch.fx.experimental.symbolic_shapes.ConvertIntKey.html)|Yes|-|
|[torch.fx.experimental.symbolic_shapes.ConvertIntKey.get](https://pytorch.org/docs/2.10/generated/torch.fx.experimental.symbolic_shapes.ConvertIntKey.html#torch.fx.experimental.symbolic_shapes.ConvertIntKey.get)|Yes|-|
|[torch.fx.experimental.symbolic_shapes.CallMethodKey](https://pytorch.org/docs/2.10/generated/torch.fx.experimental.symbolic_shapes.CallMethodKey.html)|Yes|-|
|[torch.fx.experimental.symbolic_shapes.CallMethodKey.get](https://pytorch.org/docs/2.10/generated/torch.fx.experimental.symbolic_shapes.CallMethodKey.html#torch.fx.experimental.symbolic_shapes.CallMethodKey.get)|Yes|-|
|[torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr](https://pytorch.org/docs/2.10/generated/torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr.html)|Yes|-|
|[torch.fx.experimental.symbolic_shapes.check_consistent](https://pytorch.org/docs/2.10/generated/torch.fx.experimental.symbolic_shapes.check_consistent.html)|Yes|-|
