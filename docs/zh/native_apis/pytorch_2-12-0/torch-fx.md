# torch.fx

> [!NOTE]
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbolic_sizes_strides_storage_offset|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symboolnode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symfloatnode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symintnode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unbacked_symbool|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.symbolic_trace|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.wrap|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.GraphModule|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.GraphModule.__init__|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.GraphModule.add_submodule|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.GraphModule.code|否|-|
|torch.fx.GraphModule.delete_all_unused_submodules|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.GraphModule.delete_submodule|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.GraphModule.graph|否|-|
|torch.fx.GraphModule.print_readable|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.GraphModule.recompile|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.GraphModule.to_folder|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Graph|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Graph.__init__|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Graph.call_function|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Graph.call_method|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Graph.call_module|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Graph.create_node|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Graph.eliminate_dead_code|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Graph.erase_node|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Graph.get_attr|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Graph.graph_copy|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Graph.find_nodes|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Graph.inserting_after|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Graph.inserting_before|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Graph.lint|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Graph.node_copy|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Graph.nodes|否|-|
|torch.fx.Graph.on_generate_code|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Graph.output|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Graph.output_node|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Graph.placeholder|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Graph.print_tabular|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Graph.process_inputs|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Graph.process_outputs|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Graph.python_code|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Graph.set_codegen|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Node|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Node.all_input_nodes|否|-|
|torch.fx.Node.append|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Node.args|否|-|
|torch.fx.Node.format_node|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Node.is_impure|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Node.kwargs|否|-|
|torch.fx.Node.next|否|-|
|torch.fx.Node.normalized_arguments|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Node.prepend|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Node.prev|否|-|
|torch.fx.Node.replace_all_uses_with|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Node.replace_input_with|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Node.stack_trace|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Node.update_arg|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Node.update_kwarg|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Tracer|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Tracer.call_module|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Tracer.create_arg|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Tracer.create_args_for_root|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Tracer.create_node|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Tracer.create_proxy|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Tracer.getattr|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Tracer.is_leaf_module|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Tracer.iter|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Tracer.keys|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Tracer.path_of_module|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Tracer.proxy|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Tracer.to_bool|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Tracer.trace|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Proxy|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Interpreter|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Interpreter.call_function|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Interpreter.call_method|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Interpreter.call_module|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Interpreter.fetch_args_kwargs_from_env|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Interpreter.fetch_attr|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Interpreter.get_attr|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Interpreter.map_nodes_to_values|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Interpreter.output|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Interpreter.placeholder|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Interpreter.run|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Interpreter.run_node|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Transformer|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.Transformer.call_function|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Transformer.call_module|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Transformer.get_attr|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Transformer.placeholder|是<br>暂不支持<term>Ascend 950DT</term>|支持fp32|
|torch.fx.Transformer.transform|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.replace_pattern|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.passes.regional_inductor.regional_inductor|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.experimental.symbolic_shapes.constrain_range|是<br>暂不支持<term>Ascend 950DT</term>|需通过torch.compile获取SymInt|
|torch.fx.experimental.symbolic_shapes.constrain_unify|是<br>暂不支持<term>Ascend 950DT</term>|需通过torch.compile获取SymInt|
|torch.fx.experimental.symbolic_shapes.ConvertIntKey|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.experimental.symbolic_shapes.ConvertIntKey.get|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.experimental.symbolic_shapes.CallMethodKey|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.experimental.symbolic_shapes.CallMethodKey.get|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.fx.experimental.symbolic_shapes.check_consistent|是<br>暂不支持<term>Ascend 950DT</term>|-|
