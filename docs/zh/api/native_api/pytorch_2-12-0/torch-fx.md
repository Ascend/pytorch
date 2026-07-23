# torch.fx

> [!NOTE]
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)
- [Overview](#overview)
- [Limitations of Symbolic Tracing](#limitations-of-symbolic-tracing)
- [torch.fx.node](#torchfxnode)
- [Writing Transformations](#writing-transformations)
- [torch.fx.passes.regional_inductor](#torchfxpassesregional_inductor)

## base API

### torch.fx.experimental.proxy_tensor.get_proxy_mode

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.get_proxy_mode](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.proxy_tensor.get_proxy_mode.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fx.experimental.proxy_tensor.handle_sym_dispatch

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.handle_sym_dispatch](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.proxy_tensor.handle_sym_dispatch.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fx.experimental.proxy_tensor.make_fx

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.make_fx](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.proxy_tensor.make_fx.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fx.experimental.proxy_tensor.maybe_disable_thunkify

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.maybe_disable_thunkify](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.proxy_tensor.maybe_disable_thunkify.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fx.experimental.proxy_tensor.maybe_enable_thunkify

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.maybe_enable_thunkify](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.proxy_tensor.maybe_enable_thunkify.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.fx.experimental.symbolic_shapes.ShapeEnvSettings

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnvSettings](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnvSettings.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fx.node._type_repr

<div style="margin-left: 2em">

**原生文档**：[torch.fx.node._type_repr](https://pytorch.org/docs/2.12/fx.html#torch.fx.node._type_repr)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts

<div style="margin-left: 2em">

> <font size="3">call_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_module](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_module)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">fetch_args_kwargs_from_env()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_args_kwargs_from_env](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_args_kwargs_from_env)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">fetch_attr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_attr](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_attr)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">get_attr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.get_attr](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.get_attr)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">map_nodes_to_values()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.map_nodes_to_values](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.map_nodes_to_values)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fx.experimental.symbolic_shapes.constrain_range

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.constrain_range](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.constrain_range.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 需通过torch.compile获取SymInt

</div>

### torch.fx.experimental.symbolic_shapes.constrain_unify

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.constrain_unify](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.constrain_unify.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 需通过torch.compile获取SymInt

</div>

### _`class`_ torch.fx.experimental.symbolic_shapes.ConvertIntKey

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ConvertIntKey](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ConvertIntKey.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ConvertIntKey.get](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ConvertIntKey.html#torch.fx.experimental.symbolic_shapes.ConvertIntKey.get)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.fx.experimental.symbolic_shapes.DimConstraints

<div style="margin-left: 2em">

> <font size="3">forced_specializations()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimConstraints.forced_specializations](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.forced_specializations)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">prettify_results()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimConstraints.prettify_results](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.prettify_results)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">rewrite_with_congruences()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimConstraints.rewrite_with_congruences](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.rewrite_with_congruences)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">solve()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimConstraints.solve](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.solve)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.fx.experimental.symbolic_shapes.DimDynamic

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimDynamic](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.DimDynamic.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.fx.experimental.symbolic_shapes.CallMethodKey

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.CallMethodKey](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.CallMethodKey.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.CallMethodKey.get](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.CallMethodKey.html#torch.fx.experimental.symbolic_shapes.CallMethodKey.get)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fx.experimental.symbolic_shapes.check_consistent

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.check_consistent](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.check_consistent.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fx.experimental.symbolic_shapes.is_accessor_node

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.is_accessor_node](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.is_accessor_node.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fx.experimental.symbolic_shapes.is_concrete_bool

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.is_concrete_bool](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.is_concrete_bool.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fx.experimental.symbolic_shapes.is_concrete_float

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.is_concrete_float](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.is_concrete_float.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fx.experimental.symbolic_shapes.is_concrete_int

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.is_concrete_int](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.is_concrete_int.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fx.experimental.symbolic_shapes.is_symbolic

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.is_symbolic](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.is_symbolic.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.fx.experimental.symbolic_shapes.ShapeEnv

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">create_symbolic_sizes_strides_storage_offset()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbolic_sizes_strides_storage_offset](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbolic_sizes_strides_storage_offset)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">create_symboolnode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symboolnode](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symboolnode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">create_symfloatnode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symfloatnode](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symfloatnode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">create_symintnode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symintnode](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symintnode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">create_unbacked_symbool()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unbacked_symbool](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unbacked_symbool)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">deserialize_symexpr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.deserialize_symexpr](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.deserialize_symexpr)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">evaluate_guards_expression()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_expression](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_expression)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">evaluate_guards_for_args()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_for_args](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_for_args)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">evaluate_sym_node()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_sym_node](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_sym_node)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">evaluate_symexpr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_symexpr](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_symexpr)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">size_hint()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.size_hint](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.size_hint)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">suppress_guards()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.suppress_guards](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.suppress_guards)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">produce_guards_expression()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_expression](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_expression)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">produce_guards_verbose()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_verbose](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_verbose)

**是否支持**：否

</div>

> <font size="3">replace()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.replace](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.replace)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">add_backed_var_to_val()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.add_backed_var_to_val](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.add_backed_var_to_val)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">set_unbacked_var_to_val()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.set_unbacked_var_to_val](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.set_unbacked_var_to_val)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">simplify()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.simplify](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.simplify)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">create_symbol()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbol](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbol)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">bound_sympy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.bound_sympy](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.bound_sympy)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">check_equal()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.check_equal](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.check_equal)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">cleanup()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.cleanup](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.cleanup)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">bind_symbols()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.bind_symbols](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.bind_symbols)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">render()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.render](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.html#torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.render)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fx.experimental.symbolic_shapes.sym_eq

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.sym_eq](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.sym_eq.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fx.experimental.symbolic_shapes.statically_known_true

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.statically_known_true](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.statically_known_true.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.fx.experimental.symbolic_shapes.DivideByKey

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DivideByKey](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.DivideByKey.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DivideByKey.get](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.DivideByKey.html#torch.fx.experimental.symbolic_shapes.DivideByKey.get)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.fx.experimental.symbolic_shapes.EqualityConstraint

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.EqualityConstraint](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.EqualityConstraint.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fx.experimental.symbolic_shapes.guard_size_oblivious

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.guard_size_oblivious](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.guard_size_oblivious.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fx.experimental.symbolic_shapes.has_free_symbols

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.has_free_symbols](https://pytorch.org/docs/2.12/generated/torch.fx.experimental.symbolic_shapes.has_free_symbols.html)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## Overview

### torch.fx.symbolic_trace

<div style="margin-left: 2em">

**原生文档**：[torch.fx.symbolic_trace](https://pytorch.org/docs/2.12/fx.html#torch.fx.symbolic_trace)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### _`class`_ torch.fx.GraphModule

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule](https://pytorch.org/docs/2.12/fx.html#torch.fx.GraphModule)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

> <font size="3">__init__()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule.__init__](https://pytorch.org/docs/2.12/fx.html#torch.fx.GraphModule.__init__)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">add_submodule()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule.add_submodule](https://pytorch.org/docs/2.12/fx.html#torch.fx.GraphModule.add_submodule)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">code()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule.code](https://pytorch.org/docs/2.12/fx.html#torch.fx.GraphModule.code)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">delete_all_unused_submodules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule.delete_all_unused_submodules](https://pytorch.org/docs/2.12/fx.html#torch.fx.GraphModule.delete_all_unused_submodules)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">delete_submodule()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule.delete_submodule](https://pytorch.org/docs/2.12/fx.html#torch.fx.GraphModule.delete_submodule)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">graph()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule.graph](https://pytorch.org/docs/2.12/fx.html#torch.fx.GraphModule.graph)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">print_readable()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule.print_readable](https://pytorch.org/docs/2.12/fx.html#torch.fx.GraphModule.print_readable)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">recompile()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule.recompile](https://pytorch.org/docs/2.12/fx.html#torch.fx.GraphModule.recompile)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">to_folder()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.GraphModule.to_folder](https://pytorch.org/docs/2.12/fx.html#torch.fx.GraphModule.to_folder)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

</div>

### _`class`_ torch.fx.Graph

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">__init__()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.__init__](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.__init__)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">call_function()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.call_function](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.call_function)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">call_method()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.call_method](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.call_method)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">call_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.call_module](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.call_module)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">create_node()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.create_node](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.create_node)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">eliminate_dead_code()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.eliminate_dead_code](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.eliminate_dead_code)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">erase_node()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.erase_node](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.erase_node)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">get_attr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.get_attr](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.get_attr)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">graph_copy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.graph_copy](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.graph_copy)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">find_nodes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.find_nodes](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.find_nodes)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">inserting_after()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.inserting_after](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.inserting_after)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">inserting_before()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.inserting_before](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.inserting_before)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">lint()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.lint](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.lint)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">node_copy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.node_copy](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.node_copy)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">nodes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.nodes](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.nodes)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">on_generate_code()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.on_generate_code](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.on_generate_code)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">output()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.output](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.output)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">output_node()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.output_node](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.output_node)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">placeholder()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.placeholder](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.placeholder)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">print_tabular()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.print_tabular](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.print_tabular)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">process_inputs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.process_inputs](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.process_inputs)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">process_outputs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.process_outputs](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.process_outputs)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">python_code()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.python_code](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.python_code)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">set_codegen()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Graph.set_codegen](https://pytorch.org/docs/2.12/fx.html#torch.fx.Graph.set_codegen)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.fx.Tracer

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer](https://pytorch.org/docs/2.12/fx.html#torch.fx.Tracer)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">call_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.call_module](https://pytorch.org/docs/2.12/fx.html#torch.fx.Tracer.call_module)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">create_arg()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.create_arg](https://pytorch.org/docs/2.12/fx.html#torch.fx.Tracer.create_arg)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">create_args_for_root()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.create_args_for_root](https://pytorch.org/docs/2.12/fx.html#torch.fx.Tracer.create_args_for_root)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">create_node()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.create_node](https://pytorch.org/docs/2.12/fx.html#torch.fx.Tracer.create_node)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">create_proxy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.create_proxy](https://pytorch.org/docs/2.12/fx.html#torch.fx.Tracer.create_proxy)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">getattr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.getattr](https://pytorch.org/docs/2.12/fx.html#torch.fx.Tracer.getattr)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">is_leaf_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.is_leaf_module](https://pytorch.org/docs/2.12/fx.html#torch.fx.Tracer.is_leaf_module)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">iter()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.iter](https://pytorch.org/docs/2.12/fx.html#torch.fx.Tracer.iter)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">keys()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.keys](https://pytorch.org/docs/2.12/fx.html#torch.fx.Tracer.keys)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">path_of_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.path_of_module](https://pytorch.org/docs/2.12/fx.html#torch.fx.Tracer.path_of_module)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">proxy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.proxy](https://pytorch.org/docs/2.12/fx.html#torch.fx.Tracer.proxy)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">to_bool()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.to_bool](https://pytorch.org/docs/2.12/fx.html#torch.fx.Tracer.to_bool)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">trace()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Tracer.trace](https://pytorch.org/docs/2.12/fx.html#torch.fx.Tracer.trace)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

## Limitations of Symbolic Tracing

### torch.fx.wrap

<div style="margin-left: 2em">

**原生文档**：[torch.fx.wrap](https://pytorch.org/docs/2.12/fx.html#torch.fx.wrap)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

## torch.fx.node

### torch.fx.graph.map_arg

<div style="margin-left: 2em">

**原生文档**：[torch.fx.graph.map_arg](https://pytorch.org/docs/2.12/fx.html#torch.fx.graph.map_arg)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

### torch.fx.node.map_arg

<div style="margin-left: 2em">

**原生文档**：[torch.fx.node.map_arg](https://pytorch.org/docs/2.12/fx.html#torch.fx.node.map_arg)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### torch.fx.node.map_aggregate

<div style="margin-left: 2em">

**原生文档**：[torch.fx.node.map_aggregate](https://pytorch.org/docs/2.12/fx.html#torch.fx.node.map_aggregate)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## Writing Transformations

### _`class`_ torch.fx.Node

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node](https://pytorch.org/docs/2.12/fx.html#torch.fx.Node)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

> <font size="3">all_input_nodes()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.all_input_nodes](https://pytorch.org/docs/2.12/fx.html#torch.fx.Node.all_input_nodes)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">append()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.append](https://pytorch.org/docs/2.12/fx.html#torch.fx.Node.append)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">args()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.args](https://pytorch.org/docs/2.12/fx.html#torch.fx.Node.args)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">format_node()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.format_node](https://pytorch.org/docs/2.12/fx.html#torch.fx.Node.format_node)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">is_impure()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.is_impure](https://pytorch.org/docs/2.12/fx.html#torch.fx.Node.is_impure)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">kwargs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.kwargs](https://pytorch.org/docs/2.12/fx.html#torch.fx.Node.kwargs)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">next()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.next](https://pytorch.org/docs/2.12/fx.html#torch.fx.Node.next)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">normalized_arguments()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.normalized_arguments](https://pytorch.org/docs/2.12/fx.html#torch.fx.Node.normalized_arguments)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">prepend()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.prepend](https://pytorch.org/docs/2.12/fx.html#torch.fx.Node.prepend)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">prev()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.prev](https://pytorch.org/docs/2.12/fx.html#torch.fx.Node.prev)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">replace_all_uses_with()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.replace_all_uses_with](https://pytorch.org/docs/2.12/fx.html#torch.fx.Node.replace_all_uses_with)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">replace_input_with()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.replace_input_with](https://pytorch.org/docs/2.12/fx.html#torch.fx.Node.replace_input_with)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">stack_trace()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.stack_trace](https://pytorch.org/docs/2.12/fx.html#torch.fx.Node.stack_trace)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">update_arg()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.update_arg](https://pytorch.org/docs/2.12/fx.html#torch.fx.Node.update_arg)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">update_kwarg()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.update_kwarg](https://pytorch.org/docs/2.12/fx.html#torch.fx.Node.update_kwarg)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">insert_arg()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Node.insert_arg](https://pytorch.org/docs/2.12/fx.html#torch.fx.Node.insert_arg)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.fx.Proxy

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Proxy](https://pytorch.org/docs/2.12/fx.html#torch.fx.Proxy)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.fx.Interpreter

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter](https://pytorch.org/docs/2.12/fx.html#torch.fx.Interpreter)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">call_function()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.call_function](https://pytorch.org/docs/2.12/fx.html#torch.fx.Interpreter.call_function)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">call_method()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.call_method](https://pytorch.org/docs/2.12/fx.html#torch.fx.Interpreter.call_method)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">call_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.call_module](https://pytorch.org/docs/2.12/fx.html#torch.fx.Interpreter.call_module)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">fetch_args_kwargs_from_env()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.fetch_args_kwargs_from_env](https://pytorch.org/docs/2.12/fx.html#torch.fx.Interpreter.fetch_args_kwargs_from_env)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">fetch_attr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.fetch_attr](https://pytorch.org/docs/2.12/fx.html#torch.fx.Interpreter.fetch_attr)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">get_attr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.get_attr](https://pytorch.org/docs/2.12/fx.html#torch.fx.Interpreter.get_attr)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">map_nodes_to_values()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.map_nodes_to_values](https://pytorch.org/docs/2.12/fx.html#torch.fx.Interpreter.map_nodes_to_values)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">output()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.output](https://pytorch.org/docs/2.12/fx.html#torch.fx.Interpreter.output)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">placeholder()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.placeholder](https://pytorch.org/docs/2.12/fx.html#torch.fx.Interpreter.placeholder)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">run()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.run](https://pytorch.org/docs/2.12/fx.html#torch.fx.Interpreter.run)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">run_node()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Interpreter.run_node](https://pytorch.org/docs/2.12/fx.html#torch.fx.Interpreter.run_node)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

</div>

### _`class`_ torch.fx.Transformer

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Transformer](https://pytorch.org/docs/2.12/fx.html#torch.fx.Transformer)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">call_function()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Transformer.call_function](https://pytorch.org/docs/2.12/fx.html#torch.fx.Transformer.call_function)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">call_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Transformer.call_module](https://pytorch.org/docs/2.12/fx.html#torch.fx.Transformer.call_module)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">get_attr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Transformer.get_attr](https://pytorch.org/docs/2.12/fx.html#torch.fx.Transformer.get_attr)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">placeholder()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Transformer.placeholder](https://pytorch.org/docs/2.12/fx.html#torch.fx.Transformer.placeholder)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 支持fp32

</div>

> <font size="3">transform()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.Transformer.transform](https://pytorch.org/docs/2.12/fx.html#torch.fx.Transformer.transform)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### torch.fx.replace_pattern

<div style="margin-left: 2em">

**原生文档**：[torch.fx.replace_pattern](https://pytorch.org/docs/2.12/fx.html#torch.fx.replace_pattern)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## torch.fx.passes.regional_inductor

### torch.fx.passes.regional_inductor.regional_inductor

<div style="margin-left: 2em">

**原生文档**：[torch.fx.passes.regional_inductor.regional_inductor](https://docs.pytorch.org/docs/2.12/generated/torch.fx.passes.regional_inductor.regional_inductor.html#torch.fx.passes.regional_inductor.regional_inductor)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>
