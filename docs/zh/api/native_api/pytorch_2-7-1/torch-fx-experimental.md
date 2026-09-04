# torch.fx.experimental

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://docs.pytorch.org/docs/2.7/fx.experimental.html)。
> - 原生社区提示torch.fx.experimental模块下的API属于实验性质，存在随时变更的风险，建议参考原生文档后谨慎使用。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [torch.fx.experimental.symbolic_shapes](#torchfxexperimentalsymbolic_shapes)
- [torch.fx.experimental.proxy_tensor](#torchfxexperimentalproxy_tensor)

</div>

<div style="display:none;">

## &#8203;torch.fx.experimental

</div>

## torch.fx.experimental.symbolic_shapes

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.ShapeEnv

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">format_guards()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.format_guards](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.format_guards)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">freeze()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">freeze_runtime_asserts()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze_runtime_asserts](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze_runtime_asserts)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">get_axioms()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.get_axioms](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.get_axioms)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">get_implications()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.get_implications](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.get_implications)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">create_symbolic_sizes_strides_storage_offset()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbolic_sizes_strides_storage_offset](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbolic_sizes_strides_storage_offset)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">create_symboolnode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symboolnode](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symboolnode)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">create_symfloatnode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symfloatnode](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symfloatnode)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">create_symintnode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symintnode](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symintnode)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">create_unbacked_symbool()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unbacked_symbool](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unbacked_symbool)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">deserialize_symexpr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.deserialize_symexpr](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.deserialize_symexpr)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">evaluate_guards_expression()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_expression](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_expression)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">evaluate_guards_for_args()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_for_args](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_for_args)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">evaluate_sym_node()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_sym_node](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_sym_node)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">evaluate_symexpr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_symexpr](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_symexpr)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">size_hint()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.size_hint](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.size_hint)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">suppress_guards()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.suppress_guards](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.suppress_guards)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">produce_guards_expression()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_expression](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_expression)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">produce_guards_verbose()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_verbose](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_verbose)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">replace()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.replace](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.replace)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_unbacked_var_to_val()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.set_unbacked_var_to_val](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.set_unbacked_var_to_val)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">simplify()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.simplify](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.simplify)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">create_symbol()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbol](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbol)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">bound_sympy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.bound_sympy](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.bound_sympy)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">check_equal()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.check_equal](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.check_equal)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">cleanup()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.cleanup](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.cleanup)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">bind_symbols()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.bind_symbols](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.bind_symbols)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### torch.fx.experimental.symbolic_shapes.lru_cache

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.lru_cache](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.lru_cache.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.ShapeEnvSettings

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnvSettings](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnvSettings.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.fx.experimental.symbolic_shapes.check_consistent

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.check_consistent](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.check_consistent.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.fx.experimental.symbolic_shapes.is_accessor_node

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.is_accessor_node](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.is_accessor_node.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.fx.experimental.symbolic_shapes.is_concrete_bool

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.is_concrete_bool](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.is_concrete_bool.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.fx.experimental.symbolic_shapes.is_concrete_float

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.is_concrete_float](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.is_concrete_float.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.fx.experimental.symbolic_shapes.is_concrete_int

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.is_concrete_int](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.is_concrete_int.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">render()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.render](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.html#torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.render)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.fx.experimental.symbolic_shapes.sym_eq

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.sym_eq](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.sym_eq.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.fx.experimental.symbolic_shapes.statically_known_true

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.statically_known_true](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.statically_known_true.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.DivideByKey

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DivideByKey](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DivideByKey.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DivideByKey.get](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DivideByKey.html#torch.fx.experimental.symbolic_shapes.DivideByKey.get)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.EqualityConstraint

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.EqualityConstraint](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.EqualityConstraint.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.fx.experimental.symbolic_shapes.guard_size_oblivious

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.guard_size_oblivious](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.guard_size_oblivious.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.fx.experimental.symbolic_shapes.has_free_symbols

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.has_free_symbols](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.has_free_symbols.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts

<div style="margin-left: 2em">

> <font size="3">boxed_run()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.boxed_run](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.boxed_run)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">call_function()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_function](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_function)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">call_method()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_method](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_method)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">call_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_module](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_module)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">fetch_args_kwargs_from_env()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_args_kwargs_from_env](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_args_kwargs_from_env)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">fetch_attr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_attr](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_attr)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">get_attr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.get_attr](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.get_attr)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">map_nodes_to_values()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.map_nodes_to_values](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.map_nodes_to_values)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.fx.experimental.symbolic_shapes.constrain_range

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.constrain_range](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.constrain_range.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**： 需通过`torch.compile`获取SymInt

</div>

### torch.fx.experimental.symbolic_shapes.constrain_unify

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.constrain_unify](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.constrain_unify.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

**限制与说明**： 需通过`torch.compile`获取SymInt

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.ConvertIntKey

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ConvertIntKey](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ConvertIntKey.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ConvertIntKey.get](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ConvertIntKey.html#torch.fx.experimental.symbolic_shapes.ConvertIntKey.get)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.DimConstraints

<div style="margin-left: 2em">

> <font size="3">forced_specializations()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimConstraints.forced_specializations](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.forced_specializations)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">prettify_results()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimConstraints.prettify_results](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.prettify_results)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">rewrite_with_congruences()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimConstraints.rewrite_with_congruences](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.rewrite_with_congruences)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">solve()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimConstraints.solve](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.solve)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.DimDynamic

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimDynamic](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DimDynamic.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.CallMethodKey

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.CallMethodKey](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.CallMethodKey.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.CallMethodKey.get](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.CallMethodKey.html#torch.fx.experimental.symbolic_shapes.CallMethodKey.get)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

## torch.fx.experimental.proxy_tensor

### torch.fx.experimental.proxy_tensor.get_proxy_mode

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.get_proxy_mode](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.proxy_tensor.get_proxy_mode.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.fx.experimental.proxy_tensor.handle_sym_dispatch

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.handle_sym_dispatch](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.proxy_tensor.handle_sym_dispatch.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.fx.experimental.proxy_tensor.make_fx

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.make_fx](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.proxy_tensor.make_fx.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.fx.experimental.proxy_tensor.maybe_disable_thunkify

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.maybe_disable_thunkify](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.proxy_tensor.maybe_disable_thunkify.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.fx.experimental.proxy_tensor.maybe_enable_thunkify

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.maybe_enable_thunkify](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.proxy_tensor.maybe_enable_thunkify.html)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>
