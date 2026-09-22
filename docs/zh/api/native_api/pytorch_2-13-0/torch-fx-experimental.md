# torch.fx.experimental

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.13/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://docs.pytorch.org/docs/2.13/fx.experimental.html)。
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

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id3 -->

> <font size="3">add_backed_var_to_val()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.add_backed_var_to_val](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.add_backed_var_to_val)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id6 -->

</div>

> <font size="3">format_guards()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.format_guards](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.format_guards)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id9 -->

</div>

> <font size="3">freeze()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id12 -->

</div>

> <font size="3">freeze_runtime_asserts()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze_runtime_asserts](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze_runtime_asserts)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id15 -->

</div>

> <font size="3">get_axioms()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.get_axioms](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.get_axioms)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id18 -->

</div>

> <font size="3">get_implications()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.get_implications](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.get_implications)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id21 -->

</div>

> <font size="3">create_symbolic_sizes_strides_storage_offset()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbolic_sizes_strides_storage_offset](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbolic_sizes_strides_storage_offset)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id24 -->

</div>

> <font size="3">create_symboolnode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symboolnode](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symboolnode)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id27 -->

</div>

> <font size="3">create_symfloatnode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symfloatnode](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symfloatnode)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id30 -->

</div>

> <font size="3">create_symintnode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symintnode](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symintnode)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id33 -->

</div>

> <font size="3">create_unbacked_symbool()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unbacked_symbool](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unbacked_symbool)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id36 -->

</div>

> <font size="3">deserialize_symexpr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.deserialize_symexpr](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.deserialize_symexpr)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id39 -->

</div>

> <font size="3">evaluate_guards_expression()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_expression](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_expression)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id42 -->

</div>

> <font size="3">evaluate_guards_for_args()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_for_args](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_for_args)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id45 -->

</div>

> <font size="3">evaluate_sym_node()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_sym_node](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_sym_node)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id48 -->

</div>

> <font size="3">evaluate_symexpr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_symexpr](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_symexpr)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id51 -->

</div>

> <font size="3">size_hint()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.size_hint](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.size_hint)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id54 -->

</div>

> <font size="3">suppress_guards()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.suppress_guards](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.suppress_guards)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id57 -->

</div>

> <font size="3">produce_guards_expression()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_expression](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_expression)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id60 -->

**限制与说明**： `input`仅支持fp32

</div>

> <font size="3">produce_guards_verbose()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_verbose](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_verbose)

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2 训练系列产品</term>：不支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 训练系列产品</term>：不支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id63 -->

</div>

> <font size="3">replace()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.replace](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.replace)

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id66 -->

</div>

> <font size="3">simplify()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.simplify](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.simplify)

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id69 -->

</div>

> <font size="3">create_symbol()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbol](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbol)

**产品支持情况**：

<!-- npu="910b" id70 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id72 -->

</div>

> <font size="3">bound_sympy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.bound_sympy](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.bound_sympy)

**产品支持情况**：

<!-- npu="910b" id73 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id75 -->

</div>

> <font size="3">check_equal()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.check_equal](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.check_equal)

**产品支持情况**：

<!-- npu="910b" id76 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id78 -->

</div>

> <font size="3">cleanup()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.cleanup](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.cleanup)

**产品支持情况**：

<!-- npu="910b" id79 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id81 -->

</div>

> <font size="3">bind_symbols()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.bind_symbols](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.bind_symbols)

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id84 -->

</div>

</div>

### torch.fx.experimental.symbolic_shapes.lru_cache

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.lru_cache](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.lru_cache.html)

**产品支持情况**：

<!-- npu="910b" id85 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id87 -->

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.ShapeEnvSettings

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnvSettings](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ShapeEnvSettings.html)

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id90 -->

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext.html)

**产品支持情况**：

<!-- npu="910b" id91 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id93 -->

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext.html)

**产品支持情况**：

<!-- npu="910b" id94 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id96 -->

</div>

### torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr.html)

**产品支持情况**：

<!-- npu="910b" id97 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id99 -->

</div>

### torch.fx.experimental.symbolic_shapes.check_consistent

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.check_consistent](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.check_consistent.html)

**产品支持情况**：

<!-- npu="910b" id100 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id102 -->

</div>

### torch.fx.experimental.symbolic_shapes.is_accessor_node

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.is_accessor_node](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.is_accessor_node.html)

**产品支持情况**：

<!-- npu="910b" id103 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id105 -->

</div>

### torch.fx.experimental.symbolic_shapes.is_concrete_bool

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.is_concrete_bool](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.is_concrete_bool.html)

**产品支持情况**：

<!-- npu="910b" id106 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id108 -->

</div>

### torch.fx.experimental.symbolic_shapes.is_concrete_float

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.is_concrete_float](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.is_concrete_float.html)

**产品支持情况**：

<!-- npu="910b" id109 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id111 -->

</div>

### torch.fx.experimental.symbolic_shapes.is_concrete_int

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.is_concrete_int](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.is_concrete_int.html)

**产品支持情况**：

<!-- npu="910b" id112 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id114 -->

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.html)

**产品支持情况**：

<!-- npu="910b" id115 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id117 -->

> <font size="3">render()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.render](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.html#torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.render)

**产品支持情况**：

<!-- npu="910b" id118 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id120 -->

</div>

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext.html)

**产品支持情况**：

<!-- npu="910b" id121 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id123 -->

</div>

### torch.fx.experimental.symbolic_shapes.sym_eq

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.sym_eq](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.sym_eq.html)

**产品支持情况**：

<!-- npu="910b" id124 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id126 -->

</div>

### torch.fx.experimental.symbolic_shapes.statically_known_true

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.statically_known_true](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.statically_known_true.html)

**产品支持情况**：

<!-- npu="910b" id127 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id129 -->

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.DivideByKey

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DivideByKey](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.DivideByKey.html)

**产品支持情况**：

<!-- npu="910b" id130 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id132 -->

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DivideByKey.get](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.DivideByKey.html#torch.fx.experimental.symbolic_shapes.DivideByKey.get)

**产品支持情况**：

<!-- npu="910b" id133 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id135 -->

</div>

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.EqualityConstraint

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.EqualityConstraint](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.EqualityConstraint.html)

**产品支持情况**：

<!-- npu="910b" id136 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id138 -->

</div>

### torch.fx.experimental.symbolic_shapes.guard_size_oblivious

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.guard_size_oblivious](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.guard_size_oblivious.html)

**产品支持情况**：

<!-- npu="910b" id139 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id141 -->

</div>

### torch.fx.experimental.symbolic_shapes.has_free_symbols

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.has_free_symbols](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.has_free_symbols.html)

**产品支持情况**：

<!-- npu="910b" id142 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id143 -->
<!-- npu="950" id144 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id144 -->

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts

<div style="margin-left: 2em">

> <font size="3">boxed_run()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.boxed_run](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.boxed_run)

**产品支持情况**：

<!-- npu="910b" id145 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id147 -->

</div>

> <font size="3">call_function()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_function](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_function)

**产品支持情况**：

<!-- npu="910b" id148 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id150 -->

</div>

> <font size="3">call_method()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_method](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_method)

**产品支持情况**：

<!-- npu="910b" id151 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id153 -->

</div>

> <font size="3">call_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_module](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_module)

**产品支持情况**：

<!-- npu="910b" id154 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id156 -->

</div>

> <font size="3">fetch_args_kwargs_from_env()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_args_kwargs_from_env](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_args_kwargs_from_env)

**产品支持情况**：

<!-- npu="910b" id157 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id159 -->

</div>

> <font size="3">fetch_attr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_attr](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_attr)

**产品支持情况**：

<!-- npu="910b" id160 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id162 -->

</div>

> <font size="3">get_attr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.get_attr](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.get_attr)

**产品支持情况**：

<!-- npu="910b" id163 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id165 -->

</div>

> <font size="3">map_nodes_to_values()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.map_nodes_to_values](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.map_nodes_to_values)

**产品支持情况**：

<!-- npu="910b" id166 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id168 -->

</div>

</div>

### torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings.html)

**产品支持情况**：

<!-- npu="910b" id169 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id171 -->

</div>

### torch.fx.experimental.symbolic_shapes.constrain_range

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.constrain_range](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.constrain_range.html)

**产品支持情况**：

<!-- npu="910b" id172 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id174 -->

**限制与说明**： 需通过`torch.compile`获取SymInt

</div>

### torch.fx.experimental.symbolic_shapes.constrain_unify

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.constrain_unify](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.constrain_unify.html)

**产品支持情况**：

<!-- npu="910b" id175 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id177 -->

**限制与说明**： 需通过`torch.compile`获取SymInt

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.ConvertIntKey

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ConvertIntKey](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ConvertIntKey.html)

**产品支持情况**：

<!-- npu="910b" id178 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id180 -->

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ConvertIntKey.get](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.ConvertIntKey.html#torch.fx.experimental.symbolic_shapes.ConvertIntKey.get)

**产品支持情况**：

<!-- npu="910b" id181 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id183 -->

</div>

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.DimConstraints

<div style="margin-left: 2em">

> <font size="3">forced_specializations()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimConstraints.forced_specializations](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.forced_specializations)

**产品支持情况**：

<!-- npu="910b" id184 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id186 -->

</div>

> <font size="3">prettify_results()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimConstraints.prettify_results](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.prettify_results)

**产品支持情况**：

<!-- npu="910b" id187 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id189 -->

</div>

> <font size="3">rewrite_with_congruences()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimConstraints.rewrite_with_congruences](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.rewrite_with_congruences)

**产品支持情况**：

<!-- npu="910b" id190 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id192 -->

</div>

> <font size="3">solve()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimConstraints.solve](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.solve)

**产品支持情况**：

<!-- npu="910b" id193 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id195 -->

</div>

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.DimDynamic

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimDynamic](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.DimDynamic.html)

**产品支持情况**：

<!-- npu="910b" id196 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id197 -->
<!-- npu="950" id198 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id198 -->

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.CallMethodKey

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.CallMethodKey](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.CallMethodKey.html)

**产品支持情况**：

<!-- npu="910b" id199 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id200 -->
<!-- npu="950" id201 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id201 -->

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.CallMethodKey.get](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.CallMethodKey.html#torch.fx.experimental.symbolic_shapes.CallMethodKey.get)

**产品支持情况**：

<!-- npu="910b" id202 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id203 -->
<!-- npu="950" id204 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id204 -->

</div>

</div>

### torch.fx.experimental.symbolic_shapes.is_symbolic

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.is_symbolic](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.symbolic_shapes.is_symbolic.html)

**产品支持情况**：

<!-- npu="910b" id205 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id206 -->
<!-- npu="950" id207 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id207 -->

</div>

## torch.fx.experimental.proxy_tensor

### torch.fx.experimental.proxy_tensor.get_proxy_mode

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.get_proxy_mode](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.proxy_tensor.get_proxy_mode.html)

**产品支持情况**：

<!-- npu="910b" id208 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id209 -->
<!-- npu="950" id210 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id210 -->

</div>

### torch.fx.experimental.proxy_tensor.handle_sym_dispatch

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.handle_sym_dispatch](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.proxy_tensor.handle_sym_dispatch.html)

**产品支持情况**：

<!-- npu="910b" id211 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id212 -->
<!-- npu="950" id213 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id213 -->

</div>

### torch.fx.experimental.proxy_tensor.make_fx

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.make_fx](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.proxy_tensor.make_fx.html)

**产品支持情况**：

<!-- npu="910b" id214 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id215 -->
<!-- npu="950" id216 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id216 -->

</div>

### torch.fx.experimental.proxy_tensor.maybe_disable_thunkify

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.maybe_disable_thunkify](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.proxy_tensor.maybe_disable_thunkify.html)

**产品支持情况**：

<!-- npu="910b" id217 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id218 -->
<!-- npu="950" id219 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id219 -->

</div>

### torch.fx.experimental.proxy_tensor.maybe_enable_thunkify

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.maybe_enable_thunkify](https://pytorch.org/docs/2.13/generated/torch.fx.experimental.proxy_tensor.maybe_enable_thunkify.html)

**产品支持情况**：

<!-- npu="910b" id220 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id220 -->
<!-- npu="A3" id221 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id221 -->
<!-- npu="950" id222 -->
- <term>Ascend 950DT</term>：不支持
<!-- end id222 -->

</div>
