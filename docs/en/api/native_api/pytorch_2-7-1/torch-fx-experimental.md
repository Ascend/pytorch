# torch.fx.experimental

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.7/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://docs.pytorch.org/docs/2.7/fx.experimental.html).
> - The native community notes that the APIs under the torch.fx.experimental module are experimental in nature and subject to change at any time. We recommend referring to the native documentation before using them with caution.

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [torch.fx.experimental.symbolic_shapes](#torchfxexperimentalsymbolic_shapes)
- [torch.fx.experimental.proxy_tensor](#torchfxexperimentalproxy_tensor)

</div>

<div style="display:none;">

## &#8203;torch.fx.experimental

</div>

## torch.fx.experimental.symbolic_shapes

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.ShapeEnv

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id3 -->

> <font size="3">format_guards()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.format_guards](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.format_guards)

**Support Status**:

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id6 -->

</div>

> <font size="3">freeze()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze)

**Support Status**:

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id9 -->

</div>

> <font size="3">freeze_runtime_asserts()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze_runtime_asserts](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze_runtime_asserts)

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id12 -->

</div>

> <font size="3">get_axioms()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.get_axioms](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.get_axioms)

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

> <font size="3">get_implications()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.get_implications](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.get_implications)

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

> <font size="3">create_symbolic_sizes_strides_storage_offset()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbolic_sizes_strides_storage_offset](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbolic_sizes_strides_storage_offset)

**Support Status**:

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id21 -->

</div>

> <font size="3">create_symboolnode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symboolnode](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symboolnode)

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

> <font size="3">create_symfloatnode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symfloatnode](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symfloatnode)

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id27 -->

</div>

> <font size="3">create_symintnode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symintnode](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symintnode)

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

> <font size="3">create_unbacked_symbool()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unbacked_symbool](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unbacked_symbool)

**Support Status**:

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id33 -->

</div>

> <font size="3">deserialize_symexpr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.deserialize_symexpr](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.deserialize_symexpr)

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id36 -->

</div>

> <font size="3">evaluate_guards_expression()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_expression](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_expression)

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id39 -->

</div>

> <font size="3">evaluate_guards_for_args()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_for_args](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_for_args)

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id42 -->

</div>

> <font size="3">evaluate_sym_node()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_sym_node](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_sym_node)

**Support Status**:

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id45 -->

</div>

> <font size="3">evaluate_symexpr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_symexpr](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_symexpr)

**Support Status**:

<!-- npu="910b" id46 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id48 -->

</div>

> <font size="3">size_hint()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.size_hint](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.size_hint)

**Support Status**:

<!-- npu="910b" id49 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id51 -->

</div>

> <font size="3">suppress_guards()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.suppress_guards](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.suppress_guards)

**Support Status**:

<!-- npu="910b" id52 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id54 -->

</div>

> <font size="3">produce_guards_expression()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_expression](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_expression)

**Support Status**:

<!-- npu="910b" id55 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id57 -->

**Constraints & Notes**: `input` only supports fp32

</div>

> <font size="3">produce_guards_verbose()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_verbose](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_verbose)

**Support Status**:

<!-- npu="910b" id58 -->
- <term>Atlas A2 training products</term>: Not Supported
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 training products</term>: Not Supported
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id60 -->

</div>

> <font size="3">replace()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.replace](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.replace)

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

</div>

> <font size="3">set_unbacked_var_to_val()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.set_unbacked_var_to_val](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.set_unbacked_var_to_val)

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

> <font size="3">simplify()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.simplify](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.simplify)

**Support Status**:

<!-- npu="910b" id67 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id69 -->

</div>

> <font size="3">create_symbol()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbol](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbol)

**Support Status**:

<!-- npu="910b" id70 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id72 -->

</div>

> <font size="3">bound_sympy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.bound_sympy](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.bound_sympy)

**Support Status**:

<!-- npu="910b" id73 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id75 -->

</div>

> <font size="3">check_equal()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.check_equal](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.check_equal)

**Support Status**:

<!-- npu="910b" id76 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id78 -->

</div>

> <font size="3">cleanup()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.cleanup](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.cleanup)

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

</div>

> <font size="3">bind_symbols()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.bind_symbols](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.bind_symbols)

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

</div>

### torch.fx.experimental.symbolic_shapes.lru_cache

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.lru_cache](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.lru_cache.html)

**Support Status**:

<!-- npu="910b" id85 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id87 -->

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.ShapeEnvSettings

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnvSettings](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnvSettings.html)

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

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext.html)

**Support Status**:

<!-- npu="910b" id91 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id93 -->

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext.html)

**Support Status**:

<!-- npu="910b" id94 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id96 -->

</div>

### torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr.html)

**Support Status**:

<!-- npu="910b" id97 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id99 -->

</div>

### torch.fx.experimental.symbolic_shapes.check_consistent

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.check_consistent](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.check_consistent.html)

**Support Status**:

<!-- npu="910b" id100 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id102 -->

</div>

### torch.fx.experimental.symbolic_shapes.is_accessor_node

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.is_accessor_node](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.is_accessor_node.html)

**Support Status**:

<!-- npu="910b" id103 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id105 -->

</div>

### torch.fx.experimental.symbolic_shapes.is_concrete_bool

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.is_concrete_bool](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.is_concrete_bool.html)

**Support Status**:

<!-- npu="910b" id106 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id108 -->

</div>

### torch.fx.experimental.symbolic_shapes.is_concrete_float

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.is_concrete_float](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.is_concrete_float.html)

**Support Status**:

<!-- npu="910b" id109 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id111 -->

</div>

### torch.fx.experimental.symbolic_shapes.is_concrete_int

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.is_concrete_int](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.is_concrete_int.html)

**Support Status**:

<!-- npu="910b" id112 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id114 -->

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.html)

**Support Status**:

<!-- npu="910b" id115 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id117 -->

> <font size="3">render()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.render](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.html#torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.render)

**Support Status**:

<!-- npu="910b" id118 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id120 -->

</div>

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext.html)

**Support Status**:

<!-- npu="910b" id121 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id123 -->

</div>

### torch.fx.experimental.symbolic_shapes.sym_eq

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.sym_eq](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.sym_eq.html)

**Support Status**:

<!-- npu="910b" id124 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id126 -->

</div>

### torch.fx.experimental.symbolic_shapes.statically_known_true

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.statically_known_true](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.statically_known_true.html)

**Support Status**:

<!-- npu="910b" id127 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id129 -->

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.DivideByKey

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.DivideByKey](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DivideByKey.html)

**Support Status**:

<!-- npu="910b" id130 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id132 -->

> <font size="3">get()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.DivideByKey.get](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DivideByKey.html#torch.fx.experimental.symbolic_shapes.DivideByKey.get)

**Support Status**:

<!-- npu="910b" id133 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id135 -->

</div>

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.EqualityConstraint

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.EqualityConstraint](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.EqualityConstraint.html)

**Support Status**:

<!-- npu="910b" id136 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id138 -->

</div>

### torch.fx.experimental.symbolic_shapes.guard_size_oblivious

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.guard_size_oblivious](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.guard_size_oblivious.html)

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

</div>

### torch.fx.experimental.symbolic_shapes.has_free_symbols

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.has_free_symbols](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.has_free_symbols.html)

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

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts

<div style="margin-left: 2em">

> <font size="3">boxed_run()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.boxed_run](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.boxed_run)

**Support Status**:

<!-- npu="910b" id145 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id147 -->

</div>

> <font size="3">call_function()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_function](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_function)

**Support Status**:

<!-- npu="910b" id148 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id150 -->

</div>

> <font size="3">call_method()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_method](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_method)

**Support Status**:

<!-- npu="910b" id151 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id153 -->

</div>

> <font size="3">call_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_module](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_module)

**Support Status**:

<!-- npu="910b" id154 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id156 -->

</div>

> <font size="3">fetch_args_kwargs_from_env()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_args_kwargs_from_env](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_args_kwargs_from_env)

**Support Status**:

<!-- npu="910b" id157 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id159 -->

</div>

> <font size="3">fetch_attr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_attr](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_attr)

**Support Status**:

<!-- npu="910b" id160 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id162 -->

</div>

> <font size="3">get_attr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.get_attr](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.get_attr)

**Support Status**:

<!-- npu="910b" id163 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id165 -->

</div>

> <font size="3">map_nodes_to_values()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.map_nodes_to_values](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.map_nodes_to_values)

**Support Status**:

<!-- npu="910b" id166 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id168 -->

</div>

</div>

### torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings.html)

**Support Status**:

<!-- npu="910b" id169 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id171 -->

</div>

### torch.fx.experimental.symbolic_shapes.constrain_range

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.constrain_range](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.constrain_range.html)

**Support Status**:

<!-- npu="910b" id172 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id174 -->

**Constraints & Notes**: Requires obtaining SymInt via torch.compile.

</div>

### torch.fx.experimental.symbolic_shapes.constrain_unify

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.constrain_unify](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.constrain_unify.html)

**Support Status**:

<!-- npu="910b" id175 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id177 -->

**Constraints & Notes**: Requires obtaining SymInt via torch.compile.

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.ConvertIntKey

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ConvertIntKey](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ConvertIntKey.html)

**Support Status**:

<!-- npu="910b" id178 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id180 -->

> <font size="3">get()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ConvertIntKey.get](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ConvertIntKey.html#torch.fx.experimental.symbolic_shapes.ConvertIntKey.get)

**Support Status**:

<!-- npu="910b" id181 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id183 -->

</div>

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.DimConstraints

<div style="margin-left: 2em">

> <font size="3">forced_specializations()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.DimConstraints.forced_specializations](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.forced_specializations)

**Support Status**:

<!-- npu="910b" id184 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id186 -->

</div>

> <font size="3">prettify_results()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.DimConstraints.prettify_results](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.prettify_results)

**Support Status**:

<!-- npu="910b" id187 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id189 -->

</div>

> <font size="3">rewrite_with_congruences()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.DimConstraints.rewrite_with_congruences](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.rewrite_with_congruences)

**Support Status**:

<!-- npu="910b" id190 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id192 -->

</div>

> <font size="3">solve()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.DimConstraints.solve](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.solve)

**Support Status**:

<!-- npu="910b" id193 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id195 -->

</div>

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.DimDynamic

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.DimDynamic](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DimDynamic.html)

**Support Status**:

<!-- npu="910b" id196 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id197 -->
<!-- npu="950" id198 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id198 -->

</div>

### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.CallMethodKey

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.CallMethodKey](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.CallMethodKey.html)

**Support Status**:

<!-- npu="910b" id199 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id200 -->
<!-- npu="950" id201 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id201 -->

> <font size="3">get()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.CallMethodKey.get](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.CallMethodKey.html#torch.fx.experimental.symbolic_shapes.CallMethodKey.get)

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

## torch.fx.experimental.proxy_tensor

### torch.fx.experimental.proxy_tensor.get_proxy_mode

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.proxy_tensor.get_proxy_mode](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.proxy_tensor.get_proxy_mode.html)

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

### torch.fx.experimental.proxy_tensor.handle_sym_dispatch

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.proxy_tensor.handle_sym_dispatch](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.proxy_tensor.handle_sym_dispatch.html)

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

### torch.fx.experimental.proxy_tensor.make_fx

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.proxy_tensor.make_fx](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.proxy_tensor.make_fx.html)

**Support Status**:

<!-- npu="910b" id211 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id212 -->
<!-- npu="950" id213 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id213 -->

</div>

### torch.fx.experimental.proxy_tensor.maybe_disable_thunkify

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.proxy_tensor.maybe_disable_thunkify](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.proxy_tensor.maybe_disable_thunkify.html)

**Support Status**:

<!-- npu="910b" id214 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id215 -->
<!-- npu="950" id216 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id216 -->

</div>

### torch.fx.experimental.proxy_tensor.maybe_enable_thunkify

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.proxy_tensor.maybe_enable_thunkify](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.proxy_tensor.maybe_enable_thunkify.html)

**Support Status**:

<!-- npu="910b" id217 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id218 -->
<!-- npu="950" id219 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id219 -->

</div>
