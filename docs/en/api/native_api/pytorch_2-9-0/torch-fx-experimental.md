# torch.fx.experimental

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - The scope of this support list is based on the [PyTorch Community Documentation](https://pytorch.org/docs/2.9/). APIs not listed in the community documentation are outside the scope of this list; APIs that exist in the community documentation but are not included in this list have not been verified yet. Use them with caution. We will update the document upon completion of verification.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.9/fx.experimental.html).
> - Native community note: The APIs under the torch.fx.experimental module are experimental and subject to change at any time. Refer to the native documentation and use them with caution.

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [torch.fx.experimental.symbolic_shapes](#torchfxexperimentalsymbolic_shapes)
<!-- npu="910b,A3" id361 -->
- [torch.fx.experimental.proxy_tensor](#torchfxexperimentalproxy_tensor)
<!-- end id361 -->

</div>

<div style="display:none;">

## &#8203;torch.fx.experimental

</div>

## torch.fx.experimental.symbolic_shapes

<!-- npu="910b,A3,950" id219 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.ShapeEnv
<!-- end id219 -->

<div style="margin-left: 2em">

<!-- npu="910b,A3" id220 -->
**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html)

**Support Status**:
<!-- end id220 -->

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->

<!-- npu="910b,A3" id221 -->
> <font size="3">format_guards()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.format_guards](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.format_guards)

**Support Status**:
<!-- end id221 -->

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->

<!-- npu="910b,A3" id222 -->
</div>
<!-- end id222 -->

<!-- npu="910b,A3" id223 -->
> <font size="3">freeze()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze)

**Support Status**:
<!-- end id223 -->

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->

<!-- npu="910b,A3" id224 -->
</div>
<!-- end id224 -->

<!-- npu="910b,A3" id225 -->
> <font size="3">freeze_runtime_asserts()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze_runtime_asserts](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze_runtime_asserts)

**Support Status**:
<!-- end id225 -->

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->

<!-- npu="910b,A3" id226 -->
</div>
<!-- end id226 -->

<!-- npu="910b,A3" id227 -->
> <font size="3">get_axioms()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.get_axioms](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.get_axioms)

**Support Status**:
<!-- end id227 -->

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->

<!-- npu="910b,A3" id228 -->
</div>
<!-- end id228 -->

<!-- npu="910b,A3" id229 -->
> <font size="3">get_implications()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.get_implications](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.get_implications)

**Support Status**:
<!-- end id229 -->

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->

<!-- npu="910b,A3" id230 -->
</div>
<!-- end id230 -->

<!-- npu="910b,A3" id231 -->
> <font size="3">create_symbolic_sizes_strides_storage_offset()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbolic_sizes_strides_storage_offset](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbolic_sizes_strides_storage_offset)

**Support Status**:
<!-- end id231 -->

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->

<!-- npu="910b,A3" id232 -->
</div>
<!-- end id232 -->

<!-- npu="910b,A3" id233 -->
> <font size="3">create_symboolnode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symboolnode](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symboolnode)

**Support Status**:
<!-- end id233 -->

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->

<!-- npu="910b,A3" id234 -->
</div>
<!-- end id234 -->

<!-- npu="910b,A3" id235 -->
> <font size="3">create_symfloatnode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symfloatnode](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symfloatnode)

**Support Status**:
<!-- end id235 -->

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->

<!-- npu="910b,A3" id236 -->
</div>
<!-- end id236 -->

<!-- npu="910b,A3" id237 -->
> <font size="3">create_symintnode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symintnode](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symintnode)

**Support Status**:
<!-- end id237 -->

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->

<!-- npu="910b,A3" id238 -->
</div>
<!-- end id238 -->

<!-- npu="910b,A3" id239 -->
> <font size="3">create_unbacked_symbool()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unbacked_symbool](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unbacked_symbool)

**Support Status**:
<!-- end id239 -->

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->

<!-- npu="910b,A3" id240 -->
</div>
<!-- end id240 -->

<!-- npu="910b,A3" id241 -->
> <font size="3">deserialize_symexpr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.deserialize_symexpr](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.deserialize_symexpr)

**Support Status**:
<!-- end id241 -->

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->

<!-- npu="910b,A3" id242 -->
</div>
<!-- end id242 -->

<!-- npu="910b,A3" id243 -->
> <font size="3">evaluate_guards_expression()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_expression](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_expression)

**Support Status**:
<!-- end id243 -->

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->

<!-- npu="910b,A3" id244 -->
</div>
<!-- end id244 -->

<!-- npu="910b,A3" id245 -->
> <font size="3">evaluate_guards_for_args()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_for_args](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_for_args)

**Support Status**:
<!-- end id245 -->

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->

<!-- npu="910b,A3" id246 -->
</div>
<!-- end id246 -->

<!-- npu="910b,A3" id247 -->
> <font size="3">evaluate_sym_node()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_sym_node](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_sym_node)

**Support Status**:
<!-- end id247 -->

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->

<!-- npu="910b,A3" id248 -->
</div>
<!-- end id248 -->

<!-- npu="910b,A3" id249 -->
> <font size="3">evaluate_symexpr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_symexpr](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_symexpr)

**Support Status**:
<!-- end id249 -->

<!-- npu="910b" id46 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id47 -->

<!-- npu="910b,A3" id250 -->
</div>
<!-- end id250 -->

<!-- npu="910b,A3" id251 -->
> <font size="3">size_hint()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.size_hint](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.size_hint)

**Support Status**:
<!-- end id251 -->

<!-- npu="910b" id49 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id50 -->

<!-- npu="910b,A3" id252 -->
</div>
<!-- end id252 -->

<!-- npu="910b,A3" id253 -->
> <font size="3">suppress_guards()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.suppress_guards](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.suppress_guards)

**Support Status**:
<!-- end id253 -->

<!-- npu="910b" id52 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id53 -->

<!-- npu="910b,A3" id254 -->
</div>
<!-- end id254 -->

<!-- npu="910b,A3" id255 -->
> <font size="3">produce_guards_expression()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_expression](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_expression)

**Support Status**:
<!-- end id255 -->

<!-- npu="910b" id55 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id56 -->

<!-- npu="910b,A3" id256 -->
**Constraints & Notes**: `input` only supports fp32

</div>
<!-- end id256 -->

<!-- npu="910b,A3" id257 -->
> <font size="3">replace()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.replace](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.replace)

**Support Status**:
<!-- end id257 -->

<!-- npu="910b" id61 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id62 -->

<!-- npu="910b,A3" id258 -->
</div>
<!-- end id258 -->

<!-- npu="910b,A3" id259 -->
> <font size="3">set_unbacked_var_to_val()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.set_unbacked_var_to_val](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.set_unbacked_var_to_val)

**Support Status**:
<!-- end id259 -->

<!-- npu="910b" id64 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id65 -->

<!-- npu="910b,A3" id260 -->
</div>
<!-- end id260 -->

<!-- npu="910b,A3" id261 -->
> <font size="3">simplify()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.simplify](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.simplify)

**Support Status**:
<!-- end id261 -->

<!-- npu="910b" id67 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id68 -->

<!-- npu="910b,A3" id262 -->
</div>
<!-- end id262 -->

<!-- npu="910b,A3" id263 -->
> <font size="3">create_symbol()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbol](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbol)

**Support Status**:
<!-- end id263 -->

<!-- npu="910b" id70 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id71 -->

<!-- npu="910b,A3" id264 -->
</div>
<!-- end id264 -->

<!-- npu="910b,A3" id265 -->
> <font size="3">bound_sympy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.bound_sympy](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.bound_sympy)

**Support Status**:
<!-- end id265 -->

<!-- npu="910b" id73 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id74 -->

<!-- npu="910b,A3" id266 -->
</div>
<!-- end id266 -->

<!-- npu="910b,A3" id267 -->
> <font size="3">check_equal()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.check_equal](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.check_equal)

**Support Status**:
<!-- end id267 -->

<!-- npu="910b" id76 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id77 -->

<!-- npu="910b,A3" id268 -->
</div>
<!-- end id268 -->

<!-- npu="910b,A3" id269 -->
> <font size="3">cleanup()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.cleanup](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.cleanup)

**Support Status**:
<!-- end id269 -->

<!-- npu="910b" id79 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id80 -->

<!-- npu="910b,A3" id270 -->
</div>
<!-- end id270 -->

> <font size="3">bind_symbols()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnv.bind_symbols](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.bind_symbols)

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

<!-- npu="910b,A3" id345 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.DimDynamic

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.DimDynamic](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.DimDynamic.html)

**Support Status**:
<!-- end id345 -->

<!-- npu="910b" id196 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id197 -->

<!-- npu="910b,A3" id346 -->
</div>
<!-- end id346 -->

<!-- npu="910b,A3" id305 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.EqualityConstraint

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.EqualityConstraint](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.EqualityConstraint.html)

**Support Status**:
<!-- end id305 -->

<!-- npu="910b" id136 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id137 -->

<!-- npu="910b,A3" id306 -->
</div>
<!-- end id306 -->

<!-- npu="910b,A3" id277 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext.html)

**Support Status**:
<!-- end id277 -->

<!-- npu="910b" id94 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id95 -->

<!-- npu="910b,A3" id278 -->
</div>
<!-- end id278 -->

<!-- npu="910b,A3" id275 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext.html)

**Support Status**:
<!-- end id275 -->

<!-- npu="910b" id91 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id92 -->

<!-- npu="910b,A3" id276 -->
</div>
<!-- end id276 -->

<!-- npu="910b,A3" id295 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext.html)

**Support Status**:
<!-- end id295 -->

<!-- npu="910b" id121 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id122 -->

<!-- npu="910b,A3" id296 -->
</div>
<!-- end id296 -->

<!-- npu="910b,A3" id367 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.DimConstraints

<div style="margin-left: 2em">

<!-- npu="910b,A3" id337 -->
> <font size="3">forced_specializations()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.DimConstraints.forced_specializations](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.forced_specializations)

**Support Status**:
<!-- end id337 -->

<!-- npu="910b" id184 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id185 -->

<!-- npu="910b,A3" id338 -->
</div>
<!-- end id338 -->

<!-- npu="910b,A3" id339 -->
> <font size="3">prettify_results()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.DimConstraints.prettify_results](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.prettify_results)

**Support Status**:
<!-- end id339 -->

<!-- npu="910b" id187 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id188 -->

<!-- npu="910b,A3" id340 -->
</div>
<!-- end id340 -->

<!-- npu="910b,A3" id341 -->
> <font size="3">rewrite_with_congruences()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.DimConstraints.rewrite_with_congruences](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.rewrite_with_congruences)

**Support Status**:
<!-- end id341 -->

<!-- npu="910b" id190 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id191 -->

<!-- npu="910b,A3" id342 -->
</div>
<!-- end id342 -->

<!-- npu="910b,A3" id343 -->
> <font size="3">solve()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.DimConstraints.solve](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.solve)

**Support Status**:
<!-- end id343 -->

<!-- npu="910b" id193 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id194 -->

<!-- npu="910b,A3" id344 -->
</div>
<!-- end id344 -->

</div>
<!-- end id367 -->

<!-- npu="910b,A3" id273 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.ShapeEnvSettings

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ShapeEnvSettings](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ShapeEnvSettings.html)

**Support Status**:
<!-- end id273 -->

<!-- npu="910b" id88 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id89 -->

<!-- npu="910b,A3" id274 -->
</div>
<!-- end id274 -->

<!-- npu="910b,A3" id365 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts

<div style="margin-left: 2em">

<!-- npu="910b,A3" id311 -->
> <font size="3">boxed_run()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.boxed_run](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.boxed_run)

**Support Status**:
<!-- end id311 -->

<!-- npu="910b" id145 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id146 -->

<!-- npu="910b,A3" id312 -->
</div>
<!-- end id312 -->

<!-- npu="910b,A3" id313 -->
> <font size="3">call_function()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_function](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_function)

**Support Status**:
<!-- end id313 -->

<!-- npu="910b" id148 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id149 -->

<!-- npu="910b,A3" id314 -->
</div>
<!-- end id314 -->

<!-- npu="910b,A3" id315 -->
> <font size="3">call_method()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_method](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_method)

**Support Status**:
<!-- end id315 -->

<!-- npu="910b" id151 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id152 -->

<!-- npu="910b,A3" id316 -->
</div>
<!-- end id316 -->

<!-- npu="910b,A3" id317 -->
> <font size="3">call_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_module](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_module)

**Support Status**:
<!-- end id317 -->

<!-- npu="910b" id154 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id155 -->

<!-- npu="910b,A3" id318 -->
</div>
<!-- end id318 -->

<!-- npu="910b,A3" id319 -->
> <font size="3">fetch_args_kwargs_from_env()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_args_kwargs_from_env](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_args_kwargs_from_env)

**Support Status**:
<!-- end id319 -->

<!-- npu="910b" id157 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id158 -->

<!-- npu="910b,A3" id320 -->
</div>
<!-- end id320 -->

<!-- npu="910b,A3" id321 -->
> <font size="3">fetch_attr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_attr](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_attr)

**Support Status**:
<!-- end id321 -->

<!-- npu="910b" id160 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id161 -->

<!-- npu="910b,A3" id322 -->
</div>
<!-- end id322 -->

<!-- npu="910b,A3" id323 -->
> <font size="3">get_attr()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.get_attr](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.get_attr)

**Support Status**:
<!-- end id323 -->

<!-- npu="910b" id163 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id164 -->

<!-- npu="910b,A3" id324 -->
</div>
<!-- end id324 -->

<!-- npu="910b,A3" id325 -->
> <font size="3">map_nodes_to_values()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.map_nodes_to_values](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.map_nodes_to_values)

**Support Status**:
<!-- end id325 -->

<!-- npu="910b" id166 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id167 -->

<!-- npu="910b,A3" id326 -->
</div>
<!-- end id326 -->

</div>
<!-- end id365 -->

<!-- npu="910b,A3" id289 -->
### torch.fx.experimental.symbolic_shapes.is_concrete_int

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.is_concrete_int](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.is_concrete_int.html)

**Support Status**:
<!-- end id289 -->

<!-- npu="910b" id112 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id113 -->

<!-- npu="910b,A3" id290 -->
</div>
<!-- end id290 -->

<!-- npu="910b,A3" id285 -->
### torch.fx.experimental.symbolic_shapes.is_concrete_bool

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.is_concrete_bool](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.is_concrete_bool.html)

**Support Status**:
<!-- end id285 -->

<!-- npu="910b" id106 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id107 -->

<!-- npu="910b,A3" id286 -->
</div>
<!-- end id286 -->

<!-- npu="910b,A3" id363 -->
<!-- npu="910b,A3" id291 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint
<!-- end id291 -->

<div style="margin-left: 2em">

<!-- npu="910b,A3" id292 -->
**Native Documentation**: [torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.html)

**Support Status**:
<!-- end id292 -->

<!-- npu="910b" id115 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id116 -->

<!-- npu="910b,A3" id293 -->
> <font size="3">render()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.render](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.html#torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.render)

**Support Status**:
<!-- end id293 -->

<!-- npu="910b" id118 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id119 -->

<!-- npu="910b,A3" id294 -->
</div>
<!-- end id294 -->

</div>
<!-- end id363 -->

<!-- npu="910b,A3" id287 -->
### torch.fx.experimental.symbolic_shapes.is_concrete_float

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.is_concrete_float](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.is_concrete_float.html)

**Support Status**:
<!-- end id287 -->

<!-- npu="910b" id109 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id110 -->

<!-- npu="910b,A3" id288 -->
</div>
<!-- end id288 -->

<!-- npu="910b,A3" id309 -->
### torch.fx.experimental.symbolic_shapes.has_free_symbols

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.has_free_symbols](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.has_free_symbols.html)

**Support Status**:
<!-- end id309 -->

<!-- npu="910b" id142 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id143 -->

<!-- npu="910b,A3" id310 -->
</div>
<!-- end id310 -->

<!-- npu="910b,A3" id307 -->
### torch.fx.experimental.symbolic_shapes.guard_size_oblivious

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.guard_size_oblivious](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.guard_size_oblivious.html)

**Support Status**:
<!-- end id307 -->

<!-- npu="910b" id139 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id140 -->

<!-- npu="910b,A3" id308 -->
</div>
<!-- end id308 -->

<!-- npu="910b,A3" id364 -->
<!-- npu="910b,A3" id301 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.DivideByKey
<!-- end id301 -->

<div style="margin-left: 2em">

<!-- npu="910b,A3" id302 -->
**Native Documentation**: [torch.fx.experimental.symbolic_shapes.DivideByKey](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.DivideByKey.html)

**Support Status**:
<!-- end id302 -->

<!-- npu="910b" id130 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id131 -->

<!-- npu="910b,A3" id303 -->
> <font size="3">get()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.DivideByKey.get](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.DivideByKey.html#torch.fx.experimental.symbolic_shapes.DivideByKey.get)

**Support Status**:
<!-- end id303 -->

<!-- npu="910b" id133 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id134 -->

<!-- npu="910b,A3" id304 -->
</div>
<!-- end id304 -->

</div>
<!-- end id364 -->

<!-- npu="910b,A3" id297 -->
### torch.fx.experimental.symbolic_shapes.sym_eq

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.sym_eq](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.sym_eq.html)

**Support Status**:
<!-- end id297 -->

<!-- npu="910b" id124 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id125 -->

<!-- npu="910b,A3" id298 -->
</div>
<!-- end id298 -->

<!-- npu="910b,A3" id329 -->
### torch.fx.experimental.symbolic_shapes.constrain_range

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.constrain_range](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.constrain_range.html)

**Support Status**:
<!-- end id329 -->

<!-- npu="910b" id172 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id173 -->

<!-- npu="910b,A3" id330 -->
**Constraints & Notes**: Requires obtaining SymInt through `torch.compile`.

</div>
<!-- end id330 -->

<!-- npu="910b,A3" id331 -->
### torch.fx.experimental.symbolic_shapes.constrain_unify

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.constrain_unify](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.constrain_unify.html)

**Support Status**:
<!-- end id331 -->

<!-- npu="910b" id175 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id176 -->

<!-- npu="910b,A3" id332 -->
**Constraints & Notes**: Requires obtaining SymInt through `torch.compile`.

</div>
<!-- end id332 -->

<!-- npu="910b,A3" id279 -->
### torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr.html)

**Support Status**:
<!-- end id279 -->

<!-- npu="910b" id97 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id98 -->

<!-- npu="910b,A3" id280 -->
</div>
<!-- end id280 -->

<!-- npu="910b,A3" id299 -->
### torch.fx.experimental.symbolic_shapes.statically_known_true

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.statically_known_true](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.statically_known_true.html)

**Support Status**:
<!-- end id299 -->

<!-- npu="910b" id127 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id128 -->

<!-- npu="910b,A3" id300 -->
</div>
<!-- end id300 -->

<!-- npu="910b,A3" id271 -->
### torch.fx.experimental.symbolic_shapes.lru_cache

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.lru_cache](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.lru_cache.html)

**Support Status**:
<!-- end id271 -->

<!-- npu="910b" id85 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id86 -->

<!-- npu="910b,A3" id272 -->
</div>
<!-- end id272 -->

<!-- npu="910b,A3" id281 -->
### torch.fx.experimental.symbolic_shapes.check_consistent

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.check_consistent](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.check_consistent.html)

**Support Status**:
<!-- end id281 -->

<!-- npu="910b" id100 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id101 -->

<!-- npu="910b,A3" id282 -->
</div>
<!-- end id282 -->

<!-- npu="910b,A3" id366 -->
<!-- npu="910b,A3" id333 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.ConvertIntKey
<!-- end id333 -->

<div style="margin-left: 2em">

<!-- npu="910b,A3" id334 -->
**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ConvertIntKey](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ConvertIntKey.html)

**Support Status**:
<!-- end id334 -->

<!-- npu="910b" id178 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id179 -->

<!-- npu="910b,A3" id335 -->
> <font size="3">get()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.ConvertIntKey.get](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.ConvertIntKey.html#torch.fx.experimental.symbolic_shapes.ConvertIntKey.get)

**Support Status**:
<!-- end id335 -->

<!-- npu="910b" id181 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id182 -->

<!-- npu="910b,A3" id336 -->
</div>
<!-- end id336 -->

</div>
<!-- end id366 -->

<!-- npu="910b,A3" id327 -->
### torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings.html)

**Support Status**:
<!-- end id327 -->

<!-- npu="910b" id169 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id170 -->

<!-- npu="910b,A3" id328 -->
</div>
<!-- end id328 -->

<!-- npu="910b,A3" id283 -->
### torch.fx.experimental.symbolic_shapes.is_accessor_node

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.is_accessor_node](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.is_accessor_node.html)

**Support Status**:
<!-- end id283 -->

<!-- npu="910b" id103 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id104 -->

<!-- npu="910b,A3" id284 -->
</div>
<!-- end id284 -->

<!-- npu="910b,A3" id368 -->
<!-- npu="910b,A3" id347 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.CallMethodKey
<!-- end id347 -->

<div style="margin-left: 2em">

<!-- npu="910b,A3" id348 -->
**Native Documentation**: [torch.fx.experimental.symbolic_shapes.CallMethodKey](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.CallMethodKey.html)

**Support Status**:
<!-- end id348 -->

<!-- npu="910b" id199 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id200 -->

<!-- npu="910b,A3" id349 -->
> <font size="3">get()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.symbolic_shapes.CallMethodKey.get](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.symbolic_shapes.CallMethodKey.html#torch.fx.experimental.symbolic_shapes.CallMethodKey.get)

**Support Status**:
<!-- end id349 -->

<!-- npu="910b" id202 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id203 -->

<!-- npu="910b,A3" id350 -->
</div>
<!-- end id350 -->

</div>
<!-- end id368 -->

<!-- npu="910b,A3" id362 -->
## torch.fx.experimental.proxy_tensor

<!-- npu="910b,A3" id355 -->
### torch.fx.experimental.proxy_tensor.make_fx

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.proxy_tensor.make_fx](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.proxy_tensor.make_fx.html)

**Support Status**:
<!-- end id355 -->

<!-- npu="910b" id211 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id212 -->

<!-- npu="910b,A3" id356 -->
</div>
<!-- end id356 -->

<!-- npu="910b,A3" id353 -->
### torch.fx.experimental.proxy_tensor.handle_sym_dispatch

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.proxy_tensor.handle_sym_dispatch](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.proxy_tensor.handle_sym_dispatch.html)

**Support Status**:
<!-- end id353 -->

<!-- npu="910b" id208 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id209 -->

<!-- npu="910b,A3" id354 -->
</div>
<!-- end id354 -->

<!-- npu="910b,A3" id351 -->
### torch.fx.experimental.proxy_tensor.get_proxy_mode

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.proxy_tensor.get_proxy_mode](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.proxy_tensor.get_proxy_mode.html)

**Support Status**:
<!-- end id351 -->

<!-- npu="910b" id205 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id206 -->

<!-- npu="910b,A3" id352 -->
</div>
<!-- end id352 -->

<!-- npu="910b,A3" id359 -->
### torch.fx.experimental.proxy_tensor.maybe_enable_thunkify

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.proxy_tensor.maybe_enable_thunkify](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.proxy_tensor.maybe_enable_thunkify.html)

**Support Status**:
<!-- end id359 -->

<!-- npu="910b" id217 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id218 -->

<!-- npu="910b,A3" id360 -->
</div>
<!-- end id360 -->

<!-- npu="910b,A3" id357 -->
### torch.fx.experimental.proxy_tensor.maybe_disable_thunkify

<div style="margin-left: 2em">

**Native Documentation**: [torch.fx.experimental.proxy_tensor.maybe_disable_thunkify](https://pytorch.org/docs/2.9/generated/torch.fx.experimental.proxy_tensor.maybe_disable_thunkify.html)

**Support Status**:
<!-- end id357 -->

<!-- npu="910b" id214 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id215 -->

<!-- npu="910b,A3" id358 -->
</div>
<!-- end id358 -->
<!-- end id362 -->