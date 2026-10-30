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
<!-- npu="910b,A3" id368 -->
- [torch.fx.experimental.proxy_tensor](#torchfxexperimentalproxy_tensor)
<!-- end id368 -->

</div>

<div style="display:none;">

## &#8203;torch.fx.experimental

</div>

## torch.fx.experimental.symbolic_shapes

<!-- npu="910b,A3,950" id226 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.ShapeEnv
<!-- end id226 -->

<div style="margin-left: 2em">

<!-- npu="910b,A3" id227 -->
**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html)

**产品支持情况**：
<!-- end id227 -->

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->

<!-- npu="910b,A3" id228 -->
> <font size="3">format_guards()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.format_guards](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.format_guards)

**产品支持情况**：
<!-- end id228 -->

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->

<!-- npu="910b,A3" id229 -->
</div>
<!-- end id229 -->

<!-- npu="910b,A3" id230 -->
> <font size="3">freeze()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze)

**产品支持情况**：
<!-- end id230 -->

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->

<!-- npu="910b,A3" id231 -->
</div>
<!-- end id231 -->

<!-- npu="910b,A3" id232 -->
> <font size="3">freeze_runtime_asserts()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze_runtime_asserts](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.freeze_runtime_asserts)

**产品支持情况**：
<!-- end id232 -->

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->

<!-- npu="910b,A3" id233 -->
</div>
<!-- end id233 -->

<!-- npu="910b,A3" id234 -->
> <font size="3">get_axioms()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.get_axioms](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.get_axioms)

**产品支持情况**：
<!-- end id234 -->

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->

<!-- npu="910b,A3" id235 -->
</div>
<!-- end id235 -->

<!-- npu="910b,A3" id236 -->
> <font size="3">get_implications()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.get_implications](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.get_implications)

**产品支持情况**：
<!-- end id236 -->

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->

<!-- npu="910b,A3" id237 -->
</div>
<!-- end id237 -->

<!-- npu="910b,A3" id238 -->
> <font size="3">create_symbolic_sizes_strides_storage_offset()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbolic_sizes_strides_storage_offset](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbolic_sizes_strides_storage_offset)

**产品支持情况**：
<!-- end id238 -->

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->

<!-- npu="910b,A3" id239 -->
</div>
<!-- end id239 -->

<!-- npu="910b,A3" id240 -->
> <font size="3">create_symboolnode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symboolnode](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symboolnode)

**产品支持情况**：
<!-- end id240 -->

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->

<!-- npu="910b,A3" id241 -->
</div>
<!-- end id241 -->

<!-- npu="910b,A3" id242 -->
> <font size="3">create_symfloatnode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symfloatnode](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symfloatnode)

**产品支持情况**：
<!-- end id242 -->

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->

<!-- npu="910b,A3" id243 -->
</div>
<!-- end id243 -->

<!-- npu="910b,A3" id244 -->
> <font size="3">create_symintnode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symintnode](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symintnode)

**产品支持情况**：
<!-- end id244 -->

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->

<!-- npu="910b,A3" id245 -->
</div>
<!-- end id245 -->

<!-- npu="910b,A3" id246 -->
> <font size="3">create_unbacked_symbool()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unbacked_symbool](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unbacked_symbool)

**产品支持情况**：
<!-- end id246 -->

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->

<!-- npu="910b,A3" id247 -->
</div>
<!-- end id247 -->

<!-- npu="910b,A3" id248 -->
> <font size="3">deserialize_symexpr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.deserialize_symexpr](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.deserialize_symexpr)

**产品支持情况**：
<!-- end id248 -->

<!-- npu="910b" id34 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id35 -->

<!-- npu="910b,A3" id249 -->
</div>
<!-- end id249 -->

<!-- npu="910b,A3" id250 -->
> <font size="3">evaluate_guards_expression()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_expression](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_expression)

**产品支持情况**：
<!-- end id250 -->

<!-- npu="910b" id37 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id38 -->

<!-- npu="910b,A3" id251 -->
</div>
<!-- end id251 -->

<!-- npu="910b,A3" id252 -->
> <font size="3">evaluate_guards_for_args()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_for_args](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_guards_for_args)

**产品支持情况**：
<!-- end id252 -->

<!-- npu="910b" id40 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id41 -->

<!-- npu="910b,A3" id253 -->
</div>
<!-- end id253 -->

<!-- npu="910b,A3" id254 -->
> <font size="3">evaluate_sym_node()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_sym_node](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_sym_node)

**产品支持情况**：
<!-- end id254 -->

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->

<!-- npu="910b,A3" id255 -->
</div>
<!-- end id255 -->

<!-- npu="910b,A3" id256 -->
> <font size="3">evaluate_symexpr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_symexpr](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_symexpr)

**产品支持情况**：
<!-- end id256 -->

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id47 -->

<!-- npu="910b,A3" id257 -->
</div>
<!-- end id257 -->

<!-- npu="910b,A3" id258 -->
> <font size="3">size_hint()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.size_hint](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.size_hint)

**产品支持情况**：
<!-- end id258 -->

<!-- npu="910b" id49 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id50 -->

<!-- npu="910b,A3" id259 -->
</div>
<!-- end id259 -->

<!-- npu="910b,A3" id260 -->
> <font size="3">suppress_guards()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.suppress_guards](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.suppress_guards)

**产品支持情况**：
<!-- end id260 -->

<!-- npu="910b" id52 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id53 -->

<!-- npu="910b,A3" id261 -->
</div>
<!-- end id261 -->

<!-- npu="910b,A3" id262 -->
> <font size="3">produce_guards_expression()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_expression](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.produce_guards_expression)

**产品支持情况**：
<!-- end id262 -->

<!-- npu="910b" id55 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id56 -->

<!-- npu="910b,A3" id263 -->
**限制与说明**： `input`仅支持fp32

</div>
<!-- end id263 -->

<!-- npu="910b,A3" id264 -->
> <font size="3">replace()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.replace](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.replace)

**产品支持情况**：
<!-- end id264 -->

<!-- npu="910b" id61 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id62 -->

<!-- npu="910b,A3" id265 -->
</div>
<!-- end id265 -->

<!-- npu="910b,A3" id266 -->
> <font size="3">set_unbacked_var_to_val()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.set_unbacked_var_to_val](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.set_unbacked_var_to_val)

**产品支持情况**：
<!-- end id266 -->

<!-- npu="910b" id64 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id65 -->

<!-- npu="910b,A3" id267 -->
</div>
<!-- end id267 -->

<!-- npu="910b,A3" id268 -->
> <font size="3">simplify()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.simplify](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.simplify)

**产品支持情况**：
<!-- end id268 -->

<!-- npu="910b" id67 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id68 -->

<!-- npu="910b,A3" id269 -->
</div>
<!-- end id269 -->

<!-- npu="910b,A3" id270 -->
> <font size="3">create_symbol()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbol](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symbol)

**产品支持情况**：
<!-- end id270 -->

<!-- npu="910b" id70 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id71 -->

<!-- npu="910b,A3" id271 -->
</div>
<!-- end id271 -->

<!-- npu="910b,A3" id272 -->
> <font size="3">bound_sympy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.bound_sympy](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.bound_sympy)

**产品支持情况**：
<!-- end id272 -->

<!-- npu="910b" id73 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id74 -->

<!-- npu="910b,A3" id273 -->
</div>
<!-- end id273 -->

<!-- npu="910b,A3" id274 -->
> <font size="3">check_equal()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.check_equal](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.check_equal)

**产品支持情况**：
<!-- end id274 -->

<!-- npu="910b" id76 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id77 -->

<!-- npu="910b,A3" id275 -->
</div>
<!-- end id275 -->

<!-- npu="910b,A3" id276 -->
> <font size="3">cleanup()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.cleanup](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.cleanup)

**产品支持情况**：
<!-- end id276 -->

<!-- npu="910b" id79 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id80 -->

<!-- npu="910b,A3" id277 -->
</div>
<!-- end id277 -->

> <font size="3">bind_symbols()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnv.bind_symbols](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.bind_symbols)

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id84 -->

</div>

</div>

<!-- npu="910b,A3" id278 -->
### torch.fx.experimental.symbolic_shapes.lru_cache

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.lru_cache](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.lru_cache.html)

**产品支持情况**：
<!-- end id278 -->

<!-- npu="910b" id85 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id86 -->

<!-- npu="910b,A3" id279 -->
</div>
<!-- end id279 -->

<!-- npu="910b,A3" id280 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.ShapeEnvSettings

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ShapeEnvSettings](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ShapeEnvSettings.html)

**产品支持情况**：
<!-- end id280 -->

<!-- npu="910b" id88 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id89 -->

<!-- npu="910b,A3" id281 -->
</div>
<!-- end id281 -->

<!-- npu="910b,A3" id282 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext.html)

**产品支持情况**：
<!-- end id282 -->

<!-- npu="910b" id91 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id92 -->

<!-- npu="910b,A3" id283 -->
</div>
<!-- end id283 -->

<!-- npu="910b,A3" id284 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext.html)

**产品支持情况**：
<!-- end id284 -->

<!-- npu="910b" id94 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id95 -->

<!-- npu="910b,A3" id285 -->
</div>
<!-- end id285 -->

<!-- npu="910b,A3" id286 -->
### torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr.html)

**产品支持情况**：
<!-- end id286 -->

<!-- npu="910b" id97 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id98 -->

<!-- npu="910b,A3" id287 -->
</div>
<!-- end id287 -->

<!-- npu="910b,A3" id288 -->
### torch.fx.experimental.symbolic_shapes.check_consistent

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.check_consistent](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.check_consistent.html)

**产品支持情况**：
<!-- end id288 -->

<!-- npu="910b" id100 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id101 -->

<!-- npu="910b,A3" id289 -->
</div>
<!-- end id289 -->

<!-- npu="910b,A3" id290 -->
### torch.fx.experimental.symbolic_shapes.is_accessor_node

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.is_accessor_node](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.is_accessor_node.html)

**产品支持情况**：
<!-- end id290 -->

<!-- npu="910b" id103 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id104 -->

<!-- npu="910b,A3" id291 -->
</div>
<!-- end id291 -->

<!-- npu="910b,A3" id292 -->
### torch.fx.experimental.symbolic_shapes.is_concrete_bool

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.is_concrete_bool](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.is_concrete_bool.html)

**产品支持情况**：
<!-- end id292 -->

<!-- npu="910b" id106 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id107 -->

<!-- npu="910b,A3" id293 -->
</div>
<!-- end id293 -->

<!-- npu="910b,A3" id294 -->
### torch.fx.experimental.symbolic_shapes.is_concrete_float

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.is_concrete_float](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.is_concrete_float.html)

**产品支持情况**：
<!-- end id294 -->

<!-- npu="910b" id109 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id110 -->

<!-- npu="910b,A3" id295 -->
</div>
<!-- end id295 -->

<!-- npu="910b,A3" id296 -->
### torch.fx.experimental.symbolic_shapes.is_concrete_int

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.is_concrete_int](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.is_concrete_int.html)

**产品支持情况**：
<!-- end id296 -->

<!-- npu="910b" id112 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id113 -->

<!-- npu="910b,A3" id297 -->
</div>
<!-- end id297 -->

<!-- npu="910b,A3" id370 -->
<!-- npu="910b,A3" id298 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint
<!-- end id298 -->

<div style="margin-left: 2em">

<!-- npu="910b,A3" id299 -->
**原生文档**：[torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.html)

**产品支持情况**：
<!-- end id299 -->

<!-- npu="910b" id115 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id116 -->

<!-- npu="910b,A3" id300 -->
> <font size="3">render()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.render](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.html#torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.render)

**产品支持情况**：
<!-- end id300 -->

<!-- npu="910b" id118 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id119 -->

<!-- npu="910b,A3" id301 -->
</div>
<!-- end id301 -->

</div>
<!-- end id370 -->

<!-- npu="910b,A3" id302 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext.html)

**产品支持情况**：
<!-- end id302 -->

<!-- npu="910b" id121 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id122 -->

<!-- npu="910b,A3" id303 -->
</div>
<!-- end id303 -->

<!-- npu="910b,A3" id304 -->
### torch.fx.experimental.symbolic_shapes.sym_eq

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.sym_eq](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.sym_eq.html)

**产品支持情况**：
<!-- end id304 -->

<!-- npu="910b" id124 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id125 -->

<!-- npu="910b,A3" id305 -->
</div>
<!-- end id305 -->

<!-- npu="910b,A3" id306 -->
### torch.fx.experimental.symbolic_shapes.statically_known_true

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.statically_known_true](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.statically_known_true.html)

**产品支持情况**：
<!-- end id306 -->

<!-- npu="910b" id127 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id128 -->

<!-- npu="910b,A3" id307 -->
</div>
<!-- end id307 -->

<!-- npu="910b,A3" id371 -->
<!-- npu="910b,A3" id308 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.DivideByKey
<!-- end id308 -->

<div style="margin-left: 2em">

<!-- npu="910b,A3" id309 -->
**原生文档**：[torch.fx.experimental.symbolic_shapes.DivideByKey](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DivideByKey.html)

**产品支持情况**：
<!-- end id309 -->

<!-- npu="910b" id130 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id131 -->

<!-- npu="910b,A3" id310 -->
> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DivideByKey.get](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DivideByKey.html#torch.fx.experimental.symbolic_shapes.DivideByKey.get)

**产品支持情况**：
<!-- end id310 -->

<!-- npu="910b" id133 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id134 -->

<!-- npu="910b,A3" id311 -->
</div>
<!-- end id311 -->

</div>
<!-- end id371 -->

<!-- npu="910b,A3" id312 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.EqualityConstraint

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.EqualityConstraint](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.EqualityConstraint.html)

**产品支持情况**：
<!-- end id312 -->

<!-- npu="910b" id136 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id137 -->

<!-- npu="910b,A3" id313 -->
</div>
<!-- end id313 -->

<!-- npu="910b,A3" id314 -->
### torch.fx.experimental.symbolic_shapes.guard_size_oblivious

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.guard_size_oblivious](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.guard_size_oblivious.html)

**产品支持情况**：
<!-- end id314 -->

<!-- npu="910b" id139 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id140 -->

<!-- npu="910b,A3" id315 -->
</div>
<!-- end id315 -->

<!-- npu="910b,A3" id316 -->
### torch.fx.experimental.symbolic_shapes.has_free_symbols

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.has_free_symbols](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.has_free_symbols.html)

**产品支持情况**：
<!-- end id316 -->

<!-- npu="910b" id142 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id143 -->

<!-- npu="910b,A3" id317 -->
</div>
<!-- end id317 -->

### torch.fx.experimental.symbolic_shapes.has_free_unbacked_symbols

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.has_free_unbacked_symbols](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.has_free_unbacked_symbols.html)

**产品支持情况**：

<!-- npu="910b" id220 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id220 -->
<!-- npu="A3" id221 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id221 -->
<!-- npu="950" id222 -->
- <term>Ascend 950DT</term>：支持
<!-- end id222 -->

</div>

### torch.fx.experimental.symbolic_shapes.hint_int

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.hint_int](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.hint_int.html)

**产品支持情况**：

<!-- npu="910b" id223 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id223 -->
<!-- npu="A3" id224 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id224 -->
<!-- npu="950" id225 -->
- <term>Ascend 950DT</term>：支持
<!-- end id225 -->

</div>

<!-- npu="910b,A3" id372 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts

<div style="margin-left: 2em">

<!-- npu="910b,A3" id318 -->
> <font size="3">boxed_run()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.boxed_run](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.boxed_run)

**产品支持情况**：
<!-- end id318 -->

<!-- npu="910b" id145 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id146 -->

<!-- npu="910b,A3" id319 -->
</div>
<!-- end id319 -->

<!-- npu="910b,A3" id320 -->
> <font size="3">call_function()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_function](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_function)

**产品支持情况**：
<!-- end id320 -->

<!-- npu="910b" id148 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id149 -->

<!-- npu="910b,A3" id321 -->
</div>
<!-- end id321 -->

<!-- npu="910b,A3" id322 -->
> <font size="3">call_method()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_method](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_method)

**产品支持情况**：
<!-- end id322 -->

<!-- npu="910b" id151 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id152 -->

<!-- npu="910b,A3" id323 -->
</div>
<!-- end id323 -->

<!-- npu="910b,A3" id324 -->
> <font size="3">call_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_module](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.call_module)

**产品支持情况**：
<!-- end id324 -->

<!-- npu="910b" id154 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id155 -->

<!-- npu="910b,A3" id325 -->
</div>
<!-- end id325 -->

<!-- npu="910b,A3" id326 -->
> <font size="3">fetch_args_kwargs_from_env()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_args_kwargs_from_env](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_args_kwargs_from_env)

**产品支持情况**：
<!-- end id326 -->

<!-- npu="910b" id157 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id158 -->

<!-- npu="910b,A3" id327 -->
</div>
<!-- end id327 -->

<!-- npu="910b,A3" id328 -->
> <font size="3">fetch_attr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_attr](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.fetch_attr)

**产品支持情况**：
<!-- end id328 -->

<!-- npu="910b" id160 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id161 -->

<!-- npu="910b,A3" id329 -->
</div>
<!-- end id329 -->

<!-- npu="910b,A3" id330 -->
> <font size="3">get_attr()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.get_attr](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.get_attr)

**产品支持情况**：
<!-- end id330 -->

<!-- npu="910b" id163 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id164 -->

<!-- npu="910b,A3" id331 -->
</div>
<!-- end id331 -->

<!-- npu="910b,A3" id332 -->
> <font size="3">map_nodes_to_values()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.map_nodes_to_values](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.html#torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.map_nodes_to_values)

**产品支持情况**：
<!-- end id332 -->

<!-- npu="910b" id166 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id167 -->

<!-- npu="910b,A3" id333 -->
</div>
<!-- end id333 -->

</div>
<!-- end id372 -->

<!-- npu="910b,A3" id334 -->
### torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings.html)

**产品支持情况**：
<!-- end id334 -->

<!-- npu="910b" id169 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id170 -->

<!-- npu="910b,A3" id335 -->
</div>
<!-- end id335 -->

<!-- npu="910b,A3" id336 -->
### torch.fx.experimental.symbolic_shapes.constrain_range

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.constrain_range](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.constrain_range.html)

**产品支持情况**：
<!-- end id336 -->

<!-- npu="910b" id172 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id173 -->

<!-- npu="910b,A3" id337 -->
**限制与说明**： 需通过`torch.compile`获取SymInt

</div>
<!-- end id337 -->

<!-- npu="910b,A3" id338 -->
### torch.fx.experimental.symbolic_shapes.constrain_unify

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.constrain_unify](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.constrain_unify.html)

**产品支持情况**：
<!-- end id338 -->

<!-- npu="910b" id175 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id176 -->

<!-- npu="910b,A3" id339 -->
**限制与说明**： 需通过`torch.compile`获取SymInt

</div>
<!-- end id339 -->

<!-- npu="910b,A3" id373 -->
<!-- npu="910b,A3" id340 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.ConvertIntKey
<!-- end id340 -->

<div style="margin-left: 2em">

<!-- npu="910b,A3" id341 -->
**原生文档**：[torch.fx.experimental.symbolic_shapes.ConvertIntKey](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ConvertIntKey.html)

**产品支持情况**：
<!-- end id341 -->

<!-- npu="910b" id178 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id179 -->

<!-- npu="910b,A3" id342 -->
> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.ConvertIntKey.get](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.ConvertIntKey.html#torch.fx.experimental.symbolic_shapes.ConvertIntKey.get)

**产品支持情况**：
<!-- end id342 -->

<!-- npu="910b" id181 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id182 -->

<!-- npu="910b,A3" id343 -->
</div>
<!-- end id343 -->

</div>
<!-- end id373 -->

<!-- npu="910b,A3" id374 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.DimConstraints

<div style="margin-left: 2em">

<!-- npu="910b,A3" id344 -->
> <font size="3">forced_specializations()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimConstraints.forced_specializations](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.forced_specializations)

**产品支持情况**：
<!-- end id344 -->

<!-- npu="910b" id184 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id185 -->

<!-- npu="910b,A3" id345 -->
</div>
<!-- end id345 -->

<!-- npu="910b,A3" id346 -->
> <font size="3">prettify_results()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimConstraints.prettify_results](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.prettify_results)

**产品支持情况**：
<!-- end id346 -->

<!-- npu="910b" id187 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id188 -->

<!-- npu="910b,A3" id347 -->
</div>
<!-- end id347 -->

<!-- npu="910b,A3" id348 -->
> <font size="3">rewrite_with_congruences()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimConstraints.rewrite_with_congruences](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.rewrite_with_congruences)

**产品支持情况**：
<!-- end id348 -->

<!-- npu="910b" id190 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id191 -->

<!-- npu="910b,A3" id349 -->
</div>
<!-- end id349 -->

<!-- npu="910b,A3" id350 -->
> <font size="3">solve()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimConstraints.solve](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints.solve)

**产品支持情况**：
<!-- end id350 -->

<!-- npu="910b" id193 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id194 -->

<!-- npu="910b,A3" id351 -->
</div>
<!-- end id351 -->

</div>
<!-- end id374 -->

<!-- npu="910b,A3" id352 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.DimDynamic

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.DimDynamic](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.DimDynamic.html)

**产品支持情况**：
<!-- end id352 -->

<!-- npu="910b" id196 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id197 -->

<!-- npu="910b,A3" id353 -->
</div>
<!-- end id353 -->

<!-- npu="910b,A3" id375 -->
<!-- npu="910b,A3" id354 -->
### <code><i>class</i></code> torch.fx.experimental.symbolic_shapes.CallMethodKey
<!-- end id354 -->

<div style="margin-left: 2em">

<!-- npu="910b,A3" id355 -->
**原生文档**：[torch.fx.experimental.symbolic_shapes.CallMethodKey](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.CallMethodKey.html)

**产品支持情况**：
<!-- end id355 -->

<!-- npu="910b" id199 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id200 -->

<!-- npu="910b,A3" id356 -->
> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.symbolic_shapes.CallMethodKey.get](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.symbolic_shapes.CallMethodKey.html#torch.fx.experimental.symbolic_shapes.CallMethodKey.get)

**产品支持情况**：
<!-- end id356 -->

<!-- npu="910b" id202 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id203 -->

<!-- npu="910b,A3" id357 -->
</div>
<!-- end id357 -->

</div>
<!-- end id375 -->

<!-- npu="910b,A3" id369 -->
## torch.fx.experimental.proxy_tensor

<!-- npu="910b,A3" id358 -->
### torch.fx.experimental.proxy_tensor.get_proxy_mode

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.get_proxy_mode](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.proxy_tensor.get_proxy_mode.html)

**产品支持情况**：
<!-- end id358 -->

<!-- npu="910b" id205 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id206 -->

<!-- npu="910b,A3" id359 -->
</div>
<!-- end id359 -->

<!-- npu="910b,A3" id360 -->
### torch.fx.experimental.proxy_tensor.handle_sym_dispatch

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.handle_sym_dispatch](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.proxy_tensor.handle_sym_dispatch.html)

**产品支持情况**：
<!-- end id360 -->

<!-- npu="910b" id208 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id209 -->

<!-- npu="910b,A3" id361 -->
</div>
<!-- end id361 -->

<!-- npu="910b,A3" id362 -->
### torch.fx.experimental.proxy_tensor.make_fx

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.make_fx](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.proxy_tensor.make_fx.html)

**产品支持情况**：
<!-- end id362 -->

<!-- npu="910b" id211 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id212 -->

<!-- npu="910b,A3" id363 -->
</div>
<!-- end id363 -->

<!-- npu="910b,A3" id364 -->
### torch.fx.experimental.proxy_tensor.maybe_disable_thunkify

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.maybe_disable_thunkify](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.proxy_tensor.maybe_disable_thunkify.html)

**产品支持情况**：
<!-- end id364 -->

<!-- npu="910b" id214 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id215 -->

<!-- npu="910b,A3" id365 -->
</div>
<!-- end id365 -->

<!-- npu="910b,A3" id366 -->
### torch.fx.experimental.proxy_tensor.maybe_enable_thunkify

<div style="margin-left: 2em">

**原生文档**：[torch.fx.experimental.proxy_tensor.maybe_enable_thunkify](https://pytorch.org/docs/2.7/generated/torch.fx.experimental.proxy_tensor.maybe_enable_thunkify.html)

**产品支持情况**：
<!-- end id366 -->

<!-- npu="910b" id217 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id218 -->

<!-- npu="910b,A3" id367 -->
</div>
<!-- end id367 -->
<!-- end id369 -->
