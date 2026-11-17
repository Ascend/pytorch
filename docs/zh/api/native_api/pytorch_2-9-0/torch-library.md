# torch.library

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 本支持清单的范围以[PyTorch社区文档](https://pytorch.org/docs/2.9/)为准，社区文档中未列出的接口不在本清单范围内；已在社区文档中存在但未收录于本清单的API尚未验证，请谨慎使用，我们将在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.9/library.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Testing custom ops](#testing-custom-ops)
- [Creating new custom ops in Python](#creating-new-custom-ops-in-python)
- [Extending custom ops (created from Python or C++)](#extending-custom-ops-created-from-python-or-c)
- [Low-level APIs](#low-level-apis)

</div>

<div style="display:none;">

## &#8203;torch.library

</div>

## Testing custom ops

### torch.library.opcheck

<div style="margin-left: 2em">

**原生文档**：[torch.library.opcheck](https://pytorch.org/docs/2.9/library.html#torch.library.opcheck)

**产品支持情况**：

<!-- npu="910b" id15 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id15 -->
<!-- npu="A3" id16 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id16 -->

</div>

## Creating new custom ops in Python

### torch.library.custom_op

<div style="margin-left: 2em">

**原生文档**：[torch.library.custom_op](https://pytorch.org/docs/2.9/library.html#torch.library.custom_op)

**产品支持情况**：

<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="A3" id4 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id4 -->

</div>

## Extending custom ops (created from Python or C++)

### torch.library.register_kernel

<div style="margin-left: 2em">

**原生文档**：[torch.library.register_kernel](https://pytorch.org/docs/2.9/library.html#torch.library.register_kernel)

**产品支持情况**：

<!-- npu="910b" id21 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id21 -->
<!-- npu="A3" id22 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id22 -->

</div>

### torch.library.register_autograd

<div style="margin-left: 2em">

**原生文档**：[torch.library.register_autograd](https://pytorch.org/docs/2.9/library.html#torch.library.register_autograd)

**产品支持情况**：

<!-- npu="910b" id17 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="A3" id18 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id18 -->

</div>

### torch.library.register_fake

<div style="margin-left: 2em">

**原生文档**：[torch.library.register_fake](https://pytorch.org/docs/2.9/library.html#torch.library.register_fake)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->

</div>

### torch.library.register_vmap

<div style="margin-left: 2em">

**原生文档**：[torch.library.register_vmap](https://pytorch.org/docs/2.9/library.html#torch.library.register_vmap)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->

</div>

### torch.library.get_ctx

<div style="margin-left: 2em">

**原生文档**：[torch.library.get_ctx](https://pytorch.org/docs/2.9/library.html#torch.library.get_ctx)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->

</div>

### torch.library.register_torch_dispatch

<div style="margin-left: 2em">

**原生文档**：[torch.library.register_torch_dispatch](https://pytorch.org/docs/2.9/library.html#torch.library.register_torch_dispatch)

**产品支持情况**：

<!-- npu="910b" id23 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="A3" id24 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id24 -->

</div>

### torch.library.infer_schema

<div style="margin-left: 2em">

**原生文档**：[torch.library.infer_schema](https://pytorch.org/docs/2.9/library.html#torch.library.infer_schema)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->

</div>

## Low-level APIs

### <code><i>class</i></code> torch.library.Library

<div style="margin-left: 2em">

**原生文档**：[torch.library.Library](https://pytorch.org/docs/2.9/library.html#torch.library.Library)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id35 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id35 -->

<a class="method_anchor" id="torch_library_Library_define"></a>

<!-- npu="910b,A3,950" id43 -->
> <font size="3">define()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.library.Library.define](https://pytorch.org/docs/2.9/library.html#torch.library.Library.define)

**产品支持情况**：
<!-- end id43 -->

<!-- npu="910b" id27 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id27 -->
<!-- npu="A3" id28 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id36 -->

<!-- npu="910b,A3,950" id44 -->
</div>
<!-- end id44 -->

<a class="method_anchor" id="torch_library_Library_fallback"></a>

<!-- npu="910b,A3" id41 -->
> <font size="3">fallback()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.library.Library.fallback](https://pytorch.org/docs/2.9/library.html#torch.library.Library.fallback)

**产品支持情况**：
<!-- end id41 -->

<!-- npu="910b" id29 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="A3" id30 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id30 -->

<!-- npu="910b,A3" id42 -->
</div>
<!-- end id42 -->

<a class="method_anchor" id="torch_library_Library_impl"></a>

<!-- npu="910b,A3,950" id39 -->
> <font size="3">impl()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.library.Library.impl](https://pytorch.org/docs/2.9/library.html#torch.library.Library.impl)

**产品支持情况**：
<!-- end id39 -->

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id37 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id37 -->

<!-- npu="910b,A3,950" id40 -->
</div>
<!-- end id40 -->

</div>

### torch.library.define

<div style="margin-left: 2em">

**原生文档**：[torch.library.define](https://pytorch.org/docs/2.9/library.html#torch.library.define)

**产品支持情况**：

<!-- npu="910b" id5 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="A3" id6 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id6 -->

</div>

### torch.library.impl

<div style="margin-left: 2em">

**原生文档**：[torch.library.impl](https://pytorch.org/docs/2.9/library.html#torch.library.impl)

**产品支持情况**：

<!-- npu="910b" id9 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id9 -->
<!-- npu="A3" id10 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id10 -->

<!-- npu="950" id38 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id38 -->

</div>

### torch.library.fallthrough_kernel

<div style="margin-left: 2em">

**原生文档**：[torch.library.fallthrough_kernel](https://pytorch.org/docs/2.9/library.html#torch.library.fallthrough_kernel)

**产品支持情况**：

<!-- npu="910b" id33 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id33 -->
<!-- npu="A3" id34 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id34 -->

</div>
