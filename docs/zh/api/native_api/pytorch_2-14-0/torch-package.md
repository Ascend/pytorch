# torch.package

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.14/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.14/package.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [API Reference](#api-reference)

</div>

<div style="display:none;">

## &#8203;torch.package

</div>

## API Reference

### <code><i>class</i></code> torch.package.EmptyMatchError

<div style="margin-left: 2em">

**原生文档**：[torch.package.EmptyMatchError](https://pytorch.org/docs/2.14/package.html#torch.package.EmptyMatchError)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id3 -->

</div>

### <code><i>class</i></code> torch.package.PackagingError

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackagingError](https://pytorch.org/docs/2.14/package.html#torch.package.PackagingError)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id6 -->

</div>

### <code><i>class</i></code> torch.package.PackageExporter

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id9 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.\_\_init\_\_](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.__init__)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id12 -->

</div>

> <font size="3">add_dependency()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.add_dependency](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.add_dependency)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id15 -->

</div>

> <font size="3">all_paths()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.all_paths](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.all_paths)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id18 -->

</div>

> <font size="3">close()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.close](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.close)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id21 -->

</div>

> <font size="3">denied_modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.denied_modules](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.denied_modules)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id24 -->

</div>

> <font size="3">deny()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.deny](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.deny)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id27 -->

</div>

> <font size="3">dependency_graph_string()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.dependency_graph_string](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.dependency_graph_string)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id30 -->

</div>

> <font size="3">extern()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.extern](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.extern)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id33 -->

</div>

> <font size="3">externed_modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.externed_modules](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.externed_modules)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id36 -->

</div>

> <font size="3">get_rdeps()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.get_rdeps](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.get_rdeps)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id39 -->

</div>

> <font size="3">get_unique_id()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.get_unique_id](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.get_unique_id)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id42 -->

</div>

> <font size="3">intern()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.intern](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.intern)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id45 -->

</div>

> <font size="3">interned_modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.interned_modules](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.interned_modules)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id48 -->

</div>

> <font size="3">mock()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.mock](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.mock)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id51 -->

</div>

> <font size="3">mocked_modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.mocked_modules](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.mocked_modules)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id54 -->

</div>

> <font size="3">register_extern_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.register_extern_hook](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.register_extern_hook)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id57 -->

</div>

> <font size="3">register_intern_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.register_intern_hook](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.register_intern_hook)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id60 -->

</div>

> <font size="3">register_mock_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.register_mock_hook](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.register_mock_hook)

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id63 -->

</div>

> <font size="3">save_binary()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.save_binary](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.save_binary)

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id66 -->

</div>

> <font size="3">save_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.save_module](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.save_module)

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id69 -->

</div>

> <font size="3">save_pickle()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.save_pickle](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.save_pickle)

**产品支持情况**：

<!-- npu="910b" id70 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id72 -->

</div>

> <font size="3">save_source_file()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.save_source_file](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.save_source_file)

**产品支持情况**：

<!-- npu="910b" id73 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id75 -->

</div>

> <font size="3">save_source_string()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.save_source_string](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.save_source_string)

**产品支持情况**：

<!-- npu="910b" id76 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id78 -->

</div>

> <font size="3">save_text()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.save_text](https://pytorch.org/docs/2.14/package.html#torch.package.PackageExporter.save_text)

**产品支持情况**：

<!-- npu="910b" id79 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id81 -->

</div>

</div>

### <code><i>class</i></code> torch.package.PackageImporter

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageImporter](https://pytorch.org/docs/2.14/package.html#torch.package.PackageImporter)

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id84 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageImporter.\_\_init\_\_](https://pytorch.org/docs/2.14/package.html#torch.package.PackageImporter.__init__)

**产品支持情况**：

<!-- npu="910b" id85 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id87 -->

</div>

> <font size="3">file_structure()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageImporter.file_structure](https://pytorch.org/docs/2.14/package.html#torch.package.PackageImporter.file_structure)

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id90 -->

</div>

> <font size="3">id()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageImporter.id](https://pytorch.org/docs/2.14/package.html#torch.package.PackageImporter.id)

**产品支持情况**：

<!-- npu="910b" id91 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id93 -->

</div>

> <font size="3">import_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageImporter.import_module](https://pytorch.org/docs/2.14/package.html#torch.package.PackageImporter.import_module)

**产品支持情况**：

<!-- npu="910b" id94 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id96 -->

</div>

> <font size="3">load_binary()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageImporter.load_binary](https://pytorch.org/docs/2.14/package.html#torch.package.PackageImporter.load_binary)

**产品支持情况**：

<!-- npu="910b" id97 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id99 -->

</div>

> <font size="3">load_pickle()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageImporter.load_pickle](https://pytorch.org/docs/2.14/package.html#torch.package.PackageImporter.load_pickle)

**产品支持情况**：

<!-- npu="910b" id100 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id102 -->

</div>

> <font size="3">load_text()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageImporter.load_text](https://pytorch.org/docs/2.14/package.html#torch.package.PackageImporter.load_text)

**产品支持情况**：

<!-- npu="910b" id103 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id105 -->

</div>

> <font size="3">python_version()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageImporter.python_version](https://pytorch.org/docs/2.14/package.html#torch.package.PackageImporter.python_version)

**产品支持情况**：

<!-- npu="910b" id106 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id108 -->

</div>

</div>

### <code><i>class</i></code> torch.package.Directory

<div style="margin-left: 2em">

**原生文档**：[torch.package.Directory](https://pytorch.org/docs/2.14/package.html#torch.package.Directory)

**产品支持情况**：

<!-- npu="910b" id109 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id111 -->

> <font size="3">has_file()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.Directory.has_file](https://pytorch.org/docs/2.14/package.html#torch.package.Directory.has_file)

**产品支持情况**：

<!-- npu="910b" id112 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id114 -->

</div>

</div>
