# torch.package

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.12/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.12/package.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

- [API Reference](#api-reference)

</div>

<div style="display:none;">

## &#8203;torch.package

</div>

## API Reference

### <code><i>class</i></code> torch.package.EmptyMatchError

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.EmptyMatchError](https://pytorch.org/docs/2.12/package.html#torch.package.EmptyMatchError)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id3 -->

</div>

### <code><i>class</i></code> torch.package.PackagingError

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackagingError](https://pytorch.org/docs/2.12/package.html#torch.package.PackagingError)

**Support Status**:

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id6 -->

</div>

### <code><i>class</i></code> torch.package.PackageExporter

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter)

**Support Status**:

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id9 -->

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.\_\_init\_\_](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.__init__)

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id12 -->

</div>

> <font size="3">add_dependency()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.add_dependency](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.add_dependency)

**Support Status**:

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id15 -->

</div>

> <font size="3">all_paths()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.all_paths](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.all_paths)

**Support Status**:

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id18 -->

</div>

> <font size="3">close()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.close](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.close)

**Support Status**:

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id21 -->

</div>

> <font size="3">denied_modules()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.denied_modules](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.denied_modules)

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id24 -->

</div>

> <font size="3">deny()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.deny](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.deny)

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id27 -->

</div>

> <font size="3">dependency_graph_string()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.dependency_graph_string](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.dependency_graph_string)

**Support Status**:

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id30 -->

</div>

> <font size="3">extern()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.extern](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.extern)

**Support Status**:

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id33 -->

</div>

> <font size="3">externed_modules()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.externed_modules](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.externed_modules)

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id36 -->

</div>

> <font size="3">get_rdeps()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.get_rdeps](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.get_rdeps)

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id39 -->

</div>

> <font size="3">get_unique_id()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.get_unique_id](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.get_unique_id)

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id42 -->

</div>

> <font size="3">intern()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.intern](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.intern)

**Support Status**:

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id45 -->

</div>

> <font size="3">interned_modules()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.interned_modules](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.interned_modules)

**Support Status**:

<!-- npu="910b" id46 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id48 -->

</div>

> <font size="3">mock()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.mock](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.mock)

**Support Status**:

<!-- npu="910b" id49 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id51 -->

</div>

> <font size="3">mocked_modules()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.mocked_modules](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.mocked_modules)

**Support Status**:

<!-- npu="910b" id52 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id54 -->

</div>

> <font size="3">register_extern_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.register_extern_hook](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.register_extern_hook)

**Support Status**:

<!-- npu="910b" id55 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id57 -->

</div>

> <font size="3">register_intern_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.register_intern_hook](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.register_intern_hook)

**Support Status**:

<!-- npu="910b" id58 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id60 -->

</div>

> <font size="3">register_mock_hook()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.register_mock_hook](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.register_mock_hook)

**Support Status**:

<!-- npu="910b" id61 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id63 -->

</div>

> <font size="3">save_binary()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.save_binary](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.save_binary)

**Support Status**:

<!-- npu="910b" id64 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id66 -->

</div>

> <font size="3">save_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.save_module](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.save_module)

**Support Status**:

<!-- npu="910b" id67 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id69 -->

</div>

> <font size="3">save_pickle()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.save_pickle](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.save_pickle)

**Support Status**:

<!-- npu="910b" id70 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id72 -->

</div>

> <font size="3">save_source_file()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.save_source_file](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.save_source_file)

**Support Status**:

<!-- npu="910b" id73 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id75 -->

</div>

> <font size="3">save_source_string()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.save_source_string](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.save_source_string)

**Support Status**:

<!-- npu="910b" id76 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id78 -->

</div>

> <font size="3">save_text()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageExporter.save_text](https://pytorch.org/docs/2.12/package.html#torch.package.PackageExporter.save_text)

**Support Status**:

<!-- npu="910b" id79 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id81 -->

</div>

</div>

### <code><i>class</i></code> torch.package.PackageImporter

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageImporter](https://pytorch.org/docs/2.12/package.html#torch.package.PackageImporter)

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

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageImporter.\_\_init\_\_](https://pytorch.org/docs/2.12/package.html#torch.package.PackageImporter.__init__)

**Support Status**:

<!-- npu="910b" id85 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id87 -->

</div>

> <font size="3">file_structure()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageImporter.file_structure](https://pytorch.org/docs/2.12/package.html#torch.package.PackageImporter.file_structure)

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

> <font size="3">id()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageImporter.id](https://pytorch.org/docs/2.12/package.html#torch.package.PackageImporter.id)

**Support Status**:

<!-- npu="910b" id91 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id93 -->

</div>

> <font size="3">import_module()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageImporter.import_module](https://pytorch.org/docs/2.12/package.html#torch.package.PackageImporter.import_module)

**Support Status**:

<!-- npu="910b" id94 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id96 -->

</div>

> <font size="3">load_binary()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageImporter.load_binary](https://pytorch.org/docs/2.12/package.html#torch.package.PackageImporter.load_binary)

**Support Status**:

<!-- npu="910b" id97 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id99 -->

</div>

> <font size="3">load_pickle()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageImporter.load_pickle](https://pytorch.org/docs/2.12/package.html#torch.package.PackageImporter.load_pickle)

**Support Status**:

<!-- npu="910b" id100 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id102 -->

</div>

> <font size="3">load_text()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageImporter.load_text](https://pytorch.org/docs/2.12/package.html#torch.package.PackageImporter.load_text)

**Support Status**:

<!-- npu="910b" id103 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id105 -->

</div>

> <font size="3">python_version()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.PackageImporter.python_version](https://pytorch.org/docs/2.12/package.html#torch.package.PackageImporter.python_version)

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

</div>

### <code><i>class</i></code> torch.package.Directory

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.Directory](https://pytorch.org/docs/2.12/package.html#torch.package.Directory)

**Support Status**:

<!-- npu="910b" id109 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id111 -->

> <font size="3">has_file()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.package.Directory.has_file](https://pytorch.org/docs/2.12/package.html#torch.package.Directory.has_file)

**Support Status**:

<!-- npu="910b" id112 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id114 -->

</div>

</div>
