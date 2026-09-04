# torch.package

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.7/package.html)。

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

**原生文档**：[torch.package.EmptyMatchError](https://pytorch.org/docs/2.7/package.html#torch.package.EmptyMatchError)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.package.PackagingError

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackagingError](https://pytorch.org/docs/2.7/package.html#torch.package.PackagingError)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.package.PackageExporter

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.\_\_init\_\_](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.__init__)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">add_dependency()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.add_dependency](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.add_dependency)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">all_paths()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.all_paths](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.all_paths)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">close()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.close](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.close)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">denied_modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.denied_modules](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.denied_modules)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">deny()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.deny](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.deny)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">dependency_graph_string()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.dependency_graph_string](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.dependency_graph_string)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">extern()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.extern](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.extern)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">externed_modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.externed_modules](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.externed_modules)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">get_rdeps()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.get_rdeps](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.get_rdeps)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">get_unique_id()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.get_unique_id](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.get_unique_id)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">intern()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.intern](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.intern)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">interned_modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.interned_modules](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.interned_modules)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">mock()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.mock](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.mock)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">mocked_modules()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.mocked_modules](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.mocked_modules)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">register_extern_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.register_extern_hook](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.register_extern_hook)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">register_intern_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.register_intern_hook](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.register_intern_hook)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">register_mock_hook()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.register_mock_hook](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.register_mock_hook)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">save_binary()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.save_binary](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.save_binary)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">save_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.save_module](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.save_module)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">save_pickle()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.save_pickle](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.save_pickle)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">save_source_file()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.save_source_file](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.save_source_file)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">save_source_string()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.save_source_string](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.save_source_string)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">save_text()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageExporter.save_text](https://pytorch.org/docs/2.7/package.html#torch.package.PackageExporter.save_text)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.package.PackageImporter

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageImporter](https://pytorch.org/docs/2.7/package.html#torch.package.PackageImporter)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">\_\_init\_\_()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageImporter.\_\_init\_\_](https://pytorch.org/docs/2.7/package.html#torch.package.PackageImporter.__init__)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">file_structure()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageImporter.file_structure](https://pytorch.org/docs/2.7/package.html#torch.package.PackageImporter.file_structure)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">id()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageImporter.id](https://pytorch.org/docs/2.7/package.html#torch.package.PackageImporter.id)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">import_module()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageImporter.import_module](https://pytorch.org/docs/2.7/package.html#torch.package.PackageImporter.import_module)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">load_binary()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageImporter.load_binary](https://pytorch.org/docs/2.7/package.html#torch.package.PackageImporter.load_binary)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">load_pickle()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageImporter.load_pickle](https://pytorch.org/docs/2.7/package.html#torch.package.PackageImporter.load_pickle)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">load_text()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageImporter.load_text](https://pytorch.org/docs/2.7/package.html#torch.package.PackageImporter.load_text)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">python_version()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.PackageImporter.python_version](https://pytorch.org/docs/2.7/package.html#torch.package.PackageImporter.python_version)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.package.Directory

<div style="margin-left: 2em">

**原生文档**：[torch.package.Directory](https://pytorch.org/docs/2.7/package.html#torch.package.Directory)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">has_file()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.package.Directory.has_file](https://pytorch.org/docs/2.7/package.html#torch.package.Directory.has_file)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>
