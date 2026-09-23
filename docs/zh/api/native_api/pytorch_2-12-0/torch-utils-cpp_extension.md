# torch.utils.cpp\_extension

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.12/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.12/cpp_extension.html)。

<div style="display:none;">

## &#8203;torch.utils.cpp\_extension

</div>

### <code><i>class</i></code> torch.utils.cpp_extension.CppExtension

<div style="margin-left: 2em">

**原生文档**：[torch.utils.cpp_extension.CppExtension](https://pytorch.org/docs/2.12/cpp_extension.html#torch.utils.cpp_extension.CppExtension)

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

### <code><i>class</i></code> torch.utils.cpp_extension.CUDAExtension

<div style="margin-left: 2em">

**原生文档**：[torch.utils.cpp_extension.CUDAExtension](https://pytorch.org/docs/2.12/cpp_extension.html#torch.utils.cpp_extension.CUDAExtension)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id6 -->

</div>

### <code><i>class</i></code> torch.utils.cpp_extension.BuildExtension

<div style="margin-left: 2em">

**原生文档**：[torch.utils.cpp_extension.BuildExtension](https://pytorch.org/docs/2.12/cpp_extension.html#torch.utils.cpp_extension.BuildExtension)

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

</div>

### torch.utils.cpp_extension.load

<div style="margin-left: 2em">

**原生文档**：[torch.utils.cpp_extension.load](https://pytorch.org/docs/2.12/cpp_extension.html#torch.utils.cpp_extension.load)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id12 -->

</div>

### torch.utils.cpp_extension.load_inline

<div style="margin-left: 2em">

**原生文档**：[torch.utils.cpp_extension.load_inline](https://pytorch.org/docs/2.12/cpp_extension.html#torch.utils.cpp_extension.load_inline)

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

### torch.utils.cpp_extension.include_paths

<div style="margin-left: 2em">

**原生文档**：[torch.utils.cpp_extension.include_paths](https://pytorch.org/docs/2.12/cpp_extension.html#torch.utils.cpp_extension.include_paths)

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

### torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version

<div style="margin-left: 2em">

**原生文档**：[torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version](https://pytorch.org/docs/2.12/cpp_extension.html#torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version)

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

### torch.utils.cpp_extension.verify_ninja_availability

<div style="margin-left: 2em">

**原生文档**：[torch.utils.cpp_extension.verify_ninja_availability](https://pytorch.org/docs/2.12/cpp_extension.html#torch.utils.cpp_extension.verify_ninja_availability)

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

### torch.utils.cpp_extension.is_ninja_available

<div style="margin-left: 2em">

**原生文档**：[torch.utils.cpp_extension.is_ninja_available](https://pytorch.org/docs/2.12/cpp_extension.html#torch.utils.cpp_extension.is_ninja_available)

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
