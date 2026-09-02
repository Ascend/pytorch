# torch.utils.cpp\_extension

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.7/cpp_extension.html)。

<div style="display:none;">

## &#8203;torch.utils.cpp\_extension

</div>

### <code><i>class</i></code> torch.utils.cpp_extension.CppExtension

<div style="margin-left: 2em">

**原生文档**：[torch.utils.cpp_extension.CppExtension](https://pytorch.org/docs/2.7/cpp_extension.html#torch.utils.cpp_extension.CppExtension)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.utils.cpp_extension.CUDAExtension

<div style="margin-left: 2em">

**原生文档**：[torch.utils.cpp_extension.CUDAExtension](https://pytorch.org/docs/2.7/cpp_extension.html#torch.utils.cpp_extension.CUDAExtension)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.utils.cpp_extension.BuildExtension

<div style="margin-left: 2em">

**原生文档**：[torch.utils.cpp_extension.BuildExtension](https://pytorch.org/docs/2.7/cpp_extension.html#torch.utils.cpp_extension.BuildExtension)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.utils.cpp_extension.load

<div style="margin-left: 2em">

**原生文档**：[torch.utils.cpp_extension.load](https://pytorch.org/docs/2.7/cpp_extension.html#torch.utils.cpp_extension.load)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✘ |
| <term>Atlas A3 训练系列产品</term> | ✘ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.utils.cpp_extension.load_inline

<div style="margin-left: 2em">

**原生文档**：[torch.utils.cpp_extension.load_inline](https://pytorch.org/docs/2.7/cpp_extension.html#torch.utils.cpp_extension.load_inline)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.utils.cpp_extension.include_paths

<div style="margin-left: 2em">

**原生文档**：[torch.utils.cpp_extension.include_paths](https://pytorch.org/docs/2.7/cpp_extension.html#torch.utils.cpp_extension.include_paths)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version

<div style="margin-left: 2em">

**原生文档**：[torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version](https://pytorch.org/docs/2.7/cpp_extension.html#torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.utils.cpp_extension.verify_ninja_availability

<div style="margin-left: 2em">

**原生文档**：[torch.utils.cpp_extension.verify_ninja_availability](https://pytorch.org/docs/2.7/cpp_extension.html#torch.utils.cpp_extension.verify_ninja_availability)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.utils.cpp_extension.is_ninja_available

<div style="margin-left: 2em">

**原生文档**：[torch.utils.cpp_extension.is_ninja_available](https://pytorch.org/docs/2.7/cpp_extension.html#torch.utils.cpp_extension.is_ninja_available)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>
