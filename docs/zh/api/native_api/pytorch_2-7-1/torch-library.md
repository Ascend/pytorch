# torch.library

> [!NOTE]
>
> - API的**支持情况**中，&#10004;表示API支持在对应硬件环境上运行，&#10007;表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。

## 目录

- [Extending custom ops (created from Python or C++)](#extending-custom-ops-created-from-python-or-c)
- [Low-level APIs](#low-level-apis)

## Extending custom ops (created from Python or C++)

### <code><i>class</i></code> torch.library.Library

<div style="margin-left: 2em">

**原生文档**：[torch.library.Library](https://pytorch.org/docs/2.7/library.html#torch.library.Library)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10004; |

> <font size="3">define()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.library.Library.define](https://pytorch.org/docs/2.7/library.html#torch.library.Library.define)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

> <font size="3">impl()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.library.Library.impl](https://pytorch.org/docs/2.7/library.html#torch.library.Library.impl)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>

</div>

## Low-level APIs

### torch.library.fallthrough_kernel

<div style="margin-left: 2em">

**原生文档**：[torch.library.fallthrough_kernel](https://pytorch.org/docs/2.7/library.html#torch.library.fallthrough_kernel)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | &#10004; |
| <term>Atlas A3 训练系列产品</term> | &#10004; |
| <term>Ascend 950DT</term> | &#10007; |

</div>
