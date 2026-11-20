# torch.utils.deterministic

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 本支持清单的范围以[PyTorch社区文档](https://pytorch.org/docs/2.14/)为准，社区文档中未列出的接口不在本清单范围内；已在社区文档中存在但未收录于本清单的API尚未验证，请谨慎使用，我们将在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.14/deterministic.html)。

<div style="display:none;">

## &#8203;torch.utils.deterministic

</div>

### torch.utils.deterministic.fill_uninitialized_memory

<div style="margin-left: 2em">

**原生文档**：[torch.utils.deterministic.fill_uninitialized_memory](https://pytorch.org/docs/2.14/deterministic.html#torch.utils.deterministic.fill_uninitialized_memory)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id3 -->

**限制与说明**：读取 `torch.utils.deterministic.fill_uninitialized_memory` 时，PyTorch 的默认值为 True。TorchNPU 的设备填充开关默认关闭。要在 NPU 上启用填充，调用 `torch.use_deterministic_algorithms(True)` 并显式赋值 `torch.utils.deterministic.fill_uninitialized_memory = True`。

</div>
