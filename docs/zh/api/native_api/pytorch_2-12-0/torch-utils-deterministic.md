# torch.utils.deterministic

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.12/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.12/deterministic.html)。

<div style="display:none;">

## &#8203;torch.utils.deterministic

</div>

### torch.utils.deterministic.fill_uninitialized_memory

<div style="margin-left: 2em">

**原生文档**：[torch.utils.deterministic.fill_uninitialized_memory](https://pytorch.org/docs/2.12/deterministic.html#torch.utils.deterministic.fill_uninitialized_memory)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`torch.utils.deterministic.fill_uninitialized_memory`默认值为True。设置`torch.use_deterministic_algorithms`开启确定性后，PyTorch行为是填充未初始化内存，而TorchNPU默认不填充。如需填充生效，需手动将`torch.utils.deterministic.fill_uninitialized_memory`设置为True

</div>
