# torch.nn.init

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.12/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.12/nn.init.html)。

<div style="display:none;">

## &#8203;torch.nn.init

</div>

### torch.nn.init.calculate_gain

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.calculate_gain](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.calculate_gain)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### torch.nn.init.uniform_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.uniform_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.uniform_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp32

</div>

### torch.nn.init.normal_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.normal_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.normal_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp32

</div>

### torch.nn.init.constant_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.constant_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.constant_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp32

</div>

### torch.nn.init.ones_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.ones_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.ones_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp32

</div>

### torch.nn.init.zeros_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.zeros_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.zeros_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp32

</div>

### torch.nn.init.eye_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.eye_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.eye_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp32

</div>

### torch.nn.init.dirac_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.dirac_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.dirac_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`input`仅支持fp32

</div>

### torch.nn.init.xavier_uniform_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.xavier_uniform_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.xavier_uniform_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp32

</div>

### torch.nn.init.xavier_normal_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.xavier_normal_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.xavier_normal_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp32

</div>

### torch.nn.init.kaiming_uniform_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.kaiming_uniform_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.kaiming_uniform_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp32

</div>

### torch.nn.init.kaiming_normal_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.kaiming_normal_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.kaiming_normal_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**：`input`仅支持fp32

</div>

### torch.nn.init.trunc_normal_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.trunc_normal_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.trunc_normal_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.nn.init.orthogonal_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.orthogonal_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.orthogonal_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`input`仅支持fp32

</div>

### torch.nn.init.sparse_

<div style="margin-left: 2em">

**原生文档**：[torch.nn.init.sparse_](https://pytorch.org/docs/2.12/nn.init.html#torch.nn.init.sparse_)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**：`input`仅支持fp32

</div>
